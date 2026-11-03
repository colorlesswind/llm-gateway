"""提示词模板的持久化层。

实现取向：

- **SQLAlchemy Core（同步引擎）+ ``asyncio.to_thread``**：不引入 greenlet 依赖，
  SQLite 与 PostgreSQL 共用一套代码；
- 模板正文与元数据分表：``prompt_templates`` 只保存「头指针」，正文放在
  ``prompt_template_versions``，修改正文会新增版本，旧版本仍可按 version 引用，
  保证同一请求重放时结果可复现；
- 所有操作都是幂等的 ``CREATE TABLE IF NOT EXISTS``（``create_all``），
  由 service 层懒加载，避免测试/多 worker 场景依赖 lifespan 时序；
- ``deleted_at`` 软删除，配合「未删除行」的唯一索引，删除后名称可复用。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, TypeVar

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    event,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from src.services.prompts.errors import (
    PromptNameConflict,
    PromptStoreUnavailable,
    PromptVersionConflict,
)

T = TypeVar("T")

metadata = MetaData()

prompt_templates = Table(
    "prompt_templates",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(128), nullable=False),
    Column("name", String(128), nullable=False),
    Column("description", String(512)),
    Column("enabled", Boolean, nullable=False, default=True),
    Column("head_version", Integer, nullable=False, default=1),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    Column("deleted_at", DateTime),
)

prompt_template_versions = Table(
    "prompt_template_versions",
    metadata,
    Column(
        "template_id",
        String(64),
        ForeignKey("prompt_templates.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("version", Integer, primary_key=True),
    # TEXT：PostgreSQL 超过 ~2KB 自动 TOAST（行外 + 压缩）；
    # 若改用 MySQL 需显式换 MEDIUMTEXT（TEXT 上限 64KB）
    Column("content", Text, nullable=False),
    Column("content_bytes", Integer, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("variables", JSON, nullable=False),
    Column("token_estimate", Integer, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

# 同一用户下名称唯一；软删除后名称可复用（partial index 只对 PG/SQLite 生效）
Index(
    "ux_prompt_templates_user_name",
    prompt_templates.c.user_id,
    prompt_templates.c.name,
    unique=True,
    sqlite_where=prompt_templates.c.deleted_at.is_(None),
    postgresql_where=prompt_templates.c.deleted_at.is_(None),
)
Index(
    "ix_prompt_templates_user_updated",
    prompt_templates.c.user_id,
    prompt_templates.c.updated_at,
)


def _create_engine(url: str) -> Engine:
    options: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # 同步引擎会被 to_thread 在不同线程里使用
        options["connect_args"] = {"check_same_thread": False, "timeout": 15}
        if ":memory:" in url:
            # 内存库必须复用同一个连接，否则每次连接都是空库
            options["poolclass"] = StaticPool
    engine = create_engine(url, **options)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragma(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def _utcnow() -> datetime:
    """统一存 naive UTC，读取时再补回时区，规避各数据库 DateTime 的差异。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _merge(template: Mapping[str, Any], version: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": template["id"],
        "user_id": template["user_id"],
        "name": template["name"],
        "description": template["description"],
        "enabled": bool(template["enabled"]),
        "version": version["version"],
        "content": version["content"],
        "content_bytes": version["content_bytes"],
        "content_hash": version["content_hash"],
        "variables": list(version["variables"] or []),
        "token_estimate": version["token_estimate"],
        "created_at": _as_utc(template["created_at"]),
        "updated_at": _as_utc(template["updated_at"]),
    }


class SqlPromptStore:
    """模板存储（SQLite / PostgreSQL 通用）。"""

    def __init__(self, url: str) -> None:
        self._url = url
        self._engine = _create_engine(url)
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    # -- 生命周期 ---------------------------------------------------------
    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            try:
                await asyncio.to_thread(metadata.create_all, self._engine)
            except SQLAlchemyError as exc:
                raise PromptStoreUnavailable(
                    f"模板库初始化失败：{exc}"
                ) from exc
            self._schema_ready = True

    async def ping(self) -> bool:
        try:
            await self.ensure_schema()
            await asyncio.to_thread(self._ping_sync)
        except Exception:  # noqa: BLE001 - 就绪检查只关心结果
            return False
        return True

    async def aclose(self) -> None:
        await asyncio.to_thread(self._engine.dispose)

    # -- CRUD -------------------------------------------------------------
    async def create(
        self,
        *,
        user_id: str,
        prompt_id: str,
        name: str,
        description: str | None,
        enabled: bool,
        version_data: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._run(
            self._create_sync,
            user_id,
            prompt_id,
            name,
            description,
            enabled,
            version_data,
        )

    async def get(
        self, user_id: str, prompt_id: str, *, version: int | None = None
    ) -> dict[str, Any] | None:
        return await self._run(self._get_sync, user_id, prompt_id, version)

    async def list(
        self,
        user_id: str,
        *,
        limit: int,
        offset: int,
        enabled: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        return await self._run(
            self._list_sync, user_id, limit, offset, enabled
        )

    async def update(
        self,
        user_id: str,
        prompt_id: str,
        *,
        changes: dict[str, Any],
        version_data: dict[str, Any] | None,
        expected_version: int | None,
    ) -> dict[str, Any] | None:
        return await self._run(
            self._update_sync,
            user_id,
            prompt_id,
            changes,
            version_data,
            expected_version,
        )

    async def delete(self, user_id: str, prompt_id: str) -> bool:
        return await self._run(self._delete_sync, user_id, prompt_id)

    async def count(self, user_id: str) -> int:
        return await self._run(self._count_sync, user_id)

    # -- 内部：线程池中执行的同步实现 --------------------------------------
    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        await self.ensure_schema()
        try:
            return await asyncio.to_thread(fn, *args)
        except IntegrityError as exc:
            # 唯一约束冲突由具体实现转成业务错误，这里兜底避免泄露 SQL 细节
            raise PromptStoreUnavailable(f"模板库约束冲突：{exc.orig}") from exc
        except SQLAlchemyError as exc:
            raise PromptStoreUnavailable(f"模板库操作失败：{exc}") from exc

    def _ping_sync(self) -> None:
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def _create_sync(
        self,
        user_id: str,
        prompt_id: str,
        name: str,
        description: str | None,
        enabled: bool,
        version_data: dict[str, Any],
    ) -> dict[str, Any]:
        now = _utcnow()
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    insert(prompt_templates).values(
                        id=prompt_id,
                        user_id=user_id,
                        name=name,
                        description=description,
                        enabled=enabled,
                        head_version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.execute(
                    insert(prompt_template_versions).values(
                        template_id=prompt_id,
                        version=1,
                        created_at=now,
                        **version_data,
                    )
                )
        except IntegrityError as exc:
            raise PromptNameConflict(
                f"已存在同名模板：{name}"
            ) from exc

        record = self._get_sync(user_id, prompt_id, None)
        assert record is not None
        return record

    def _get_sync(
        self, user_id: str, prompt_id: str, version: int | None
    ) -> dict[str, Any] | None:
        with self._engine.connect() as conn:
            template = (
                conn.execute(
                    select(prompt_templates).where(
                        prompt_templates.c.id == prompt_id,
                        prompt_templates.c.user_id == user_id,
                        prompt_templates.c.deleted_at.is_(None),
                    )
                )
                .mappings()
                .first()
            )
            if template is None:
                return None
            content = (
                conn.execute(
                    select(prompt_template_versions).where(
                        prompt_template_versions.c.template_id == prompt_id,
                        prompt_template_versions.c.version
                        == (version or template["head_version"]),
                    )
                )
                .mappings()
                .first()
            )
            if content is None:
                return None
            return _merge(template, content)

    def _list_sync(
        self, user_id: str, limit: int, offset: int, enabled: bool | None
    ) -> tuple[list[dict[str, Any]], int]:
        conditions = [
            prompt_templates.c.user_id == user_id,
            prompt_templates.c.deleted_at.is_(None),
        ]
        if enabled is not None:
            conditions.append(prompt_templates.c.enabled.is_(enabled))

        version = prompt_template_versions
        with self._engine.connect() as conn:
            total = conn.execute(
                select(func.count()).select_from(prompt_templates).where(*conditions)
            ).scalar_one()
            rows = (
                conn.execute(
                    select(prompt_templates, version)
                    .select_from(
                        prompt_templates.join(
                            version,
                            and_(
                                version.c.template_id == prompt_templates.c.id,
                                version.c.version == prompt_templates.c.head_version,
                            ),
                        )
                    )
                    .where(*conditions)
                    .order_by(
                        prompt_templates.c.updated_at.desc(),
                        prompt_templates.c.id,
                    )
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
        items = [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "name": row["name"],
                "description": row["description"],
                "enabled": bool(row["enabled"]),
                "version": row["version"],
                "content_hash": row["content_hash"],
                "variables": list(row["variables"] or []),
                "token_estimate": row["token_estimate"],
                "created_at": _as_utc(row["created_at"]),
                "updated_at": _as_utc(row["updated_at"]),
            }
            for row in rows
        ]
        return items, int(total)

    def _update_sync(
        self,
        user_id: str,
        prompt_id: str,
        changes: dict[str, Any],
        version_data: dict[str, Any] | None,
        expected_version: int | None,
    ) -> dict[str, Any] | None:
        version = prompt_template_versions
        now = _utcnow()
        try:
            with self._engine.begin() as conn:
                template = (
                    conn.execute(
                        select(prompt_templates).where(
                            prompt_templates.c.id == prompt_id,
                            prompt_templates.c.user_id == user_id,
                            prompt_templates.c.deleted_at.is_(None),
                        )
                    )
                    .mappings()
                    .first()
                )
                if template is None:
                    return None
                if (
                    expected_version is not None
                    and expected_version != template["head_version"]
                ):
                    raise PromptVersionConflict(
                        "模板已被更新，请重新拉取后再提交",
                        details={
                            "expected_version": expected_version,
                            "head_version": template["head_version"],
                        },
                    )

                head = template["head_version"]
                values: dict[str, Any] = dict(changes)

                if version_data is not None:
                    head_hash = conn.execute(
                        select(version.c.content_hash).where(
                            version.c.template_id == prompt_id,
                            version.c.version == head,
                        )
                    ).scalar_one_or_none()
                    # 正文未变化时只更新元数据，避免版本号无意义地膨胀
                    if version_data["content_hash"] != head_hash:
                        head += 1
                        conn.execute(
                            insert(version).values(
                                template_id=prompt_id,
                                version=head,
                                created_at=now,
                                **version_data,
                            )
                        )
                values["head_version"] = head
                values["updated_at"] = now
                conn.execute(
                    update(prompt_templates)
                    .where(prompt_templates.c.id == prompt_id)
                    .values(**values)
                )
        except IntegrityError as exc:
            raise PromptNameConflict("该名称已被同用户的另一个模板占用") from exc

        return self._get_sync(user_id, prompt_id, None)

    def _delete_sync(self, user_id: str, prompt_id: str) -> bool:
        now = _utcnow()
        with self._engine.begin() as conn:
            result = conn.execute(
                update(prompt_templates)
                .where(
                    prompt_templates.c.id == prompt_id,
                    prompt_templates.c.user_id == user_id,
                    prompt_templates.c.deleted_at.is_(None),
                )
                .values(deleted_at=now, enabled=False, updated_at=now)
            )
        return bool(result.rowcount)

    def _count_sync(self, user_id: str) -> int:
        with self._engine.connect() as conn:
            return int(
                conn.execute(
                    select(func.count())
                    .select_from(prompt_templates)
                    .where(
                        prompt_templates.c.user_id == user_id,
                        prompt_templates.c.deleted_at.is_(None),
                    )
                ).scalar_one()
            )


__all__ = ["SqlPromptStore", "metadata"]
