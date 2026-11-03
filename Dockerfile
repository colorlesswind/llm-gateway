# syntax=docker/dockerfile:1

# LLM Gateway 运行镜像
#
# 构建：docker build -t llm-gateway:latest .
# 运行：docker run --rm -p 8000:8000 --env-file .env llm-gateway:latest
#
# 配置全部通过环境变量注入（进程环境变量优先于 .env 文件），镜像内不含密钥。

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # uvicorn 监听端口
    PORT=8000 \
    # 进程数：MAX_CONCURRENT_REQUESTS 是进程内计数，多 worker 时总上限要按倍数理解
    UVICORN_WORKERS=1

WORKDIR /app

# 依赖单独一层：只改业务代码时不会触发重装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 非 root 运行；/data 预留给 SQLite（提示词模板库）挂载点
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data

COPY --chown=app:app src ./src
COPY --chown=app:app schemas ./schemas

USER app

EXPOSE 8000

# 存活探针用 /health：/ready 在未配置 DATABASE_URL 时会返回 503
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=3)"

# exec 让 uvicorn 成为 PID 1，SIGTERM 才能触发优雅停机
# UVICORN_EXTRA_ARGS 用于追加参数，例如 "--proxy-headers --forwarded-allow-ips=*"
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1} ${UVICORN_EXTRA_ARGS:-}"]
