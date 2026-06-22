# syntax=docker/dockerfile:1

# Stage 1: Install dependencies
FROM uhub.service.ucloud.cn/tt139/python:3.10-slim AS deps
WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary :all: -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

# Stage 2: Final image (without build tools)
FROM uhub.service.ucloud.cn/tt139/python:3.10-slim
WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY *.py .

# Create necessary directories
RUN mkdir -p chroma_db uploads hf_cache

EXPOSE 8000 8501

CMD ["sh", "-c", "python backend.py & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
