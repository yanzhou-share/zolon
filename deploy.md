# 生产环境部署方案

## 当前状态

| 项目 | 状态 | 问题 |
|------|------|------|
| Dockerfile | ✅ 已有 | 单容器运行两个服务 |
| docker-compose | ✅ 已有 | 缺少 Nginx/Redis/监控 |
| .env 配置 | ✅ 已有 | 缺少生产环境配置 |
| 健康检查 | ✅ 已有 | 无探针配置 |
| 日志 | ⚠️ 基础 | 无日志收集 |

---

## 推荐部署架构

### 方案 A：单机 Docker Compose（小型部署）

```
┌─────────────────────────────────────┐
│            Nginx (80/443)            │
│         反向代理 + SSL + 静态文件     │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼───┐           ┌────▼────┐
│ 后端  │           │  前端   │
│ :8000 │           │  :8501  │
└───┬───┘           └─────────┘
    │
┌───▼───────────────────────┐
│        Redis (6379)       │
│   会话缓存 + 限流 + 队列   │
└───────────────────────────┘
```

### 方案 B：云服务部署（中型部署）

```
┌─────────────────────────────────────┐
│         阿里云/腾讯云 CDN            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         负载均衡 (SLB/CLB)          │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼───┐           ┌────▼────┐
│ ECS-1 │           │ ECS-2   │
│ 后端  │           │ 后端    │
└───┬───┘           └────┬────┘
    │                     │
┌───▼─────────────────────▼───┐
│     RDS (MySQL/PostgreSQL)   │
│     + Redis (会话/限流)       │
└─────────────────────────────┘
```

---

## 实施步骤

### 1. 优化 Dockerfile

```dockerfile
# 添加非 root 用户
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### 2. Nginx 配置

```nginx
upstream backend {
    server backend:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # SSL
    ssl_certificate /etc/ssl/certs/app.crt;
    ssl_certificate_key /etc/ssl/private/app.key;
    
    # 后端 API
    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket
    location /chat/stream {
        proxy_pass http://backend/chat/stream;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
    }
    
    # 前端
    location / {
        proxy_pass http://frontend:8501;
    }
}
```

### 3. Redis 缓存（可选）

```python
# 替代内存限流
import redis
r = redis.Redis(host='redis', port=6379, db=0)

def check_rate_limit(api_key: str, limit: int = 100) -> bool:
    key = f"rate:{api_key}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, 60)
    return current <= limit
```

### 4. 生产环境 .env

```env
# 安全
ADMIN_SECRET=<strong-random-password>
DEEPSEEK_API_KEY=<your-key>

# 性能
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.3
CHUNK_SIZE=500

# 日志
LOG_LEVEL=INFO

# CORS（生产环境限制）
CORS_ORIGINS=https://your-domain.com
```

---

## 部署命令

### 方案 A：单机部署

```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

### 方案 B：云服务部署

```bash
# 1. 推送镜像到阿里云 ACR
docker tag ai-sales-agent:latest registry.cn-hangzhou.aliyuncs.com/your-ns/ai-sales-agent:v1.0
docker push registry.cn-hangzhou.aliyuncs.com/your-ns/ai-sales-agent:v1.0

# 2. ECS 上拉取运行
docker pull registry.cn-hangzhou.aliyuncs.com/your-ns/ai-sales-agent:v1.0
docker run -d --name ai-sales-agent -p 8000:8000 -v /data/chroma_db:/app/chroma_db -e DEEPSEEK_API_KEY=xxx ai-sales-agent:v1.0
```

---

## 监控建议

| 层级 | 工具 | 说明 |
|------|------|------|
| 应用监控 | Prometheus + Grafana | 请求量、延迟、错误率 |
| 日志收集 | ELK / Loki | 集中日志 |
| 告警 | AlertManager | 服务异常告警 |
| 链路追踪 | Jaeger / Zipkin | 请求链路追踪 |

---

## 安全清单

- [ ] 更改默认 ADMIN_SECRET
- [ ] 配置 HTTPS/SSL
- [ ] 限制 CORS origins
- [ ] 配置防火墙规则
- [ ] 定期备份 ChromaDB 和 SQLite
- [ ] 启用日志审计
