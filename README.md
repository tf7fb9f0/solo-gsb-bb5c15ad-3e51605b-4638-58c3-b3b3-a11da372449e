# 校准不确定度传播 API

基于 FastAPI + SymPy + Pint 的测量不确定度传播服务（GUM 传播律：符号偏导 + 协方差矩阵）。

## 启动

```bash
docker compose up --build
```

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST_PORT` | `8000` | Compose 宿主机映射端口 |
| `PORT` | `8000` | 容器内服务端口 |

## API

- 端口：`8000`（容器内）
- `POST /v1/propagate` — 不确定度传播计算
- `GET /health` — 健康检查
- 交互文档：`http://localhost:8000/docs`
