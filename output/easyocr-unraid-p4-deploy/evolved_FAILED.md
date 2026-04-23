---
name: easyocr-unraid-p4-deploy
description: 在 Unraid Tesla P4 (Pascal) 上部署 EasyOCR GPU 版 OCR 服务。关键：必须用 CUDA 11.x 镜像（CUDA 12.x 已放弃 Pascal）；用 pip install 而非 git clone（国内网络）；修改 Dockerfile 后必须 --no-cache 重建（Docker 缓存导致新内容不生效）。
tags: [unraid, docker, easyocr, ocr, gpu, tesla-p4, pascal]
category: mlops
---

# EasyOCR GPU 部署到 Unraid Tesla P4

## 背景

- Tesla P4 是 **Pascal 架构（Compute Capability 6.1）**
- **CUDA 12.x 已放弃对 Pascal 的支持**，必须用 CUDA 11.x
- EasyOCR 中英文约需 **500MB 显存**，比 PaddleOCR 轻很多
- 可与 Infinity 向量服务（占用 53-54% 显存）共存

## 项目文件

```
easyocr-gpu/
├── Dockerfile           # 基于 pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
├── docker-compose.yml  # Unraid Docker Compose 配置
├── api.py              # FastAPI OCR 服务
└── build.sh            # 构建脚本
```

## 部署步骤

### 1. 上传文件到 Unraid

将 `easyocr-gpu/` 目录上传到 `/mnt/user/appdata/easyocr-gpu/`

### 2. Dockerfile 正确内容

> ⚠️ 注意：不要用 git clone 方式安装 EasyOCR，国内网络会卡住。用 pip install。

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-dev libgomp1 \
    curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ 用 pip install，不要 git clone（国内网络会挂）
RUN python -m pip install --no-cache-dir easyocr
RUN python -m pip install --no-cache-dir fastapi uvicorn python-multipart aiofiles opencv-python-headless

COPY api.py /app/api.py

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. 构建镜像（SSH）

> ⚠️ **关键**：修改 Dockerfile 后必须加 `--no-cache`，否则 Docker 会复用旧缓存层，新内容不生效！

```bash
cd /mnt/user/appdata/easyocr-gpu
mkdir -p cache
docker build --no-cache -t easyocr-gpu:latest . 2>&1 | tee build.log
```

### 4. Docker Compose 配置

```yaml
services:
  easyocr-gpu:
    image: easyocr-gpu:latest
    container_name: easyocr-gpu
    restart: unless-stopped
    ports:
      - "8082:8000"
    volumes:
      - /mnt/user/appdata/easyocr-gpu/cache:/root/.EasyOCR
    environment:
      - LANG=C.UTF-8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
```

### 4. 验证

```bash
curl http://localhost:8082/health
# 返回: {"status":"ok"}
```

## API 调用

```bash
curl -X POST http://localhost:8082/ocr \
  -F "image=@/path/to/image.jpg" \
  -F "languages=ch_sim,en"
```

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'easyocr'` 但镜像存在 | Docker 用的是旧镜像缓存，未真正安装 | 修改 Dockerfile 后必须用 `--no-cache` 重新构建 |
| `docker build` 一直卡住不动 | 网络问题（git clone 下载慢） | 换成 `pip install easyocr`，不要 git clone |
| 容器路径出现 `/opt/conda/bin/uvicorn` | 基础镜像错误，用了 miniconda 而非 PyTorch | 确认 FROM 为 `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime` |
| CUDA OOM | 显存不够 | 关闭其他 GPU 服务 |
| 模型下载慢 | 首次运行需下载 | 耐心等待，或预先下载到 cache 目录 |
| `docker build` 报错缺参数 | 命令末尾少 `.` | 应该是 `docker build -t easyocr-gpu:latest .` |

## 为什么不直接用 Docker Hub 上的 easyocr 镜像

- `challisa/easyocr`：最后更新 2020-08-16，已 5 年未维护
- 基础镜像 `pytorch/pytorch`（无 tag）版本未知
- 必须自己用新版 PyTorch (CUDA 11.x) 构建
