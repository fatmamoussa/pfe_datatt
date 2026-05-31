# =============================================================
# Dockerfile — Chatbot Tunisie Telecom — Qwen1.5-1.8B
# Base : Python 3.11 slim + CUDA 12.1 (Azure GPU NC series)
# =============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── Variables d'environnement système ────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Africa/Tunis

# ── Dépendances système ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        build-essential \
        git \
        curl \
        libgomp1 \
        libsqlite3-0 \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3    /usr/bin/python

# ── Répertoire de travail ─────────────────────────────────────
WORKDIR /app

# ── Upgrade pip ───────────────────────────────────────────────
RUN python -m pip install --upgrade pip setuptools wheel

# ── PyTorch CUDA 12.1 (installé séparément pour le cache Docker)
RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Dépendances Python ────────────────────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Code applicatif ───────────────────────────────────────────
COPY . .

# ── Création des répertoires persistants ─────────────────────
RUN mkdir -p /app/data \
             /app/models \
             /app/chroma_tt_db \
             /app/static \
             /app/logs

# ── Volumes (montés depuis Azure File Share ou ACI volumes) ──
# /app/data         → base SQLite + fichiers JSON
# /app/models       → modèle Qwen fusionné
# /app/chroma_tt_db → index ChromaDB
VOLUME ["/app/data", "/app/models", "/app/chroma_tt_db"]

# ── Variables d'environnement par défaut (override via ACI/AKS)
ENV MODEL_DIR_QWEN=/app/models/qwen15-1b8-tt-merged \
    CHROMA_DB_DIR=/app/chroma_tt_db \
    COLLECTION_NAME=tt_train \
    DB_PATH=/app/data/chatbot.db \
    STATIC_DIR=/app/static \
    API_PORT=8002 \
    EXTRACTIVE_MODE=false \
    TOP_K=3 \
    MAX_NEW_TOKENS=150 \
    ADMIN_SCORE_BOOST=0.10 \
    ADMIN_DIRECT_THRESHOLD=0.72

# ── Port exposé ───────────────────────────────────────────────
EXPOSE 8002

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "api_qwen:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "120"]