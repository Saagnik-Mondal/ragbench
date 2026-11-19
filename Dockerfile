FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 RAGBENCH_DATA_DIR=/app/data
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[llm]"

# Warm the dataset and model caches at build time so the container starts fast.
RUN python - <<'EOF'
from ragbench.data import load_scifact
from ragbench.retrievers import FastEmbedder, build_retrievers, cross_encoder_scorer
from pathlib import Path
c = load_scifact(Path("/app/data"))
build_retrievers(c, FastEmbedder(), cross_encoder_scorer(), cache_dir=Path("/app/data/cache"))
EOF

EXPOSE 8000
CMD ["ragbench-serve"]
