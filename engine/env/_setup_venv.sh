#!/bin/bash
set -e
cd /home/dongjukim/Documents/workspace/tmp/SST14-M_scr
echo "[1/3] conda create --prefix ./.venv python=3.12"
conda create --yes --prefix ./.venv python=3.12
echo "[2/3] pip install core deps"
./.venv/bin/pip install --no-input \
  "fastapi==0.135.1" "uvicorn[standard]==0.41.0" "pydantic==2.12.5" \
  "requests==2.32.5" "biopython>=1.79" "python-multipart>=0.0.22" \
  "pytest>=7.0" "numpy" "pyyaml" "ollama"
echo "[3/3] done"
./.venv/bin/python -c "import fastapi,uvicorn,pydantic,Bio,numpy,yaml,requests; print('core deps OK')"
