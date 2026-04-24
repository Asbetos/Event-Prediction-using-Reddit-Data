#!/usr/bin/env bash
# ============================================================================
# RunPod A100 80GB Environment Setup
# Installs all dependencies for GPU-accelerated Reddit data processing
# Usage: bash setup_runpod.sh
# ============================================================================
set -euo pipefail

echo "============================================================"
echo "  RunPod A100 GPU Pod Setup - Reddit Event Prediction"
echo "============================================================"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ── 0. Base image validation ────────────────────────────────────────────────
echo ""
echo "[0/6] Validating RunPod base image..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')" || {
    echo "  ERROR: PyTorch is not installed in this image."
    echo "  Use a RunPod PyTorch/CUDA template image, then rerun setup."
    exit 1
}

if command -v nvcc >/dev/null 2>&1; then
    echo "  nvcc detected."
else
    echo "  nvcc not found; will detect CUDA version from PyTorch runtime."
fi

echo "  Base image looks usable."

# ── 1. System packages ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq build-essential libfontconfig1 fonts-dejavu-core \
    curl wget git htop nvtop 2>/dev/null || true
echo "  System packages installed."

# ── 2. RAPIDS cuDF and dask-cudf ────────────────────────────────────────────
echo ""
echo "[2/6] Installing RAPIDS cuDF + dask-cudf..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' | tr -d ' ')
else
    CUDA_VERSION=$(python3 -c "import torch; print((torch.version.cuda or '').strip())")
fi

if [ -z "$CUDA_VERSION" ]; then
    echo "  ERROR: Could not determine CUDA version from nvcc or torch.version.cuda"
    exit 1
fi

CUDA_MAJOR=$(echo "$CUDA_VERSION" | head -c2)
echo "  Detected Python ${PYTHON_VERSION}, CUDA ${CUDA_VERSION}"

pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu${CUDA_MAJOR} dask-cudf-cu${CUDA_MAJOR} cuml-cu${CUDA_MAJOR} \
    2>/dev/null || {
    echo "  RAPIDS pip install failed, trying conda..."
    conda install -y -c rapidsai -c conda-forge -c nvidia \
        cudf cuml dask-cudf python=${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1} \
        cuda-version=${CUDA_MAJOR} 2>/dev/null || {
        echo "  WARNING: Could not install RAPIDS. cuDF will fall back to pandas."
        echo "  You may need to install manually for your CUDA version."
    }
}
echo "  RAPIDS installation step complete."

# ── 3. NLP and ML packages ──────────────────────────────────────────────────
echo ""
echo "[3/6] Installing NLP and ML packages..."
pip install -q \
    transformers>=4.36 \
    sentence-transformers>=2.2 \
    bertopic>=0.16 \
    hdbscan>=0.8.33 \
    umap-learn>=0.5.5 \
    xgboost>=2.0 \
    scipy>=1.11

echo "  NLP/ML packages installed."

# ── 4. spaCy with GPU transformer model ────────────────────────────────────
echo ""
echo "[4/6] Installing spaCy + en_core_web_trf..."
pip install -q "spacy[cuda-autodetect]>=3.7" || pip install -q "spacy>=3.7"
python3 -m spacy download en_core_web_trf
echo "  spaCy + transformer model installed."

# ── 5. Data and utility packages ────────────────────────────────────────────
echo ""
echo "[5/6] Installing data/utility packages..."
pip install -q \
    s3fs>=2024.2.0 \
    boto3>=1.34 \
    pyarrow>=14.0 \
    pandas>=2.0 \
    numpy>=1.24 \
    scikit-learn>=1.3 \
    tqdm>=4.66 \
    matplotlib>=3.8

echo "  Data/utility packages installed."

# ── 6. AWS credentials setup ────────────────────────────────────────────────
echo ""
echo "[6/6] Configuring AWS credentials..."
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "  WARNING: AWS_ACCESS_KEY_ID not set. Set it before running pipeline stages."
    echo "  Export these environment variables:"
    echo "    export AWS_ACCESS_KEY_ID=..."
    echo "    export AWS_SECRET_ACCESS_KEY=..."
    echo "    export AWS_SESSION_TOKEN=...  (if using temporary credentials)"
else
    mkdir -p ~/.aws
    cat > ~/.aws/credentials <<CREDS
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
CREDS

    if [ -n "${AWS_SESSION_TOKEN:-}" ]; then
        echo "aws_session_token = ${AWS_SESSION_TOKEN}" >> ~/.aws/credentials
    fi

    cat > ~/.aws/config <<CONFIG
[default]
region = us-east-1
output = json
CONFIG

    chmod 600 ~/.aws/credentials
    echo "  AWS credentials written to ~/.aws/credentials"
fi

# ── Verify installation ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"
python3 -c "
import sys
print(f'Python:      {sys.version}')

try:
    import cudf; print(f'cuDF:        {cudf.__version__}')
except ImportError:
    print('cuDF:        NOT INSTALLED (will fall back to pandas)')

try:
    import cuml; print(f'cuML:        {cuml.__version__}')
except ImportError:
    print('cuML:        NOT INSTALLED (will fall back to sklearn)')

import torch; print(f'PyTorch:     {torch.__version__}  CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers:{transformers.__version__}')
import spacy; print(f'spaCy:       {spacy.__version__}')
import bertopic; print(f'BERTopic:    {bertopic.__version__}')
import s3fs; print(f's3fs:        {s3fs.__version__}')
import pandas; print(f'pandas:      {pandas.__version__}')

if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'VRAM:        {mem_gb:.1f} GB')
"

echo ""
echo "============================================================"
echo "  Setup complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
