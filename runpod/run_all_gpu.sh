#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════
#  RunPod GPU Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════
#
#  Runs all GPU stages with optimal GPU/CPU parallelism:
#
#  Phase 0: Pre-check GPU state + existing outputs
#  Phase 1: Stage 1 — GPU aggregation (cuDF on A100)
#  Phase 2: Pre-cache S3 text data to /workspace (CPU, 16 threads)
#  Phase 3: GPU track (6→7→8) + CPU track (10,11) IN PARALLEL
#  Phase 4: Stage 9 — classification (needs results from 6,7,8)
#
#  GPU track runs NLP stages sequentially (each needs GPU memory).
#  CPU track runs ML stages in parallel using sklearn on EPYC cores.
#  This keeps BOTH the A100 and the 128-core EPYC busy simultaneously.
#
#  Safety: uses FORCE_CPU=1 for stages 10/11 so they don't compete
#  for GPU memory with the NLP stages or other pods processes.
#
#  Usage:
#    cd /workspace/Event-Prediction-using-Reddit-Data/runpod
#    bash run_all_gpu.sh
#
# ══════════════════════════════════════════════════════════════════

PROJECT_DIR="/workspace/Event-Prediction-using-Reddit-Data"
RUNPOD_DIR="$PROJECT_DIR/runpod"
LOG_DIR="/workspace/logs"

mkdir -p "$LOG_DIR"
cd "$RUNPOD_DIR"

echo "══════════════════════════════════════════════════════════"
echo "  RunPod GPU Pipeline — Full Run"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════"

# ── Phase 0: Pre-check ─────────────────────────────────────────
echo ""
echo "[Phase 0] GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""
echo "[Phase 0] Disk space on /workspace:"
df -h /workspace 2>/dev/null | tail -1
echo ""

# ── Phase 1: Aggregation ──────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Phase 1] Stage 1: GPU Aggregation (cuDF)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python stage1_aggregate_gpu.py 2>&1 | tee "$LOG_DIR/stage1.out"
echo "[Phase 1] Stage 1 complete."

# ── Phase 2: Pre-cache S3 text data ───────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Phase 2] Pre-caching S3 text to /workspace/s3_cache/..."
echo "  (16 parallel threads, CPU-only, no GPU needed)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python s3_text_cache.py 2>&1 | tee "$LOG_DIR/s3_cache.out"
echo "[Phase 2] S3 cache ready."

# ── Phase 3: GPU NLP + CPU ML in parallel ─────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Phase 3] Parallel execution:"
echo "  GPU track: Stage 6 (NER) → Stage 7 (Sentiment) → Stage 8 (Topics)"
echo "  CPU track: Stage 10 (Sustain) + Stage 11 (Forecast) in parallel"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# GPU track: NLP stages run sequentially (each needs GPU memory)
(
    echo "[GPU] Starting Stage 6: NER..."
    python stage6_ner_gpu.py 2>&1 | tee "$LOG_DIR/stage6.out"
    echo "[GPU] Stage 6 complete. Starting Stage 7: Sentiment..."
    python stage7_sentiment_gpu.py 2>&1 | tee "$LOG_DIR/stage7.out"
    echo "[GPU] Stage 7 complete. Starting Stage 8: Topics..."
    python stage8_topics_gpu.py 2>&1 | tee "$LOG_DIR/stage8.out"
    echo "[GPU] All NLP stages complete."
) &
GPU_PID=$!

# CPU track: ML stages run in parallel using sklearn (FORCE_CPU=1)
# These don't need GPU — they use the 128 EPYC cores for pandas/sklearn
(
    echo "[CPU] Starting Stage 10 (Sustain) and Stage 11 (Forecast) in parallel..."
    FORCE_CPU=1 python stage10_sustain_gpu.py 2>&1 | tee "$LOG_DIR/stage10.out" &
    PID10=$!
    FORCE_CPU=1 python stage11_forecast_gpu.py 2>&1 | tee "$LOG_DIR/stage11.out" &
    PID11=$!
    wait $PID10
    echo "[CPU] Stage 10 complete."
    wait $PID11
    echo "[CPU] Stage 11 complete."
) &
CPU_PID=$!

# Wait for both tracks
wait $GPU_PID
echo ""
echo "[Phase 3] GPU track finished."
wait $CPU_PID
echo "[Phase 3] CPU track finished."

# ── Phase 4: Classification (needs 6,7,8 results) ────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Phase 4] Stage 9: Event Classification (cuML + XGBoost)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python stage9_classification_gpu.py 2>&1 | tee "$LOG_DIR/stage9.out"
echo "[Phase 4] Stage 9 complete."

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Pipeline Complete!"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Logs:    $LOG_DIR/"
echo "Results: s3://ven-bda-s3-v2/reddit-data/intermediate/"
echo "Cache:   /workspace/s3_cache/"
echo ""
echo "Execution summary:"
echo "  Phase 1: Stage 1  (GPU aggregation)"
echo "  Phase 2: S3 cache (CPU parallel download)"
echo "  Phase 3: Stages 6,7,8 on GPU + Stages 10,11 on CPU (simultaneous)"
echo "  Phase 4: Stage 9  (GPU classification)"
