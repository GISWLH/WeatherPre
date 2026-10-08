#!/usr/bin/env bash
# Research model run on a shared GPU box (xuanze: 2x RTX 4090 24 GB, yongqiang: 1x RTX 4090 24 GB -- re-check every time).
# Isolated from operational services: own venv per model family, own cache and output roots, one GPU per job, resumable.
#
#   scripts/remote/run_model.sh fuxi-s2s t2m,tp,sst 6w 2022-12-01 --members 11 --seed 0
#   scripts/remote/run_model.sh orca-dl tos 7m 2022-12-01 --members 1,2,3,4,5,6,7,8 --source godas
#
# Environment (defaults shown): WP_ROOT=$HOME/weatherpre-research  WP_VENV_<FAMILY>=$WP_ROOT/venv-<family>
# WEATHERPRE_WEIGHTS_<MODEL> must point at locally downloaded weights (never committed / redistributed).
set -euo pipefail
MODEL=$1; VARS=$2; LEAD=$3; INIT=$4; shift 4
WP_ROOT=${WP_ROOT:-$HOME/weatherpre-research}
case "$MODEL" in
  fuxi-s2s) FAMILY=onnx ;; orca-dl|ace2) FAMILY=torch ;; neuralgcm) FAMILY=jax ;;
  *) echo "no research runner for $MODEL" >&2; exit 2 ;;
esac
VENV_VAR="WP_VENV_${FAMILY^^}"; VENV=${!VENV_VAR:-$WP_ROOT/venv-$FAMILY}
OUT=$WP_ROOT/runs/$MODEL/$INIT; mkdir -p "$OUT" "$WP_ROOT/cache"
# 1) GPU state before anything else: pick the GPU with the most free memory; never sum memory across GPUs
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv | tee "$OUT/nvidia-smi.before.csv"
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
NEED=$("$VENV/bin/python" -c "from weatherpre import catalog as C; print(int((C.get('$MODEL').gpu_mem_gb or 8) * 1024 * 1.2))")
DEVICE=cuda; if [ "$FREE" -lt "$NEED" ]; then echo "GPU $GPU has $FREE MiB free < $NEED MiB needed" >&2; DEVICE=cpu; fi
# 2) availability (no download) then the run; members are processed one at a time and written as separate files
export CUDA_VISIBLE_DEVICES=$GPU WEATHERAI_CACHE=$WP_ROOT/cache WEATHERPRE_DATA=$WP_ROOT/cache
"$VENV/bin/python" -m weatherpre check "$MODEL" "$VARS" "$LEAD" --init "$INIT" || { echo "availability check failed" >&2; exit 3; }
/usr/bin/time -v "$VENV/bin/python" -m weatherpre forecast "$MODEL" "$VARS" "$LEAD" --init "$INIT" --device "$DEVICE" \
    --out "$OUT" --out-dir "$OUT" "$@" 2> >(tee "$OUT/time.log" >&2)
nvidia-smi --query-gpu=index,memory.used --format=csv | tee "$OUT/nvidia-smi.after.csv"
echo "outputs + manifests in $OUT"
