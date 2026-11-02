#!/usr/bin/env bash
# HYPOTHESIS. Full Benzene 70/30, single-scale IDEA vs multi-scale DXMulE.
# Not a gate approval. Numbers written by the entry points stay PENDING_RESULT
# until an evidence check. This script must not pass --max_graphs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dxmule/formal_logs

python3 - << 'PY'
import os
import zipfile

filename = "benzene_with_gt.pkl"
pickle_path = os.path.join("data", filename)
zip_path = pickle_path + ".zip"
if os.path.isfile(pickle_path):
    print(f"pickle present: {pickle_path} ({os.path.getsize(pickle_path)} bytes)")
else:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if os.path.basename(name) == filename and not name.startswith("__MACOSX")
        ]
        if not members:
            raise SystemExit(f"{zip_path} does not contain {filename}")
        with archive.open(members[0]) as source, open(pickle_path, "wb") as target:
            target.write(source.read())
    print(f"unzipped {zip_path} -> {pickle_path} ({os.path.getsize(pickle_path)} bytes)")
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

if python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  DEVICE=0
else
  DEVICE=-1
fi

fail=0

run() {
  local name="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--max_graphs" || "${arg}" == --max_graphs=* ]]; then
      echo "REFUSING ${name}: --max_graphs is forbidden on the full Benzene protocol" >&2
      fail=1
      return
    fi
  done
  local log="dxmule/formal_logs/${name}.log"
  local porcelain
  porcelain="$(git status --porcelain || true)"
  {
    echo "=== HOST ==="
    date -u +%Y-%m-%dT%H:%M:%SZ
    uname -a
    echo "nproc $(nproc)"
    free -h || true
    echo "COMMIT $(git rev-parse HEAD)"
    echo "PORCELAIN_START"
    printf '%s\n' "${porcelain}"
    echo "PORCELAIN_END"
    echo "DEVICE_ARG ${DEVICE}"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
    echo "TQDM_DISABLE=${TQDM_DISABLE}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || true
    else
      echo "nvidia-smi not found"
    fi
    python3 - << 'PY'
import importlib.metadata as meta
import torch
import numpy
import sklearn
import torch_geometric
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
print("pyg", torch_geometric.__version__)
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
print("geomloss", meta.version("geomloss"))
PY
    echo "=== CMD ==="
    printf '%q ' "$@"
    echo
    echo "=== RUN ==="
  } > "${log}"
  local start end ec
  start="$(date +%s.%N)"
  set +e
  "$@" >> "${log}" 2>&1
  ec=$?
  set -e
  end="$(date +%s.%N)"
  python3 -c "print('ELAPSED_SEC {:.3f}'.format(${end} - ${start})); print('EXIT ${ec}')" | tee -a "${log}"
  if [[ "${ec}" -ne 0 ]]; then
    echo "FAILED ${name} (exit ${ec}). See ${log}" >&2
    fail=1
  fi
}

run idea_benzene_full \
  python3 -u explainer/eval_idea.py \
  --dataset Benzene --root_path ./ --device "${DEVICE}" \
  --epochs 20 --lr 0.001 --batch_size 32 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --fidelity_scales 0.1,0.3,0.5,0.7,0.9

run dxmule_benzene_full \
  python3 -u explainer/eval_dxmule.py \
  --dataset Benzene --root_path ./ --device "${DEVICE}" \
  --epochs 20 --lr 0.001 --batch_size 32 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --scales 0.1,0.3,0.5,0.7,0.9 \
  --cross_scale_weight 0.1 --complement_weight 0.1 \
  --shell_weight 0.1 --retention_temperature 1.0

echo "Logs: dxmule/formal_logs/idea_benzene_full.log dxmule/formal_logs/dxmule_benzene_full.log"
exit "${fail}"
