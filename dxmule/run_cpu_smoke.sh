#!/usr/bin/env bash
# HYPOTHESIS / smoke / provisional.
# CPU Benzene subset for single-scale IDEA vs multi-scale DXMulE.
# This is a plumbing check. It is not a paper result and it does not rank methods.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dxmule/smoke_logs

python3 - << 'PY'
import os
import zipfile

filename = "benzene_with_gt.pkl"
pickle_path = os.path.join("data", filename)
zip_path = pickle_path + ".zip"
if os.path.isfile(pickle_path):
    print(f"pickle present: {pickle_path}")
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
    print(f"unzipped {zip_path} -> {pickle_path}")
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

run() {
  local name="$1"
  shift
  local log="dxmule/smoke_logs/${name}.log"
  {
    echo "=== HOST ==="
    date -u +%Y-%m-%dT%H:%M:%SZ
    uname -a
    echo "nproc $(nproc)"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
    python3 -c 'import torch, torch_geometric, sklearn, numpy; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("pyg", torch_geometric.__version__); print("numpy", numpy.__version__); print("sklearn", sklearn.__version__); print("geomloss", "0.2.6")'
    echo "=== CMD ==="
    printf '%s ' "$@"
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
    exit "${ec}"
  fi
}

run idea_cpu_smoke \
  python3 -u explainer/eval_idea.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 2 --lr 0.001 --batch_size 4 --max_graphs 8 \
  --seed 2025 --use_ckpt 1 --codebook_size 32

run dxmule_cpu_smoke \
  python3 -u explainer/eval_dxmule.py \
  --dataset Benzene --root_path ./ --device -1 \
  --epochs 2 --lr 0.001 --batch_size 4 --max_graphs 8 \
  --seed 2025 --use_ckpt 1 --codebook_size 32 \
  --scales 0.1,0.3,0.5,0.7,0.9 \
  --cross_scale_weight 0.1 --complement_weight 0.1 \
  --shell_weight 0.1 --retention_temperature 1.0

echo "Logs: dxmule/smoke_logs/idea_cpu_smoke.log dxmule/smoke_logs/dxmule_cpu_smoke.log"
