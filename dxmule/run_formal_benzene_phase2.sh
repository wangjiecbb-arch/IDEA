#!/usr/bin/env bash
# HYPOTHESIS. Phase-2 fair Benzene compare: single-scale IDEA vs multi-scale DXMulE.
# Full 70/30 split. No --max_graphs. Matched-subset ROC-AUC uses one shared skip
# rule. Not a gate approval. Loss values are not comparable across methods.
# Numbers written by the entry points stay PENDING_RESULT until an evidence check.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dxmule/formal_logs_phase2

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

SEEDS="${PHASE2_SEEDS:-2025 2026 2027}"
fail=0

run() {
  local name="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--max_graphs" || "${arg}" == --max_graphs=* ]]; then
      echo "REFUSING ${name}: --max_graphs is forbidden on the full Benzene protocol" >&2
      fail=1
      return 2
    fi
  done
  local log="dxmule/formal_logs_phase2/${name}.log"
  local porcelain
  porcelain="$(git status --porcelain || true)"
  {
    echo "=== HOST ==="
    date -u +%Y-%m-%dT%H:%M:%SZ
    uname -a
    echo "nproc $(nproc)"
    free -h || true
    echo "COMMIT $(git rev-parse HEAD)"
    echo "PYTHON $(command -v python3)"
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
    return "${ec}"
  fi
  return 0
}

echo "=== test_core ==="
if ! python3 -m dxmule.test_core > dxmule/formal_logs_phase2/test_core.log 2>&1; then
  echo "FAILED test_core. See dxmule/formal_logs_phase2/test_core.log" >&2
  exit 1
fi
tail -n 5 dxmule/formal_logs_phase2/test_core.log

matched_files=()
for seed in ${SEEDS}; do
  idea_ec=0
  dx_ec=0
  run "idea_benzene_s${seed}" \
    python3 -u explainer/eval_idea.py \
    --dataset Benzene --root_path ./ --device "${DEVICE}" \
    --epochs 20 --lr 0.001 --batch_size 32 \
    --seed "${seed}" --use_ckpt 1 --codebook_size 32 \
    --fidelity_scales 0.1,0.3,0.5,0.7,0.9 \
    --auc_record "dxmule/formal_logs_phase2/idea_s${seed}_auc.json" || idea_ec=$?
  run "dxmule_benzene_s${seed}" \
    python3 -u explainer/eval_dxmule.py \
    --dataset Benzene --root_path ./ --device "${DEVICE}" \
    --epochs 20 --lr 0.001 --batch_size 32 \
    --seed "${seed}" --use_ckpt 1 --codebook_size 32 \
    --scales 0.1,0.3,0.5,0.7,0.9 \
    --cross_scale_weight 0.1 --complement_weight 0.1 \
    --shell_weight 0.1 --retention_temperature 1.0 \
    --auc_record "dxmule/formal_logs_phase2/dxmule_s${seed}_auc.json" || dx_ec=$?
  if [[ "${idea_ec}" -eq 0 && "${dx_ec}" -eq 0 ]]; then
    set +e
    python3 -m dxmule.matched_auc \
      --idea "dxmule/formal_logs_phase2/idea_s${seed}_auc.json" \
      --dxmule "dxmule/formal_logs_phase2/dxmule_s${seed}_auc.json" \
      --out "dxmule/formal_logs_phase2/matched_s${seed}.json" \
      --require-full-split \
      --expect-n-test 3600 \
      | tee "dxmule/formal_logs_phase2/matched_s${seed}.log"
    mec=$?
    set -e
    if [[ "${mec}" -ne 0 ]]; then
      echo "FAILED matched AUC for seed ${seed}" >&2
      fail=1
    else
      matched_files+=("dxmule/formal_logs_phase2/matched_s${seed}.json")
    fi
  else
    echo "SKIP matched AUC for seed ${seed} (idea exit ${idea_ec}, dxmule exit ${dx_ec})" >&2
    fail=1
  fi
done

if [[ "${#matched_files[@]}" -gt 0 ]]; then
  set +e
  python3 -m dxmule.matched_auc \
    --summary "${matched_files[@]}" \
    --summary-out dxmule/formal_logs_phase2/matched_summary.json \
    | tee dxmule/formal_logs_phase2/matched_summary.log
  sec=$?
  set -e
  if [[ "${sec}" -ne 0 ]]; then
    echo "FAILED matched summary" >&2
    fail=1
  fi
else
  echo "No matched JSON files. Summary skipped." >&2
  fail=1
fi

echo "HYPOTHESIS PENDING_RESULT. Loss is not comparable across methods. Do not rank methods from these logs."
echo "Logs: dxmule/formal_logs_phase2/"
exit "${fail}"
