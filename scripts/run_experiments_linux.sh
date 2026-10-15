#!/usr/bin/env bash
# 以 GPU UUID 固定物理卡；本服务器 CUDA 数字顺序与 nvidia-smi 不一致。
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_BIN="${CONDA_BIN:-/opt/conda/bin/conda}"
GPU1_UUID="${GPU1_UUID:-GPU-eaedd20a-bb8d-becb-b36e-7ec155d7309e}"
GPU3_UUID="${GPU3_UUID:-GPU-1475e7c4-6dde-5be5-c5fc-e0d787fb252d}"
export CUDA_VISIBLE_DEVICES="$GPU3_UUID"
export PYTHONUNBUFFERED=1
run() { "$CONDA_BIN" run -n zcp_train --no-capture-output python run.py "$@"; }
probe() { "$CONDA_BIN" run -n zcp_train --no-capture-output python scripts/probe_gpu_mapping.py --expected "$1"; }
probe 3

if [[ "${1:-}" == "--final-test" ]]; then
  for variant in baseline eca p2 eca_p2; do
    run evaluate --weights "outputs/runs/${variant}_e100/weights/best.pt" \
      --split test --output "outputs/evaluation/${variant}_e100_test_960"
  done
  run pack --output outputs/results_bundle_e100.zip
  exit 0
fi

if [[ "${1:-}" == "--parallel" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU1_UUID" probe 1
fi
run check
if [[ ! -f data/prepared/manifest.json ]]; then
  run download --offline
  run prepare --strict-counts
fi
for variant in baseline eca p2 eca_p2; do
  if [[ -e "outputs/runs/${variant}_e100" ]]; then
    echo "已存在训练目录 outputs/runs/${variant}_e100，请检查后手动断点恢复，避免覆盖。" >&2
    exit 1
  fi
done
run preflight --config configs/train.yaml
if [[ "${1:-}" == "--parallel" ]]; then
  mkdir -p outputs/logs
  run_variant() {
    local gpu="$1" variant="$2"
    if [[ "$gpu" == 1 ]]; then
      export CUDA_VISIBLE_DEVICES="$GPU1_UUID"
    else
      export CUDA_VISIBLE_DEVICES="$GPU3_UUID"
    fi
    run train --variant "$variant" --config configs/train.yaml \
      --name "${variant}_e100" > "outputs/logs/${variant}_e100.log" 2>&1
  }
  run_pair() {
    run_variant 1 "$1" & local first=$!
    run_variant 3 "$2" & local second=$!
    local failed=0
    if ! wait "$first"; then echo "$1 在 GPU 1 上训练失败；查看 outputs/logs/${1}_e100.log" >&2; failed=1; fi
    if ! wait "$second"; then echo "$2 在 GPU 3 上训练失败；查看 outputs/logs/${2}_e100.log" >&2; failed=1; fi
    (( failed == 0 )) || exit 1
  }
  run_pair baseline eca_p2
  run_pair eca p2
  export CUDA_VISIBLE_DEVICES="$GPU3_UUID"
else
  run train-all --config configs/train.yaml --tag e100
fi
for variant in baseline eca p2 eca_p2; do
  run evaluate --weights "outputs/runs/${variant}_e100/weights/best.pt" \
    --split val --output "outputs/evaluation/${variant}_e100_val_960"
done
sample_images=(data/prepared/originals/val/*.jpg)
for variant in baseline eca p2 eca_p2; do
  run benchmark --weights "outputs/runs/${variant}_e100/weights/best.pt" \
    --image "${sample_images[0]}" --output "outputs/benchmark_${variant}_e100.json"
done
run analyze --split val --tag e100
for variant in baseline eca p2 eca_p2; do
  run evaluate --weights "outputs/runs/${variant}_e100/weights/best.pt" \
    --split test --output "outputs/evaluation/${variant}_e100_test_960"
done
run pack --output outputs/results_bundle_e100.zip
echo '完成：outputs/results_bundle_e100.zip'
