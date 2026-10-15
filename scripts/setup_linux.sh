#!/usr/bin/env bash
# 离线服务器：只在缺少环境时克隆现有的 cu128 环境，不改动原环境。
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_BIN="${CONDA_BIN:-/opt/conda/bin/conda}"
SOURCE_ENV="${SOURCE_ENV:-geneval_blackwell2}"
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq zcp_train; then
  if ! "$CONDA_BIN" create -n zcp_train --clone "$SOURCE_ENV" -y --offline; then
    # Conda 离线克隆在少数源包未缓存时会失败，已安装文件仍可独立复制。
    source_prefix="$HOME/.conda/envs/$SOURCE_ENV"
    target_prefix="$HOME/.conda/envs/zcp_train"
    test -x "$source_prefix/bin/python"
    mkdir -p "$target_prefix"
    rsync -a "$source_prefix/" "$target_prefix/"
  fi
fi
export CUDA_VISIBLE_DEVICES="${GPU3_UUID:-GPU-1475e7c4-6dde-5be5-c5fc-e0d787fb252d}"
export PYTHONUNBUFFERED=1
"$CONDA_BIN" run -n zcp_train --no-capture-output python scripts/check_linux_env.py
"$CONDA_BIN" run -n zcp_train --no-capture-output python run.py check
