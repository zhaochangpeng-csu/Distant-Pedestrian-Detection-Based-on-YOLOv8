"""核对 CUDA_VISIBLE_DEVICES 映射到的 nvidia-smi 物理卡号。"""
import os
import argparse

import pynvml
import torch
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "vendor"))
from ultralytics.utils.torch_utils import select_device

parser = argparse.ArgumentParser()
parser.add_argument("--expected", type=int, required=True)
args = parser.parse_args()

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))
select_device("0", verbose=False)
print("CUDA_VISIBLE_DEVICES after Ultralytics selection:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("logical device count:", torch.cuda.device_count())
if torch.cuda.device_count() != 1:
    raise SystemExit("本次探测必须只暴露一张 GPU")
tensor = torch.ones(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
torch.cuda.synchronize()
pynvml.nvmlInit()
mapped = []
for index in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if process.pid == os.getpid():
            mapped.append((index, pynvml.nvmlDeviceGetUUID(handle)))
print("physical nvidia-smi mapping:", mapped)
if len(mapped) != 1:
    raise SystemExit("无法唯一确认物理 GPU 编号")
if mapped[0][0] != args.expected:
    raise SystemExit(f"GPU映射错误：期望物理卡 {args.expected}，实际为 {mapped[0][0]}")
