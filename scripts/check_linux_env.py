"""Linux 离线环境的实际版本检查；Python 版本偏差需如实记录。"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

import torch
import torchvision

required = ("numpy", "cv2", "PIL", "yaml", "matplotlib", "scipy", "pandas",
            "requests", "tqdm", "psutil", "cpuinfo", "thop")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"zcp_train 缺少依赖：{', '.join(missing)}。请先安装到该环境。")
assert torch.__version__.split("+")[0] == "2.7.1", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.22.1", torchvision.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA 不可用"
print(f"Python {sys.version.split()[0]} / PyTorch {torch.__version__} / CUDA {torch.version.cuda}")
if sys.version_info[:2] != (3, 11):
    print("记录偏差：离线源环境为 Python 3.10；模型和四组实验仍使用相同环境。")
