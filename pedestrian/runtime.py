"""只使用项目内源码，缓存写到项目目录，不修改用户全局环境。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "runtime_cache" / "ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime_cache" / "matplotlib")
loaded = sys.modules.get("ultralytics")
if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(ROOT / "vendor"):
    raise RuntimeError("已加载其他Ultralytics版本。请在独立进程中运行项目入口run.py。")
sys.path.insert(0, str(ROOT / "vendor"))


def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))
