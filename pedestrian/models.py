"""构建四组模型及共同骨干初始化。"""
from pathlib import Path
from .runtime import ROOT, write_json
import torch
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.nn.tasks import DetectionModel

VARIANTS = ("baseline", "eca", "p2", "eca_p2")


def model_config(variant):
    if variant not in VARIANTS:
        raise ValueError(f"未知模型 {variant}，可选 {VARIANTS}")
    filename = "yolov8-p2.yaml" if "p2" in variant else "yolov8.yaml"
    cfg = yaml_load(ROOT / "vendor/ultralytics/cfg/models/v8" / filename)
    cfg.update(nc=1, scale="n")
    if "eca" in variant:
        for i in (2, 4):
            cfg["backbone"][i][2] = "C2fECA"
    return cfg


def new_model(variant, seed=42):
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(seed, deterministic=True)
    return DetectionModel(model_config(variant), ch=3, nc=1, verbose=False)


def load_backbone(model, weights, log_path=None):
    # 仅加载可信的官方权重或本项目产出的文件。
    source = YOLO(str(weights)).model.float().state_dict()
    target = model.state_dict()
    selected = {}
    for key, tensor in source.items():
        parts = key.split(".")
        if parts[0] == "model" and int(parts[1]) < 10 and key in target and target[key].shape == tensor.shape:
            selected[key] = tensor
    if not selected:
        raise RuntimeError("没有加载任何骨干参数，请检查权重来源与版本。")
    model.load_state_dict(selected, strict=False)
    report = {"weights": str(Path(weights).resolve()), "policy": "backbone indices 0..9 only; neck/head fresh",
              "loaded_tensor_count": len(selected), "loaded_parameter_elements": sum(t.numel() for t in selected.values()),
              "keys": sorted(selected), "fresh_keys": sorted(set(target) - set(selected))}
    if log_path:
        write_json(log_path, report)
    return report
