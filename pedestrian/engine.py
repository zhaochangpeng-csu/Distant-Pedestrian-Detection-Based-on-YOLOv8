"""训练、评估、显存检查、计时和可复现结果归档。"""
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from .runtime import ROOT, read_json, write_json
from .models import VARIANTS, new_model, load_backbone
from .metrics import evaluate_records, curve, threshold_sweep, match_image
from .training import PersonTrainer
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import get_flops


def environment():
    import ultralytics
    import importlib.metadata
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "torch_cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "source": ultralytics.__file__,
            "gpu_available": torch.cuda.is_available(), "created_utc": datetime.now(timezone.utc).isoformat()}
    if info["gpu_available"]:
        info.update(gpu_name=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
                    gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
                    compiled_architectures=torch.cuda.get_arch_list())
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "unavailable"
    return info


def check_environment(strict=False):
    info = environment(); write_json(ROOT / "outputs/environment.json", info)
    if strict:
        if sys.version_info[:2] != (3, 11) or torch.__version__.split("+")[0] != "2.7.1":
            raise RuntimeError("服务器正式环境要求Python 3.11 / PyTorch 2.7.1，请先运行setup_windows.ps1。")
        if info["torchvision"].split("+")[0] != "0.22.1" or torch.version.cuda != "12.8":
            raise RuntimeError("服务器正式环境要求torchvision 0.22.1与cu128 PyTorch轮子。")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，不能启动正式训练。")
        if torch.cuda.get_device_capability(0)[0] >= 12 and tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8):
            raise RuntimeError("RTX 5080需要支持Blackwell的CUDA轮子，本项目采用cu128。")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.randn(16, 16, device=device, requires_grad=True)
    (x @ x.T).sum().backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA前向/反向校验失败。")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def get_settings(config):
    settings=yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    required={"epochs","imgsz","batch","seed","optimizer","lr0","amp"}
    if not isinstance(settings,dict) or not required.issubset(settings):
        raise ValueError("--config需要训练设置文件，例如configs/train.yaml；模型结构文件不适用于此参数。")
    return settings


def preflight(data, config, device="0", strict=False):
    """使用最大结构、密集GT和优化器一步检查显存，再统一固定四组batch。"""
    check_environment(strict)
    cfg = get_settings(config); cfg["device"] = device
    manifest = read_json(Path(data).resolve().parent / "manifest.json")
    records = sorted(manifest["splits"]["train"], key=lambda x: len(x["boxes"]), reverse=True)
    if not records:
        raise ValueError("训练集为空")
    target_device = torch.device("cpu" if str(device) == "cpu" else "cuda:" + str(device))
    chosen, peak = None, 0
    for batch_size in (cfg["batch"], max(1, cfg["batch"]//2), max(1, cfg["batch"]//4), 1):
        if chosen is not None:
            break
        model, optimizer, batch, target, loss, scaler = None, None, None, None, None, None
        try:
            if target_device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(target_device)
            model = new_model("eca_p2", cfg["seed"]).to(target_device).train()
            model.args = get_cfg(overrides={k:v for k,v in cfg.items() if k != "device"})
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr0"])
            rows = []
            for i in range(batch_size):
                # 以四块密集标注近似Mosaic的最坏目标数，增强显存探测的保守程度。
                for tile in range(4):
                    rec = records[(i*4+tile) % len(records)]
                    for x1, y1, x2, y2 in rec["boxes"]:
                        rows.append([i, 0, ((x1+x2)/2/rec["width"] + tile%2)/2,
                                     ((y1+y2)/2/rec["height"] + tile//2)/2,
                                     (x2-x1)/rec["width"]/2, (y2-y1)/rec["height"]/2])
            target = torch.tensor(rows, device=target_device, dtype=torch.float32).reshape(-1, 6)
            batch = {"img": torch.rand(batch_size, 3, cfg["imgsz"], cfg["imgsz"], device=target_device),
                     "batch_idx": target[:, 0], "cls": target[:, 1:2], "bboxes": target[:, 2:]}
            amp_enabled = bool(cfg["amp"] and target_device.type == "cuda")
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            with torch.autocast(device_type=target_device.type, enabled=amp_enabled):
                loss, _ = model(batch)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            chosen = batch_size
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device); peak = torch.cuda.max_memory_allocated(target_device)
            print(f"显存检查通过：batch={chosen}, peak={peak/1024**3:.2f} GiB")
        except torch.OutOfMemoryError:
            print(f"batch={batch_size} 显存不足，统一尝试更小batch。")
        finally:
            del model, optimizer, batch, target, loss, scaler
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError("batch=1仍然显存不足，请降低所有实验的输入分辨率后重新检查。")
    cfg["batch"] = chosen
    write_json(ROOT / "outputs/experiment_settings.json", {"settings": cfg, "data": str(Path(data).resolve()),
                "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(), "probe_peak_bytes": peak,
                "manifest_sha256": hashlib.sha256((Path(data).resolve().parent / "manifest.json").read_bytes()).hexdigest()})
    return cfg


def train_one(variant, data, settings, weights, name=None, resume=None):
    name = name or variant
    overrides = dict(settings, data=str(Path(data).resolve()), project=str(ROOT / "outputs/runs"), name=name,
                     exist_ok=False, pretrained=False)
    if resume:
        overrides["resume"] = str(Path(resume).resolve())
        overrides["model"] = str(Path(resume).resolve())
    else:
        model = new_model(variant, settings["seed"])
        if weights:
            load_backbone(model, weights, ROOT / "outputs/initialization" / f"{name}.json")
        overrides["model"] = str(ROOT / "configs/models" / f"{variant}.yaml")
    trainer = PersonTrainer(overrides=overrides)
    if not resume:
        trainer.model = model
    started = time.perf_counter()
    trainer.train()
    run = trainer.save_dir
    write_json(run / "run_provenance.json", {"variant": variant, "initialization": "shared pretrained backbone only" if weights else "scratch smoke only",
                "settings": settings, "elapsed_seconds": time.perf_counter()-started, "data": str(Path(data).resolve()),
                "data_manifest_sha256": hashlib.sha256((Path(data).resolve().parent / "manifest.json").read_bytes()).hexdigest(),
                "environment": environment(), "resume": str(resume) if resume else None})
    return str(run)


def predict_records(weights, data, split="val", imgsz=960, device="0", half=False):
    root = Path(data).resolve().parent
    annotations = read_json(root / "manifest.json")["splits"][split]
    model = YOLO(str(weights)); records = []
    for i, ann in enumerate(annotations):
        path = root / "originals" / split / ann["file"]
        # ndarray入口避免Windows视频/路径编码差异，坐标仍由Ultralytics还原至原图。
        image = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        result = model.predict(image, imgsz=imgsz, conf=0.001, iou=0.7, max_det=1000,
                               device=device, half=half, verbose=False)[0]
        boxes = result.boxes
        pred = torch.cat((boxes.xyxy, boxes.conf[:, None]), 1).cpu().numpy().tolist()
        records.append(dict(ann, pred=pred))
        if (i+1) % 100 == 0:
            print(f"{split}: {i+1}/{len(annotations)}", flush=True)
    return records


def evaluate(weights, data, output, split="val", imgsz=960, device="0", half=False, conf=0.25):
    records = predict_records(weights, data, split, imgsz, device, half)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    values = evaluate_records(records, conf)
    values.update(weights=str(Path(weights).resolve()), split=split, imgsz=imgsz, half=half,
                  image_count=len(records), environment=environment())
    write_json(out / "metrics.json", values); write_json(out / "predictions.json", records)
    write_json(out / "threshold_analysis.json", threshold_sweep(records,np.arange(0.05,0.951,0.05)))
    plot_evaluation(records, out)
    create_case_samples(records, Path(data).resolve().parent, split, out)
    return values


def plot_evaluation(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = curve(records)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c["recall"], c["precision"]); ax.set(xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1), title="Person PR at IoU=0.50")
    fig.tight_layout(); fig.savefig(out / "pr_curve.png", dpi=200); plt.close(fig)
    sweep = threshold_sweep(records,np.linspace(0.05,0.95,19))
    fig, ax = plt.subplots(figsize=(6,4))
    for key in ("precision", "recall", "f1"):
        ax.plot([r["operating_confidence"] for r in sweep], [r[key] for r in sweep], label=key)
    ax.set(xlabel="Confidence threshold", ylabel="Score", ylim=(0,1)); ax.legend()
    fig.tight_layout(); fig.savefig(out / "threshold_curve.png", dpi=200); plt.close(fig)


def create_case_samples(records, root, split, out):
    from PIL import ImageDraw
    selected = {}
    def add(candidates, category, count=2):
        for rec in candidates[:count]:
            selected.setdefault(rec["id"], {"record": rec, "categories": []})["categories"].append(category)

    add(sorted(records, key=lambda r: len(r["boxes"]), reverse=True), "dense")
    small = [(sum((b[2]-b[0])*(b[3]-b[1]) < 1024 for b in r["boxes"]), r) for r in records]
    add([r for n, r in sorted(small, key=lambda row: row[0], reverse=True) if n], "small")
    occluded = [(sum(a.get("occlusion", 0) > 0 for a in r.get("attributes", [])), r) for r in records]
    add([r for n, r in sorted(occluded, key=lambda row: row[0], reverse=True) if n], "occluded")
    failures = []
    for rec in records:
        scores, tp, fp, n, _ = match_image([p for p in rec["pred"] if p[4] >= 0.25], rec["boxes"], rec.get("ignores", []))
        failures.append((n-int(tp.sum())+int(fp.sum()), rec))
    add([rec for count, rec in sorted(failures, key=lambda row: row[0], reverse=True) if count > 0], "error")
    add([rec for rec in records if not rec["boxes"]], "negative")
    sample_index = []
    for item in selected.values():
        rec = item["record"]
        im = Image.open(root / "originals" / split / rec["file"]).convert("RGB"); draw=ImageDraw.Draw(im)
        for b in rec["boxes"]:
            draw.rectangle(b, outline="lime", width=2)
        for x1,y1,x2,y2,score in rec["pred"]:
            if score >= 0.25:
                draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        draw.text((8,8), "GREEN: GT   RED: prediction(conf>=0.25)", fill="yellow")
        im.save(out / f"case_{rec['id']}.jpg")
        sample_index.append({"id": rec["id"], "categories": item["categories"], "gt": len(rec["boxes"]),
                             "pred_at_025": sum(p[4] >= 0.25 for p in rec["pred"])})
    write_json(out / "case_selection.json", sample_index)


def benchmark(weights, image, output, imgsz=960, device="0", half=False, warmup=50, repeats=500):
    if repeats < 500 or warmup < 50:
        raise ValueError("正式速度测试要求至少50次预热和500次计时。")
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.tasks import attempt_load_one_weight
    import cv2
    d = torch.device("cpu" if device == "cpu" else "cuda:" + str(device))
    if half and d.type == "cpu":
        raise ValueError("本项目不在CPU上进行FP16测试。")
    raw = np.array(Image.open(image).convert("RGB"))[:,:,::-1].copy()
    model, _ = attempt_load_one_weight(str(weights), device=d, fuse=True)
    model = model.half() if half else model.float(); model.eval()
    def prep():
        im = LetterBox((imgsz,imgsz), auto=False, stride=32)(image=raw)
        t = torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))).unsqueeze(0).to(d)
        return (t.half() if half else t.float())/255
    tensor = prep()
    def sync():
        if d.type == "cuda": torch.cuda.synchronize(d)
    times, end_to_end = [], []
    with torch.inference_mode():
        for _ in range(warmup): model(tensor)
        sync()
        if d.type == "cuda": torch.cuda.reset_peak_memory_stats(d)
        for _ in range(repeats):
            sync(); t=time.perf_counter(); model(tensor); sync(); times.append((time.perf_counter()-t)*1000)
        parameters=sum(p.numel() for p in model.parameters())
        model_peak=torch.cuda.max_memory_allocated(d) if d.type=="cuda" else None
        gflops=float(get_flops(model,imgsz))
        del model, tensor
        if d.type=="cuda": torch.cuda.empty_cache()
        predictor=YOLO(str(weights))
        for _ in range(warmup):
            predictor.predict(raw, imgsz=imgsz, rect=False, half=half, device=device, conf=0.25, max_det=1000, verbose=False)
        if d.type=="cuda": torch.cuda.reset_peak_memory_stats(d)
        for _ in range(repeats):
            sync(); t=time.perf_counter()
            predictor.predict(raw, imgsz=imgsz, rect=False, half=half, device=device, conf=0.25, max_det=1000, verbose=False)
            sync(); end_to_end.append((time.perf_counter()-t)*1000)
    result={"imgsz":imgsz,"batch":1,"precision":"FP16" if half else "FP32","warmup":warmup,"repeats":repeats,
            "model_ms_mean":float(np.mean(times)),"model_ms_p95":float(np.percentile(times,95)),
            "end_to_end_ms_mean":float(np.mean(end_to_end)),"end_to_end_ms_p95":float(np.percentile(end_to_end,95)),
            "model_fps":1000/float(np.mean(times)),"end_to_end_fps":1000/float(np.mean(end_to_end)),
            "parameters":parameters,"gflops":gflops,"model_peak_allocated_bytes":model_peak,
            "end_to_end_peak_allocated_bytes":torch.cuda.max_memory_allocated(d) if d.type=="cuda" else None,
            "timing_boundary":"cached BGR image to detections; excludes disk read, plotting and encoding", "environment":environment()}
    write_json(output,result); return result


def pack_results(output, include_weights=True):
    target = Path(output).resolve()
    files=[]
    for base in (ROOT / "outputs/runs", ROOT / "outputs/evaluation", ROOT / "outputs/initialization", ROOT / "outputs/benchmarks", ROOT / "configs"):
        for p in base.rglob("*"):
            if p.is_file() and p.resolve()!=target and p.suffix not in (".zip", ".part") and (include_weights or p.suffix!=".pt"):
                files.append(p)
    files += [p for p in (ROOT/"outputs").glob("*") if p.is_file() and p.suffix in (".json", ".csv", ".png", ".txt")]
    manifests = [ROOT / "data/prepared/manifest.json", ROOT / "data/prepared/data_summary.json", ROOT / "data/prepared/duplicate_images.json"]
    files += [p for p in manifests if p.exists()]
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,str(p.relative_to(ROOT)))
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        z.writestr("bundle_manifest.json",json.dumps({"files_sha256":hashes,"raw_images_included":False},ensure_ascii=False,indent=2))
    print(target)


def summarize_results():
    rows=[]
    for p in sorted((ROOT/"outputs/evaluation").glob("*/metrics.json")):
        if p.parent.name.startswith("smoke"):
            continue
        m=read_json(p)
        rows.append({"experiment":p.parent.name,**{k:m.get(k) for k in ("split","imgsz","precision","recall","map50","map50_95","ap_small","tp","fp","fn")}})
    if rows:
        with open(ROOT/"outputs/experiment_summary.csv","w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows
