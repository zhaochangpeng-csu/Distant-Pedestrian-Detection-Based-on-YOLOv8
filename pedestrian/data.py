"""VisDrone原始数据转换；不改动原始文件，训练遮罩与原图评估分离。"""
import csv
import hashlib
import shutil
import zipfile
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import yaml
from .runtime import write_json, read_json

SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
EXPECTED_COUNTS = {"train": 6471, "val": 548, "test": 1610}
BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download_dataset(raw, splits=("train", "val", "test"), offline=False):
    raw = Path(raw).resolve(); raw.mkdir(parents=True, exist_ok=True)
    sources = []
    for split in splits:
        folder = SPLITS[split]
        archive = raw / (folder + ".zip")
        extracted = raw / folder
        if not archive.exists():
            images = len(list((extracted / "images").glob("*.jpg")))
            labels = len(list((extracted / "annotations").glob("*.txt")))
            if images == EXPECTED_COUNTS[split] and labels == EXPECTED_COUNTS[split]:
                sources.append({"split": split, "source": "local_extracted", "images": images, "annotations": labels})
                continue
            if offline:
                raise FileNotFoundError(
                    f"离线模式缺少 {archive} 或完整的 {extracted}；"
                    f"请从可联网电脑复制 {archive.name} 到 data/raw，现有图片/标注 {images}/{labels}。"
                )
            temp = archive.with_suffix(".zip.part")
            print(f"下载 {BASE_URL + archive.name}", flush=True)
            try:
                urllib.request.urlretrieve(BASE_URL + archive.name, temp)
            except (urllib.error.URLError, OSError) as exc:
                raise RuntimeError(
                    f"下载 {archive.name} 失败：{exc}。可在能访问GitHub的电脑下载该ZIP，"
                    f"复制到 {raw}，再运行 run.py download --offline。"
                ) from exc
            temp.replace(archive)
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"压缩包校验失败: {bad}")
            for member in z.infolist():
                dest = (raw / member.filename).resolve()
                if not dest.is_relative_to(raw):
                    raise ValueError("压缩包包含越界路径")
            have_images = len(list((extracted / "images").glob("*.jpg")))
            have_labels = len(list((extracted / "annotations").glob("*.txt")))
            if have_images != EXPECTED_COUNTS[split] or have_labels != EXPECTED_COUNTS[split]:
                z.extractall(raw)
        images = len(list((extracted / "images").glob("*.jpg")))
        labels = len(list((extracted / "annotations").glob("*.txt")))
        if images != EXPECTED_COUNTS[split] or labels != EXPECTED_COUNTS[split]:
            raise ValueError(f"{folder}解压不完整：图片/标注 {images}/{labels}，应各有{EXPECTED_COUNTS[split]}。")
        sources.append({"split": split, "url": BASE_URL + archive.name, "sha256": sha256(archive)})
    write_json(raw / "download_provenance.json", sources)


def parse_annotation(path, width, height):
    boxes, ignores, attributes, issues = [], [], [], Counter()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.strip().rstrip(",").split(",")
        if len(fields) < 8:
            raise ValueError(f"错误标注格式: {path}: {line}")
        x, y, w, h, score, category, truncation, occlusion = map(float, fields[:8])
        if not np.isfinite([x, y, w, h, score, category, truncation, occlusion]).all():
            raise ValueError(f"非有限标注: {path}")
        issues[f"raw_class_{int(category)}"] += 1
        original = np.array([x, y, x+w, y+h])
        box = np.clip(original, [0, 0, 0, 0], [width, height, width, height]).tolist()
        if box[2] <= box[0] or box[3] <= box[1]:
            issues["invalid_area"] += 1
            continue
        if not np.array_equal(original, box):
            issues["clipped_boxes"] += 1
        if score == 0 or int(category) in (0, 11):
            ignores.append(box)
        elif int(category) in (1, 2):
            if box in boxes:
                issues["duplicate_person_box"] += 1
                continue
            boxes.append(box)
            attributes.append({"source_class": int(category), "occlusion": int(occlusion), "truncation": int(truncation)})
    return boxes, ignores, attributes, issues


def mask_ignores(image, boxes, ignores):
    result = image.copy()
    for b in ignores:
        x1, y1, x2, y2 = map(int, [np.floor(b[0]), np.floor(b[1]), np.ceil(b[2]), np.ceil(b[3])])
        result[y1:y2, x1:x2] = 114
    # 恢复人体框内像素，避免忽略区与有效GT相交时破坏监督信息。
    for b in boxes:
        x1, y1, x2, y2 = map(int, [np.floor(b[0]), np.floor(b[1]), np.ceil(b[2]), np.ceil(b[3])])
        result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return result


def prepare_dataset(raw, output, splits=("train", "val", "test"), limit=None, strict_counts=False):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"已存在数据清单，避免覆盖：{output}。请指定新的输出目录。")
    manifest = {"protocol": "VisDrone2019-DET person=source classes 1+2", "raw_root": str(raw), "splits": {}, "issues": {}}
    hashes, duplicates, global_issues = {}, [], Counter()
    for split in splits:
        src = raw / SPLITS[split]
        images = sorted((src / "images").glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"未找到 {src / 'images'}")
        if strict_counts and len(images) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}应有{EXPECTED_COUNTS[split]}张官方图像，实际{len(images)}张；请核对下载与解压。")
        if limit:
            images = images[:limit]
        target = output / "images" / split
        labels = output / "labels" / split
        originals = output / "originals" / split
        for d in (target, labels, originals):
            d.mkdir(parents=True, exist_ok=True)
        records = []
        for image_path in images:
            digest = sha256(image_path)
            try:
                with Image.open(image_path) as im:
                    image = np.array(im.convert("RGB")); width, height = im.size
            except (OSError, ValueError):
                global_issues["corrupt_image"] += 1
                continue
            ann = src / "annotations" / (image_path.stem + ".txt")
            boxes, ignores, attrs, issues = parse_annotation(ann, width, height)
            annotation_digest = sha256(ann)
            if digest in hashes:
                previous = hashes[digest]
                kind = "within_split" if previous["split"] == split else "across_splits"
                duplicates.append({"first": previous["path"], "second": f"{split}/{image_path.name}",
                                   "kind": kind, "image_sha256": digest,
                                   "same_annotation_file": previous["annotation_sha256"] == annotation_digest})
                global_issues[f"duplicate_{kind}"] += 1
            else:
                hashes[digest] = {"split": split, "path": f"{split}/{image_path.name}",
                                  "annotation_sha256": annotation_digest}
            global_issues.update(issues)
            original_path = originals / image_path.name
            shutil.copy2(image_path, original_path)
            if split == "train" and ignores:
                Image.fromarray(mask_ignores(image, boxes, ignores)).save(target / image_path.name, quality=95)
            else:
                shutil.copy2(image_path, target / image_path.name)
            lines = []
            for x1, y1, x2, y2 in boxes:
                lines.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
            (labels / (image_path.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            records.append({"id": image_path.stem, "file": image_path.name, "width": width, "height": height,
                            "sha256": digest, "boxes": boxes, "ignores": ignores, "attributes": attrs})
            if len(records) <= 6:
                preview = Image.fromarray(image); draw = ImageDraw.Draw(preview)
                for b in ignores:
                    draw.rectangle(b, outline="orange", width=2)
                for b in boxes:
                    draw.rectangle(b, outline="lime", width=2)
                sample_dir = output / "audit_samples"; sample_dir.mkdir(exist_ok=True)
                preview.save(sample_dir / f"{split}_{image_path.name}")
        manifest["splits"][split] = records
        if strict_counts and len(records) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}成功转换{len(records)}张，少于官方{EXPECTED_COUNTS[split]}张；请检查损坏图像。")
    manifest["issues"] = dict(global_issues)
    manifest["duplicate_images"] = duplicates
    manifest["limited_for_smoke_only"] = bool(limit)
    write_json(output / "manifest.json", manifest)
    write_json(output / "duplicate_images.json", duplicates)
    if duplicates:
        print(f"发现{len(duplicates)}对字节相同图像（同划分{global_issues['duplicate_within_split']}，跨划分{global_issues['duplicate_across_splits']}）；已保留官方划分，详见duplicate_images.json。", flush=True)
    cfg = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "person"}}
    (output / "dataset.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summarize_dataset(output)
    return manifest


def summarize_dataset(output):
    output = Path(output); manifest = read_json(output / "manifest.json")
    summary, areas, occlusion = {}, [], Counter()
    for split, records in manifest["splits"].items():
        all_boxes = [b for r in records for b in r["boxes"]]
        aa = [(b[2]-b[0])*(b[3]-b[1]) for b in all_boxes]
        summary[split] = {"images": len(records), "negative_images": sum(not r["boxes"] for r in records),
                          "person_boxes": len(aa), "small_area_lt_1024": sum(a < 1024 for a in aa),
                          "ignore_regions": sum(len(r["ignores"]) for r in records)}
        for r in records:
            for b, attr in zip(r["boxes"], r["attributes"]):
                a = (b[2]-b[0])*(b[3]-b[1]); areas.append((split, r["id"], a, b[3]-b[1], attr["occlusion"]))
                occlusion[f"{split}:{attr['occlusion']}"] += 1
    write_json(output / "data_summary.json", {"splits": summary, "occlusion": dict(occlusion), "issues": manifest["issues"],
                                                "duplicate_images": {"within_split": sum(d["kind"] == "within_split" for d in manifest["duplicate_images"]),
                                                                     "across_splits": sum(d["kind"] == "across_splits" for d in manifest["duplicate_images"])}})
    with open(output / "object_sizes.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["split", "id", "area_px2", "height_px", "occlusion"]); writer.writerows(areas)
    plot_distributions(output, areas, manifest["issues"])
    return summary


def plot_distributions(output, areas, issues):
    """保存原图像素面积、遮挡程度和原始类别分布，供报告核对。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["train", "val", "test"]
    values = [np.array([row[2] for row in areas if row[0] == split]) for split in splits]
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, data in zip(splits, values):
        if len(data):
            ax.hist(np.log2(np.maximum(data, 1)), bins=32, alpha=0.45, label=split)
    ax.axvline(10, color="black", linestyle="--", label="32 x 32 px")
    ax.set(xlabel="log2 box area (pixel squared)", ylabel="Person boxes", title="Original-image person area")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "area_distribution.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    for i, split in enumerate(splits):
        counts = Counter(row[4] for row in areas if row[0] == split)
        ax.bar(x + (i-1)*0.25, [counts.get(level, 0) for level in x], width=0.25, label=split)
    ax.set(xticks=x, xticklabels=["none", "partial", "heavy"], ylabel="Person boxes", title="Occlusion annotations")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "occlusion_distribution.png", dpi=180); plt.close(fig)

    raw = sorted((int(k.removeprefix("raw_class_")), v) for k, v in issues.items() if k.startswith("raw_class_"))
    if raw:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(k) for k, _ in raw], [v for _, v in raw])
        ax.set(xlabel="VisDrone raw category", ylabel="Annotations", title="Raw class distribution")
        fig.tight_layout(); fig.savefig(output / "raw_class_distribution.png", dpi=180); plt.close(fig)
