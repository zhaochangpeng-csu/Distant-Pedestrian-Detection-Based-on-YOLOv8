"""单类、原图坐标、忽略区域感知的101点AP。与官方多类别榜单口径区分。"""
import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def overlap(a, b, ioa=False):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)[:, None]
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)[None, :]
    return inter / np.maximum(aa if ioa else aa + bb - inter, 1e-12)


def match_image(pred, boxes, ignores, iou=0.5, small=False):
    """先匹配计分GT，再忽略非计分GT/区域；重复检测仍可能是FP。"""
    pred = np.asarray(pred, dtype=float).reshape(-1, 5)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    ignores = np.asarray(ignores, dtype=float).reshape(-1, 4)
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    active = areas < 1024 if small else np.ones(len(boxes), dtype=bool)
    gt, non_scoring = boxes[active], boxes[~active]
    pred = pred[np.argsort(-pred[:, 4], kind="stable")]
    used = set()
    tp, fp, scores, ignored = [], [], [], 0
    valid_iou = overlap(pred[:, :4], gt)
    other_iou = overlap(pred[:, :4], non_scoring)
    ignore_ioa = overlap(pred[:, :4], ignores, ioa=True)
    for j, p in enumerate(pred):
        candidates = [g for g in range(len(gt)) if g not in used and valid_iou[j, g] >= iou]
        if candidates:
            g = max(candidates, key=lambda g: valid_iou[j, g])
            used.add(g)
            tp.append(1); fp.append(0); scores.append(p[4])
        elif ((len(non_scoring) and other_iou[j].max() >= iou) or
              (len(ignores) and ignore_ioa[j].max() >= 0.5)):
            ignored += 1
        else:
            tp.append(0); fp.append(1); scores.append(p[4])
    return np.asarray(scores), np.asarray(tp), np.asarray(fp), len(gt), ignored


def curve(records, iou=0.5, small=False):
    items = [match_image(r["pred"], r["boxes"], r.get("ignores", []), iou, small) for r in records]
    n = sum(x[3] for x in items)
    scores = np.concatenate([x[0] for x in items]) if items else np.array([])
    tp = np.concatenate([x[1] for x in items]) if items else np.array([])
    fp = np.concatenate([x[2] for x in items]) if items else np.array([])
    order = np.argsort(-scores, kind="stable")
    scores, tp, fp = scores[order], tp[order], fp[order]
    recall = np.cumsum(tp) / max(n, 1)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1)
    # 在101个召回位置取得右侧最大精度；没有预测时AP为0，没有GT时不定义。
    ap = float(np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0.0
                        for r in np.linspace(0, 1, 101)])) if n else None
    return {"ap": ap, "scores": scores, "tp": tp, "fp": fp, "recall": recall,
            "precision": precision, "gt": n, "ignored": sum(x[4] for x in items)}


def evaluate_records(records, conf=0.25):
    aps = [curve(records, float(t))["ap"] for t in IOU_THRESHOLDS]
    small_aps = [curve(records, float(t), True)["ap"] for t in IOU_THRESHOLDS]
    c = curve(records)
    selected = c["scores"] >= conf
    tp, fp = int(c["tp"][selected].sum()), int(c["fp"][selected].sum())
    p, r = tp / max(tp + fp, 1), tp / max(c["gt"], 1)
    return {"precision": p, "recall": r, "f1": 2*p*r/max(p+r, 1e-12),
            "map50": aps[0], "map50_95": float(np.mean(aps)) if aps[0] is not None else None,
            "ap_small": float(np.mean(small_aps)) if small_aps[0] is not None else None,
            "ap_by_iou": aps, "tp": tp, "fp": fp, "fn": c["gt"]-tp,
            "ground_truth_count": c["gt"], "ignored_predictions_at_iou50": c["ignored"],
            "operating_confidence": conf, "protocol": "person-only; 101 recall points; max_det=1000; native-image area<1024 for small"}


def threshold_sweep(records, thresholds):
    """阈值变化不影响完整AP；一次匹配即可计算所有P/R工作点。"""
    c=curve(records)
    rows=[]
    for threshold in thresholds:
        keep=c["scores"]>=threshold
        tp,fp=int(c["tp"][keep].sum()),int(c["fp"][keep].sum())
        p,r=tp/max(tp+fp,1),tp/max(c["gt"],1)
        rows.append({"operating_confidence":float(threshold),"precision":p,"recall":r,
                     "f1":2*p*r/max(p+r,1e-12),"tp":tp,"fp":fp,"fn":c["gt"]-tp})
    return rows
