"""自定义验证器在每轮使用相同忽略区感知AP，按mAP50-95保存最佳权重。"""
from copy import copy
from pathlib import Path
import numpy as np
from .runtime import read_json, write_json
from .metrics import evaluate_records, curve
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator


class PersonValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        manifest = read_json(Path(self.data["path"]) / "manifest.json")
        self.annotation_index = {(split,r["id"]): r for split,records in manifest["splits"].items() for r in records}
        self.records = []

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            self.seen += 1
            prepared = self._prepare_batch(si, batch)
            native = self._prepare_pred(pred, prepared).detach().cpu().numpy()
            image_path=Path(batch["im_file"][si])
            rid = image_path.stem
            ann = self.annotation_index[(image_path.parent.name,rid)]
            self.records.append({"id": rid, "pred": native[:, :5].tolist(), "boxes": ann["boxes"], "ignores": ann["ignores"]})

    def get_stats(self):
        values = evaluate_records(self.records)
        if not values["ground_truth_count"]:
            raise ValueError("验证集没有有效人体GT，无法选择最佳模型。")
        p, r, f1 = values["precision"], values["recall"], values["f1"]
        x = np.linspace(0, 1, 101)
        c = curve(self.records)
        pr = np.array([c["precision"][c["recall"] >= t].max() if np.any(c["recall"] >= t) else 0 for t in x])
        pc, rc = [], []
        for threshold in x:
            selected = c["scores"] >= threshold
            tp, fp = c["tp"][selected].sum(), c["fp"][selected].sum()
            pc.append(tp/max(tp+fp, 1)); rc.append(tp/max(c["gt"], 1))
        pc, rc = np.array([pc]), np.array([rc])
        self.metrics.box.nc = 1
        self.metrics.box.update((np.array([p]), np.array([r]), np.array([f1]),
                                 np.array([values["ap_by_iou"]]), np.array([0]), pc, rc,
                                 2*pc*rc/np.maximum(pc+rc, 1e-12), x, np.array([pr])))
        self.nt_per_class = np.array([values["ground_truth_count"]])
        self.nt_per_image = np.array([sum(bool(r["boxes"]) for r in self.records)])
        write_json(self.save_dir / "person_metrics.json", values)
        result = dict(self.metrics.results_dict)
        result["fitness"] = values["map50_95"]
        return result

    def print_results(self):
        from ultralytics.utils import LOGGER
        LOGGER.info(f"person: {self.seen} images, {self.nt_per_class.sum()} boxes, "
                    f"P/R/AP50/AP50-95={self.metrics.mean_results()}")


class PersonTrainer(DetectionTrainer):
    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return PersonValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        try:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        except (OSError, RuntimeError) as exc:
            if self.args.workers == 0:
                raise
            print(f"数据加载进程创建失败，回退workers=0: {exc}")
            self.args.workers = 0
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
