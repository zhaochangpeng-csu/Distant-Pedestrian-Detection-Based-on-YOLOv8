"""在第1轮保存后模拟中断，再从last.pt完成第2轮。仅用于流程检查。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pedestrian.runtime import ROOT,write_json
from pedestrian.models import new_model,load_backbone,model_config
from pedestrian.training import PersonTrainer
from pedestrian.engine import get_settings
import yaml


def stop_after_first_epoch(trainer):
    if trainer.epoch==0: raise RuntimeError("CONTROLLED_INTERRUPTION_AFTER_CHECKPOINT")


def main():
    settings=get_settings(ROOT/"configs/train.yaml")
    settings.update(epochs=2,imgsz=256,batch=2,workers=0,plots=False)
    model_file=ROOT/"configs/models/eca_p2.yaml"
    model_file.parent.mkdir(parents=True,exist_ok=True)
    model_file.write_text(yaml.safe_dump(model_config("eca_p2")),encoding="utf-8")
    options=dict(settings,data=str(ROOT/"data/smoke/dataset.yaml"),model=str(model_file),
                 project=str(ROOT/"outputs/runs"),name="smoke_resume",pretrained=False)
    trainer=PersonTrainer(overrides=options)
    trainer.model=new_model("eca_p2"); load_backbone(trainer.model,ROOT/"downloads/yolov8n.pt")
    trainer.add_callback("on_fit_epoch_end",stop_after_first_epoch)
    try: trainer.train()
    except RuntimeError as exc:
        if "CONTROLLED_INTERRUPTION" not in str(exc): raise
    checkpoint=trainer.save_dir/"weights/last.pt"
    if not checkpoint.exists(): raise AssertionError("未生成检查点")
    resumed=PersonTrainer(overrides=dict(options,resume=str(checkpoint)))
    resumed.train()
    write_json(ROOT/"outputs/resume_check.json",{"checkpoint":str(checkpoint),"status":"passed",
               "interrupted_after_epoch":1,"completed_epochs":2,"smoke_only":True})


if __name__=="__main__": main()
