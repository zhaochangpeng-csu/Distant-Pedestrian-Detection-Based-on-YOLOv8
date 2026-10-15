"""只打包源码、材料、可信官方权重和小型检查证据，排除环境与原始数据。"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pedestrian.runtime import ROOT


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="../YOLOv8远景行人检测_服务器训练包.zip"); args=parser.parse_args()
    output=(ROOT/args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    directories=[ROOT/"pedestrian",ROOT/"configs",ROOT/"scripts",ROOT/"tests",ROOT/"materials",ROOT/"vendor"]
    files=[ROOT/"README.md",ROOT/"run.py",ROOT/"requirements.txt",ROOT/".gitignore",ROOT/"downloads/yolov8n.pt"]
    excluded={"__pycache__",".pytest_cache",".venv","runtime_cache"}
    for base in directories:
        files += [p for p in base.rglob("*") if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts) and p.suffix not in (".pyc", ".pyo")]
    evidence=[ROOT/"outputs/model_inventory.json",ROOT/"outputs/resume_check.json",ROOT/"outputs/video_check/verification.json"]
    files += [p for p in evidence if p.exists()]
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        hashes={}
        for p in sorted(set(files)):
            if not p.exists(): continue
            relative=str(p.relative_to(ROOT)).replace("\\","/")
            z.write(p,"yolo_pedestrian/"+relative)
            hashes[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
        z.writestr("yolo_pedestrian/package_manifest.json",json.dumps({"stage":"server-ready source and Markdown; formal results pending",
                    "files_sha256":hashes,"raw_dataset_included":False,"venv_included":False},ensure_ascii=False,indent=2))
    print(output,"bytes:",output.stat().st_size,"files:",len(files))


if __name__=="__main__": main()
