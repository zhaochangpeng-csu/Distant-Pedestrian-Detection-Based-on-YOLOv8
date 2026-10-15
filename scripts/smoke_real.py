"""仅验证流程：从官方val取互不重叠的图像构造临时train/val/test，不能作为正式实验。"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pedestrian.runtime import ROOT,write_json,read_json
from pedestrian.data import prepare_dataset,SPLITS


def main():
    p=argparse.ArgumentParser(); p.add_argument("--archive",default="downloads/VisDrone2019-DET-val.zip"); p.add_argument("--output",default="data/smoke"); args=p.parse_args()
    archive=ROOT/args.archive
    raw=ROOT/"data/smoke_raw"; original=ROOT/"data/official_val"
    original.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not (original/member).resolve().is_relative_to(original.resolve()): raise ValueError("越界ZIP路径")
        z.extractall(original)
    src=original/"VisDrone2019-DET-val"
    images=sorted((src/"images").glob("*.jpg"))
    subsets={"train":images[:8],"val":images[8:12],"test":images[12:16]}
    for split,files in subsets.items():
        dest=raw/SPLITS[split]
        (dest/"images").mkdir(parents=True,exist_ok=True); (dest/"annotations").mkdir(exist_ok=True)
        for im in files:
            shutil.copy2(im,dest/"images"/im.name)
            shutil.copy2(src/"annotations"/(im.stem+".txt"),dest/"annotations"/(im.stem+".txt"))
    out=ROOT/args.output
    prepare_dataset(raw,out)
    manifest=read_json(out/"manifest.json"); manifest["smoke_only"]=True
    manifest["smoke_notice"]="All images originate from official val. Split is temporary and not a research benchmark."
    write_json(out/"manifest.json",manifest)
    print(out/"dataset.yaml")


if __name__=="__main__": main()
