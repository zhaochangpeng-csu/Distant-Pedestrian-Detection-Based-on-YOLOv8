"""制作三帧接口测试视频并验证输出可解码；不作为正式演示视频。"""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pedestrian.runtime import ROOT,write_json
from pedestrian.inference import predict_source
import cv2
import numpy as np
from PIL import Image


def main():
    out=ROOT/"outputs/video_check"; out.mkdir(parents=True,exist_ok=True)
    image=next((ROOT/"data/smoke/originals/val").glob("*.jpg"))
    im=cv2.resize(np.array(Image.open(image).convert("RGB"))[:,:,::-1].copy(),(640,360))
    path=out/"接口测试输入.mp4"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10,(640,360))
    assert writer.isOpened()
    for _ in range(3): writer.write(im)
    writer.release()
    target=out/"中文输出"
    n=predict_source(ROOT/"outputs/runs/smoke_eca_p2_verified/weights/best.pt",str(path),target,256)
    cap=cv2.VideoCapture(str(target/"annotated.mp4")); frames=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        assert frame.shape[:2]==(360,640)
        frames+=1
    cap.release()
    entries=json.loads((target/"detections.json").read_text(encoding="utf-8"))
    assert n==frames==len(entries)==3
    assert [r["frame"] for r in entries]==[0,1,2]
    write_json(out/"verification.json",{"status":"passed","frames":frames,"chinese_paths":True,
               "empty_detections_retained":all("detections" in r for r in entries),"not_a_submission_demo":True})


if __name__=="__main__": main()
