"""图片、目录、视频与摄像头推理；流式写JSON数组及CSV，保留空帧。"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .runtime import write_json
from ultralytics import YOLO


def predict_source(weights, source, output, imgsz=960, device="0", conf=0.25, max_frames=None):
    out=Path(output); out.mkdir(parents=True, exist_ok=True)
    model=YOLO(str(weights)); path=Path(str(source))
    image_extensions={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    is_image=path.is_file() and path.suffix.lower() in image_extensions
    image_paths=sorted(p for p in path.iterdir() if p.suffix.lower() in image_extensions) if path.is_dir() else ([path] if is_image else [])
    if path.is_dir() and not image_paths:
        raise ValueError("目录内没有支持的图片。")
    capture, writer=None,None
    if not image_paths:
        capture=cv2.VideoCapture(int(source) if str(source).isdigit() else str(path.resolve()))
        if not capture.isOpened(): raise OSError(f"无法打开视频或摄像头 {source}")
    json_file=open(out/"detections.json","w",encoding="utf-8")
    csv_file=open(out/"detections.csv","w",newline="",encoding="utf-8-sig")
    csv_writer=csv.writer(csv_file); csv_writer.writerow(["frame","source","class","confidence","x1","y1","x2","y2"])
    json_file.write("["); frame=0
    try:
        while True:
            if max_frames is not None and frame >= max_frames: break
            if image_paths:
                if frame >= len(image_paths): break
                image=np.array(Image.open(image_paths[frame]).convert("RGB"))[:,:,::-1].copy()
                name=image_paths[frame].name
            else:
                ok,image=capture.read()
                if not ok: break
                name=str(source)
            result=model.predict(image,imgsz=imgsz,device=device,conf=conf,iou=0.7,max_det=1000,verbose=False)[0]
            detections=[]
            for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                detections.append({"class":"person","confidence":score,"xyxy":box})
                csv_writer.writerow([frame,name,"person",score,*box])
            if frame: json_file.write(",\n")
            json.dump({"frame":frame,"source":name,"width":image.shape[1],"height":image.shape[0],"detections":detections},json_file,ensure_ascii=False)
            # 密集远景人体的默认文字标签会相互遮挡，画面只保留细框；置信度仍完整写入JSON/CSV。
            rendered=result.plot(labels=False, conf=False, line_width=1)
            if image_paths:
                cv2.imencode(".jpg",rendered)[1].tofile(str(out/f"{frame:06d}_{Path(name).stem}.jpg"))
            else:
                if writer is None:
                    fps=capture.get(cv2.CAP_PROP_FPS)
                    fps=fps if np.isfinite(fps) and 1<=fps<=120 else 25.0
                    writer=cv2.VideoWriter(str(out/"annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(image.shape[1],image.shape[0]))
                    if not writer.isOpened(): raise OSError("视频编码器初始化失败。")
                writer.write(rendered)
            frame+=1
    finally:
        json_file.write("]"); json_file.close(); csv_file.close()
        if capture: capture.release()
        if writer: writer.release()
    write_json(out/"inference_info.json",{"frames":frame,"weights":str(weights),"imgsz":imgsz,"confidence":conf,"coordinate_system":"original image pixels"})
    return frame
