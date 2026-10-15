"""统一入口。Windows多进程训练必须保留 __main__ 保护。"""
import argparse
import multiprocessing
from pathlib import Path
import sys
from pedestrian.runtime import ROOT, read_json, write_json


def main():
    parser=argparse.ArgumentParser(description="YOLOv8远景人体检测工程")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("download"); p.add_argument("--raw",default="data/raw"); p.add_argument("--splits",nargs="+",default=["train","val","test"]); p.add_argument("--offline",action="store_true")
    p=sub.add_parser("prepare"); p.add_argument("--raw",default="data/raw"); p.add_argument("--output",default="data/prepared"); p.add_argument("--splits",nargs="+",default=["train","val","test"]); p.add_argument("--limit",type=int); p.add_argument("--strict-counts",action="store_true")
    p=sub.add_parser("check"); p.add_argument("--strict",action="store_true")
    p=sub.add_parser("preflight"); p.add_argument("--data",default="data/prepared/dataset.yaml"); p.add_argument("--config",default="configs/train.yaml"); p.add_argument("--device",default="0"); p.add_argument("--strict",action="store_true")
    for name in ("train","train-all"):
        p=sub.add_parser(name); p.add_argument("--data",default="data/prepared/dataset.yaml"); p.add_argument("--config",default="configs/train.yaml"); p.add_argument("--weights",default="downloads/yolov8n.pt"); p.add_argument("--device"); p.add_argument("--imgsz",type=int); p.add_argument("--batch",type=int); p.add_argument("--epochs",type=int); p.add_argument("--workers",type=int); p.add_argument("--no-plots",action="store_true"); p.add_argument("--scratch-smoke",action="store_true")
        if name=="train":
            p.add_argument("--variant",choices=["baseline","eca","p2","eca_p2"],required=True); p.add_argument("--name"); p.add_argument("--resume")
        else:
            p.add_argument("--tag",default="",help="独立实验目录后缀，例如e25")
    p=sub.add_parser("evaluate"); p.add_argument("--weights",required=True); p.add_argument("--data",default="data/prepared/dataset.yaml"); p.add_argument("--split",choices=["val","test"],default="val"); p.add_argument("--imgsz",type=int,default=960); p.add_argument("--device",default="0"); p.add_argument("--half",action="store_true"); p.add_argument("--conf",type=float,default=.25); p.add_argument("--output",required=True)
    p=sub.add_parser("analyze"); p.add_argument("--data",default="data/prepared/dataset.yaml"); p.add_argument("--device",default="0"); p.add_argument("--split",choices=["val","test"],default="val"); p.add_argument("--tag",default="")
    p=sub.add_parser("benchmark"); p.add_argument("--weights",required=True); p.add_argument("--image",required=True); p.add_argument("--output",required=True); p.add_argument("--imgsz",type=int,default=960); p.add_argument("--device",default="0"); p.add_argument("--half",action="store_true")
    p=sub.add_parser("predict"); p.add_argument("--weights",required=True); p.add_argument("--source",required=True); p.add_argument("--output",default="outputs/inference"); p.add_argument("--imgsz",type=int,default=960); p.add_argument("--device",default="0"); p.add_argument("--conf",type=float,default=.25); p.add_argument("--max-frames",type=int)
    p=sub.add_parser("pack"); p.add_argument("--output",default="outputs/results_bundle.zip"); p.add_argument("--no-weights",action="store_true")
    sub.add_parser("summary")
    args=parser.parse_args()
    if args.command in ("download","prepare"):
        from pedestrian.data import download_dataset,prepare_dataset
        if args.command=="download": download_dataset(args.raw,args.splits,args.offline)
        else: prepare_dataset(args.raw,args.output,args.splits,args.limit,args.strict_counts)
    elif args.command=="predict":
        from pedestrian.inference import predict_source
        predict_source(args.weights,args.source,args.output,args.imgsz,args.device,args.conf,args.max_frames)
    else:
        from pedestrian.engine import check_environment,preflight,get_settings,train_one,evaluate,benchmark,pack_results,summarize_results
        from pedestrian.models import VARIANTS, model_config
        if args.command=="check": check_environment(args.strict)
        elif args.command=="preflight": preflight(args.data,args.config,args.device,args.strict)
        elif args.command in ("train","train-all"):
            import yaml
            import urllib.request
            cfg=get_settings(args.config)
            lock=ROOT/"outputs/experiment_settings.json"
            if lock.exists():
                saved=read_json(lock)
                if saved["data"]==str(Path(args.data).resolve()): cfg=saved["settings"]
            for key in ("device","imgsz","batch","epochs","workers"):
                if getattr(args,key) is not None: cfg[key]=getattr(args,key)
            if args.no_plots: cfg["plots"]=False
            if args.scratch_smoke and cfg["epochs"]>3:
                raise ValueError("随机初始化仅用于最多3轮的流程检查。")
            if args.command=="train-all":
                if not lock.exists() or read_json(lock)["data"]!=str(Path(args.data).resolve()):
                    raise ValueError("四组正式训练前请先对相同数据运行preflight。")
                manifest=read_json(Path(args.data).resolve().parent/"manifest.json")
                if manifest.get("limited_for_smoke_only") or manifest.get("smoke_only"):
                    raise ValueError("禁止将流程检查子集用于正式四组实验。")
                from pedestrian.data import EXPECTED_COUNTS
                for split, expected in EXPECTED_COUNTS.items():
                    actual=len(manifest["splits"].get(split, []))
                    if actual != expected:
                        raise ValueError(f"正式{split}应有{expected}张图像，当前清单只有{actual}张；请使用完整官方划分。")
                import hashlib
                saved=read_json(lock)
                if saved["config_sha256"]!=hashlib.sha256(Path(args.config).read_bytes()).hexdigest():
                    raise ValueError("训练配置在显存检查后发生变化，请重新运行preflight。")
                if saved["manifest_sha256"]!=hashlib.sha256((Path(args.data).resolve().parent/"manifest.json").read_bytes()).hexdigest():
                    raise ValueError("数据清单发生变化，请重新运行preflight。")
                if any(getattr(args,k) is not None for k in ("device","imgsz","batch","epochs","workers")) or args.no_plots:
                    raise ValueError("正式四组使用preflight锁定设置，请先修改共同config再重新preflight。")
            weights=None if args.scratch_smoke else Path(args.weights)
            if weights and not weights.exists():
                weights.parent.mkdir(parents=True,exist_ok=True)
                urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",weights)
            variants=VARIANTS if args.command=="train-all" else [args.variant]
            for variant in variants:
                path=ROOT/"configs/models"/f"{variant}.yaml"; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(yaml.safe_dump(model_config(variant),sort_keys=False),encoding="utf-8")
                name = (f"{variant}_{args.tag}" if args.tag else variant) if args.command=="train-all" else args.name
                train_one(variant,args.data,cfg,weights,name,getattr(args,"resume",None))
        elif args.command=="evaluate":
            evaluate(args.weights,args.data,args.output,args.split,args.imgsz,args.device,args.half,args.conf); summarize_results()
        elif args.command=="analyze":
            for variant in ("baseline","eca_p2"):
                for size in (640,960,1280):
                    name=f"{variant}_{args.tag}" if args.tag else variant
                    evaluate(ROOT/f"outputs/runs/{name}/weights/best.pt",args.data,ROOT/f"outputs/evaluation/{name}_{args.split}_{size}",args.split,size,args.device)
            summarize_results()
        elif args.command=="benchmark": benchmark(args.weights,args.image,args.output,args.imgsz,args.device,args.half)
        elif args.command=="pack": pack_results(args.output,not args.no_weights)
        elif args.command=="summary": print(summarize_results())


if __name__=="__main__":
    multiprocessing.freeze_support()
    main()
