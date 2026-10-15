import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pedestrian.runtime import ROOT,write_json
from pedestrian.models import VARIANTS,new_model,model_config
import yaml
from ultralytics.utils.torch_utils import get_flops
import torch


def main():
    torch.set_num_threads(4)
    rows=[]
    for variant in VARIANTS:
        config_file=ROOT/"configs/models"/f"{variant}.yaml"
        config_file.parent.mkdir(parents=True,exist_ok=True)
        config_file.write_text(yaml.safe_dump(model_config(variant),sort_keys=False),encoding="utf-8")
        model=new_model(variant)
        rows.append({"variant":variant,"parameters_unfused":sum(p.numel() for p in model.parameters()),
                     "strides":model.stride.tolist(),"gflops_960":float(get_flops(model,960))})
    write_json(ROOT/"outputs/model_inventory.json",rows)
    print(rows)


if __name__=="__main__": main()
