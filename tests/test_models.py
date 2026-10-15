import torch
import pytest
from pedestrian.models import new_model,load_backbone
from pedestrian.modules import C2fECA
from pedestrian.runtime import ROOT

torch.set_num_threads(4)


@pytest.mark.parametrize("variant",["baseline","eca","p2","eca_p2"])
def test_forward_gradient_and_strides(variant):
    m=new_model(variant); m.train()
    y=m(torch.randn(2,3,64,64))
    assert m.stride.tolist()==([4,8,16,32] if "p2" in variant else [8,16,32])
    sum(t.square().mean() for t in y).backward()
    if "eca" in variant:
        modules=[x for x in m.modules() if isinstance(x,C2fECA)]
        assert len(modules)==2
        assert all(x.eca.conv.weight.grad is not None and torch.isfinite(x.eca.conv.weight.grad).all() for x in modules)


def test_attention_checkpoint_reload(tmp_path):
    m=new_model("eca_p2").eval(); x=torch.randn(1,3,64,64)
    with torch.no_grad(): y=m(x)[0]
    p=tmp_path/"model.pt"; torch.save(m,p)
    restored=torch.load(p,weights_only=False).eval()
    with torch.no_grad(): torch.testing.assert_close(y,restored(x)[0])


def test_shared_backbone_does_not_load_head():
    weights=ROOT/"downloads/yolov8n.pt"
    if not weights.exists(): pytest.skip("需官方权重")
    m=new_model("eca_p2")
    before={k:v.clone() for k,v in m.state_dict().items() if int(k.split('.')[1])>=10}
    report=load_backbone(m,weights)
    assert report["loaded_tensor_count"]>100
    assert all(int(k.split('.')[1])<10 for k in report["keys"])
    for k,v in before.items(): torch.testing.assert_close(m.state_dict()[k],v)
