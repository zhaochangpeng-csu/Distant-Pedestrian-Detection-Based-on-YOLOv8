from pathlib import Path
import zipfile
import numpy as np
import pytest
from PIL import Image
from pedestrian.data import parse_annotation,mask_ignores,prepare_dataset,download_dataset
from pedestrian import data as data_module


def test_annotation_clip_merge_and_ignore(tmp_path):
    p=tmp_path/"a.txt"
    p.write_text("-2,0,12,10,1,1,0,0\n12,12,5,5,1,2,0,2\n0,0,20,20,0,0,0,0\n0,0,0,4,1,1,0,0\n",encoding="utf-8")
    boxes,ignores,attrs,issues=parse_annotation(p,20,20)
    assert len(boxes)==2 and boxes[0]==[0,0,10,10] and len(ignores)==1
    assert issues["clipped_boxes"]==1 and issues["invalid_area"]==1 and attrs[1]["occlusion"]==2


def test_mask_preserves_valid_person():
    im=np.full((20,20,3),255,dtype=np.uint8)
    masked=mask_ignores(im,[[5,5,10,10]],[[0,0,20,20]])
    assert masked[0,0,0]==114 and masked[6,6,0]==255 and im[0,0,0]==255


def test_empty_label_and_chinese_path(tmp_path):
    raw=tmp_path/"原始数据"/"VisDrone2019-DET-val"
    (raw/"images").mkdir(parents=True); (raw/"annotations").mkdir()
    Image.new("RGB",(32,24),"white").save(raw/"images/空样本.jpg")
    (raw/"annotations/空样本.txt").write_text("0,0,10,10,1,4,0,0",encoding="utf-8")
    out=tmp_path/"输出数据"
    m=prepare_dataset(raw.parent,out,["val"])
    assert m["splits"]["val"][0]["boxes"]==[]
    assert (out/"labels/val/空样本.txt").read_text()==""
    with pytest.raises(ValueError, match="548"):
        prepare_dataset(raw.parent,tmp_path/"strict_output",["val"],strict_counts=True)


def test_duplicate_images_recorded_without_changing_split(tmp_path):
    raw=tmp_path/"raw/VisDrone2019-DET-val"
    (raw/"images").mkdir(parents=True); (raw/"annotations").mkdir()
    for name in ("a","b"):
        Image.new("RGB",(32,32)).save(raw/f"images/{name}.jpg")
        (raw/f"annotations/{name}.txt").write_text("",encoding="utf-8")
    manifest=prepare_dataset(raw.parent,tmp_path/"out",["val"])
    assert len(manifest["splits"]["val"])==2
    assert manifest["duplicate_images"][0]["kind"]=="within_split"
    assert manifest["duplicate_images"][0]["same_annotation_file"]
    assert (tmp_path/"out/duplicate_images.json").exists()


def test_cross_split_duplicates_are_flagged(tmp_path):
    root=tmp_path/"raw"
    for folder,name in (("VisDrone2019-DET-train","a"),("VisDrone2019-DET-val","b")):
        source=root/folder
        (source/"images").mkdir(parents=True)
        (source/"annotations").mkdir()
        Image.new("RGB",(32,32)).save(source/f"images/{name}.jpg")
        (source/f"annotations/{name}.txt").write_text("",encoding="utf-8")
    manifest=prepare_dataset(root,tmp_path/"out",["train","val"])
    assert manifest["duplicate_images"][0]["kind"]=="across_splits"
    assert len(manifest["splits"]["train"])==len(manifest["splits"]["val"])==1


def test_offline_zip_import_and_missing_split(tmp_path, monkeypatch):
    monkeypatch.setitem(data_module.EXPECTED_COUNTS, "test", 1)
    raw = tmp_path / "raw"
    raw.mkdir()
    folder = "VisDrone2019-DET-test-dev"
    with zipfile.ZipFile(raw / f"{folder}.zip", "w") as z:
        z.writestr(f"{folder}/images/example.jpg", b"image bytes")
        z.writestr(f"{folder}/annotations/example.txt", b"")
    download_dataset(raw, ["test"], offline=True)
    assert (raw / folder / "images/example.jpg").exists()
    assert (raw / "download_provenance.json").exists()
    (raw / f"{folder}.zip").unlink()
    download_dataset(raw, ["test"], offline=True)
    (raw / folder / "images/example.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="离线模式缺少"):
        download_dataset(raw, ["test"], offline=True)
