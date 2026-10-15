"""整理当前真实证据、GB/T参考文献与源程序阅读附录，不填写未测指标。"""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pedestrian.runtime import ROOT,read_json,write_json


def main():
    refs=read_json(ROOT/"materials/references.json")
    audit={r["id"]:r for r in read_json(ROOT/"materials/reference_audit.json")}
    lines=["# 参考文献","", "共28条，其中中文6条，占21.4%。采用GB/T 7714顺序编码格式。DOI条目的页码采用其对应出版记录；CVF公开版与IEEE记录可能使用不同页码。",""]
    ris=[]
    for r in refs:
        citation=r["citation"]
        if r.get("doi") and "DOI:" not in citation:
            citation+=f" DOI: {r['doi']}."
        url=r.get("source") or f"https://doi.org/{r['doi']}"
        lines += [f"[{r['id']}] {citation}",f"核验来源：{url}",""]
        typ="BOOK" if r["id"]==6 else ("ELEC" if r["id"] in (12,13,15) else ("JOUR" if r["id"] in (1,2,3,4,5,25) else "CONF"))
        ris += [f"TY  - {typ}",f"ID  - {r['id']}",f"TI  - {r['title']}",f"UR  - {url}"]
        metadata=audit.get(r["id"],{}).get("metadata",{})
        if metadata.get("author"):
            ris += ["AU  - "+a.get("family","")+", "+a.get("given","") for a in metadata["author"]]
        else:
            prefix=r["citation"].split(". ")[0]
            ris += ["AU  - "+a.strip() for a in prefix.split(",") if a.strip() not in ("等","et al")]
        year=re.search(r"\b(?:19|20)\d{2}\b",r["citation"])
        if year and r["id"]!=15: ris.append("PY  - "+year.group())
        if metadata.get("container-title"): ris.append("T2  - "+metadata["container-title"][0])
        if metadata.get("volume"): ris.append("VL  - "+metadata["volume"])
        if metadata.get("issue"): ris.append("IS  - "+metadata["issue"])
        if metadata.get("page"):
            pages=metadata["page"].split("-"); ris.append("SP  - "+pages[0])
            if len(pages)>1: ris.append("EP  - "+pages[-1])
        if r.get("doi"): ris.append(f"DO  - {r['doi']}")
        ris += [f"N1  - {citation}","ER  - ",""]
    (ROOT/"materials/参考文献.md").write_text("\n".join(lines),encoding="utf-8")
    (ROOT/"materials/references.ris").write_text("\n".join(ris),encoding="utf-8")
    source_lines=["# 带注释源程序阅读附录","","上游Ultralytics完整源码在vendor目录。本附录列出本项目自定义代码，正式报告排版时可按教师要求选取核心段落，完整源码必须随附。",""]
    for p in sorted((ROOT/"pedestrian").glob("*.py")):
        source_lines += [f"## {p.name}","","```python",p.read_text(encoding="utf-8"),"```",""]
    source_lines += ["## run.py","","```python",(ROOT/"run.py").read_text(encoding="utf-8"),"```",""]
    (ROOT/"materials/源程序附录.md").write_text("\n".join(source_lines),encoding="utf-8")
    inventory=read_json(ROOT/"outputs/model_inventory.json")
    table=["# 四组模型结构统计","","统计对象为单类person、未融合BN的模型，输入960。GFLOPs为统一工具近似值，可能未完整统计池化、Sigmoid和广播乘法，不能替代实测速度。", "",
           "| 模型 | 参数量 | GFLOPs近似 | 检测步长 |","|---|---:|---:|---|"]
    for r in inventory:
        table.append(f"| {r['variant']} | {r['parameters_unfused']:,} | {r['gflops_960']:.3f} | {', '.join(str(int(x)) for x in r['strides'])} |")
    table += ["","两个ECA增加6个权重；P2版本更改了颈部通道，参数量略减但空间计算增大。正式FPS与峰值显存待RTX 5080服务器测量。"]
    (ROOT/"materials/模型结构统计.md").write_text("\n".join(table),encoding="utf-8")
    report=ROOT/"materials/课程报告.md"
    content=report.read_text(encoding="utf-8")
    marker="<!-- AUTO_MODEL_INVENTORY -->"
    if marker not in content:
        content=content.replace("## 6 软件实现与运行环境",marker+"\n\n"+"\n".join(table[4:])+"\n\n## 6 软件实现与运行环境")
        report.write_text(content,encoding="utf-8")
    print("Markdown materials refreshed; references:",len(refs))


if __name__=="__main__": main()
