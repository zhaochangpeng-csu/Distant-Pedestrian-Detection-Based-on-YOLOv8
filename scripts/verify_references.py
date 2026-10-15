"""通过Crossref核对DOI标题，存储完整元数据；网页来源由sources记录核验。"""
import concurrent.futures
import difflib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pedestrian.runtime import ROOT,read_json,write_json


def normalized(text):
    return re.sub(r"[^\w]","",text.casefold())


def verify(ref):
    if not ref.get("doi"):
        return {"id":ref["id"],"status":ref.get("verified","primary_page_to_check"),"source":ref.get("source")}
    url="https://api.crossref.org/works/"+urllib.parse.quote(ref["doi"],safe="")
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"PedestrianCourseReferenceAudit/1.0"})
        with urllib.request.urlopen(req,timeout=30) as f:
            metadata=json.load(f)["message"]
        title=metadata["title"][0]
        similarity=difflib.SequenceMatcher(None,normalized(title),normalized(ref.get("crossref_title",ref["title"]))).ratio()
        metadata={k:metadata[k] for k in ("DOI","title","author","container-title","publisher","volume","issue","page","issued","published","published-print","published-online","URL","type") if k in metadata}
        return {"id":ref["id"],"doi":ref["doi"],"status":"title_verified" if similarity>=.85 else "needs_review",
                "similarity":similarity,"metadata":metadata}
    except Exception as exc:
        return {"id":ref["id"],"doi":ref["doi"],"status":ref.get("verified","request_failed"),"error":str(exc),"source":ref.get("source")}


def main():
    refs=read_json(ROOT/"materials/references.json")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(verify,refs))
    write_json(ROOT/"materials/reference_audit.json",result)
    for row in result:
        m=row.get("metadata",{})
        print(row["id"],row["status"],m.get("page"),m.get("published",{}).get("date-parts"),m.get("title"))


if __name__=="__main__": main()
