"""使用本机 Microsoft Word 将制作稿导出为 PDF，供分页检查。"""

from pathlib import Path

import win32com.client


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "materials" / "基于YOLOv8的远景行人检测_PPT制作稿.docx"
    target = root / "runtime_cache" / "ppt_brief_render" / "ppt_brief.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(source.resolve()), ReadOnly=True)
        document.Repaginate()
        page_count = document.ComputeStatistics(2)
        document.ExportAsFixedFormat(str(target.resolve()), 17)
        print(f"pages={page_count}")
        print(target)
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()


if __name__ == "__main__":
    main()
