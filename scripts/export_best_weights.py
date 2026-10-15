"""从已核验的正式结果包提取四组最佳权重，供 Git 仓库中的推理示例使用。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs" / "results_bundle_e100_complete.zip"
TARGET = ROOT / "weights"
VARIANTS = ("baseline", "eca", "p2", "eca_p2")


def main() -> None:
    TARGET.mkdir(exist_ok=True)
    checksums: list[str] = []
    with ZipFile(BUNDLE) as archive:
        for variant in VARIANTS:
            source = f"outputs/runs/{variant}_e100/weights/best.pt"
            data = archive.read(source)
            if len(data) >= 100_000_000:
                raise ValueError(f"单个权重超过 GitHub 常规文件限制：{source}")
            name = f"{variant}_e100.pt"
            (TARGET / name).write_bytes(data)
            checksums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (TARGET / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="ascii")
    for line in checksums:
        print(line)


if __name__ == "__main__":
    main()
