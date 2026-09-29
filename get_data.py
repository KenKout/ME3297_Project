"""Lấy bộ dữ liệu DataCo Smart Supply Chain (Constante, Silva & Pereira, 2019).

Nguồn    : https://data.mendeley.com/datasets/8gx2fvg2k6/5  (DOI 10.17632/8gx2fvg2k6.5)
Giấy phép: CC BY 4.0 (được dùng cho học tập, phải trích dẫn nguồn)

Mendeley chỉ cho tải file qua giao diện web, nên cần tải file zip bằng tay
(nút "Download All", khoảng 26 MB, tên file 8gx2fvg2k6-5.zip). Script này sau đó
giải nén 2 file CSV cần dùng vào data/raw/.

Theo brief §7, KHÔNG được nộp file dữ liệu gốc; chỉ nộp script này và link tải.

Cách dùng:
    python get_data.py                      # tìm file ./8gx2fvg2k6-5.zip
    python get_data.py path/to/8gx2fvg2k6-5.zip
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
# File dữ liệu chính + file mô tả các cột (nguồn cho data dictionary).
# File tokenized_access_logs.csv (clickstream) trong zip không dùng tới.
NEEDED = ["DataCoSupplyChainDataset.csv", "DescriptionDataCoSupplyChain.csv"]


def main() -> None:
    # Đường dẫn zip: lấy từ tham số dòng lệnh nếu có, nếu không thì tìm trong thư mục project.
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "8gx2fvg2k6-5.zip"
    # Nếu đã giải nén rồi thì không làm lại.
    if all((RAW_DIR / f).exists() for f in NEEDED):
        print(f"Data already present in {RAW_DIR}")
        return
    if not zip_path.exists():
        sys.exit(
            f"Zip not found at {zip_path}.\n"
            "Download it from https://data.mendeley.com/datasets/8gx2fvg2k6/5 "
            "('Download All') and re-run: python get_data.py <path-to-zip>"
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Chỉ giải nén đúng các file cần dùng.
    with zipfile.ZipFile(zip_path) as zf:
        for name in NEEDED:
            zf.extract(name, RAW_DIR)
            print(f"Extracted {name} -> {RAW_DIR}")


if __name__ == "__main__":
    main()
