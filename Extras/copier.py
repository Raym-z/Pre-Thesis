import csv
import shutil
from pathlib import Path

# ——— Configuration ———
CSV_PATH   = Path("images2\manifest10k.csv")      # your CSV file
SRC_BASE   = Path("images2")        # root folder containing content/, styles/, stylizations/
DST_BASE   = Path("images")         # new root folder to copy into
# ————————————————————

def main():
    # ensure destination root exists
    DST_BASE.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # row["path"] might use backslashes on Windows
            src_path = Path(row["path"])
            # Make sure it's under SRC_BASE
            try:
                rel_path = src_path.relative_to(SRC_BASE)
            except ValueError:
                print(f"Skipping {src_path}: not under {SRC_BASE}")
                continue

            dst_path = DST_BASE / rel_path
            # create any needed subdirectories (e.g. images/styles/)
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy(src_path, dst_path)
                print(f"Copied {src_path} {dst_path}")
            except FileNotFoundError:
                print(f"Source not found: {src_path}")
            except Exception as e:
                print(f"Error copying {src_path}: {e}")

if __name__ == "__main__":
    main()
