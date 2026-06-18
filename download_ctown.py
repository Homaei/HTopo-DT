"""
Download C-Town (BATADAL) dataset and EPANET INP file.
Run this script from your project root:
    python3 Download_ctown.py
"""

import os
import urllib.request
import zipfile
import shutil

# Dataset dir is located one level up from this code file
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

FILES = {
    # BATADAL training set 1 (normal operations without attacks)
    "BATADAL_dataset03.csv": "http://www.batadal.net/data/BATADAL_dataset03.csv",
    # BATADAL training set 2 (with attacks)
    "BATADAL_dataset04.csv": "http://www.batadal.net/data/BATADAL_dataset04.csv",
    # BATADAL test set (zipped)
    "BATADAL_test_dataset.zip": "http://www.batadal.net/data/BATADAL_test_dataset.zip",
    # C-Town EPANET INP file from a stable SCEPTRE-Lab repository
    "ctown.inp": "https://raw.githubusercontent.com/SCEPTRE-Lab/EPANET-based-Digital-Twin-of-Water-Distributions-Networks/master/CTOWN.INP",
}


def download_file(url: str, dest: str) -> bool:
    print(f"  Downloading {os.path.basename(dest)} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        size_kb = os.path.getsize(dest) / 1024
        print(f"OK  ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"FAILED  ({e})")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def verify() -> None:
    print("\n── Verification ──────────────────────────────")
    expected = [
        "BATADAL_dataset03.csv",
        "BATADAL_dataset04.csv",
        "BATADAL_test_dataset.csv",
        "ctown.inp",
    ]
    all_ok = True
    for name in expected:
        path = os.path.join(DATASET_DIR, name)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✅  {name:<35}  {size_kb:>8.1f} KB")
        else:
            print(f"  ❌  {name:<35}  MISSING")
            all_ok = False

    print()
    if all_ok:
        print("All files present. dataset/ folder is ready.")
        print(f"Path: {os.path.abspath(DATASET_DIR)}")
    else:
        print("Some files are missing — check the errors above.")
        print("Manual download links:")
        print("  BATADAL: http://www.batadal.net/data/")
        print("  C-Town INP: https://github.com/SCEPTRE-Lab/EPANET-based-Digital-Twin-of-Water-Distributions-Networks")


def main() -> None:
    print("═" * 54)
    print("  HTopo-DT — Dataset Downloader")
    print(f"  Target folder: {os.path.abspath(DATASET_DIR)}")
    print("═" * 54)

    # ── Step 1: BATADAL CSVs ──────────────────────────────────
    print("\n[1/2] BATADAL datasets (SCADA time-series + labels)")
    
    csv_files = ["BATADAL_dataset03.csv", "BATADAL_dataset04.csv"]
    for fname in csv_files:
        dest = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(dest):
            download_file(FILES[fname], dest)
        else:
            print(f"  {fname} already exists, skipping.")
            
    # Test dataset is zipped
    test_csv = os.path.join(DATASET_DIR, "BATADAL_test_dataset.csv")
    test_zip = os.path.join(DATASET_DIR, "BATADAL_test_dataset.zip")
    if not os.path.exists(test_csv):
        if download_file(FILES["BATADAL_test_dataset.zip"], test_zip):
            print("  Extracting BATADAL_test_dataset.csv ...", end=" ", flush=True)
            try:
                with zipfile.ZipFile(test_zip, "r") as z:
                    for member in z.namelist():
                        if member.endswith(".csv"):
                            with z.open(member) as src, open(test_csv, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            print("OK")
                            break
            except Exception as e:
                print(f"FAILED ({e})")
            finally:
                if os.path.exists(test_zip):
                    os.remove(test_zip)
    else:
        print(f"  BATADAL_test_dataset.csv already exists, skipping.")

    # ── Step 2: C-Town INP ───────────────────────────────────
    print("\n[2/2] C-Town EPANET hydraulic model (ctown.inp)")
    ctown_dest = os.path.join(DATASET_DIR, "ctown.inp")
    if os.path.exists(ctown_dest):
        print(f"  ctown.inp already exists, skipping.")
    else:
        download_file(FILES["ctown.inp"], ctown_dest)

    verify()


if __name__ == "__main__":
    main()