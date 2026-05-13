"""
Download the USA Real Estate Dataset from Kaggle.

Usage:
    python download_data.py

Requires: pip install kagglehub
The dataset will be saved to data/realtor-data.csv
"""
import os
import shutil
import kagglehub

def main():
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "realtor-data.csv")

    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return

    print("Downloading USA Real Estate Dataset from Kaggle...")
    download_path = kagglehub.dataset_download(
        "ahmedshahriarsakib/usa-real-estate-dataset"
    )

    src = os.path.join(download_path, "realtor-data.zip.csv")
    shutil.copy2(src, output_path)
    print(f"Dataset saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
