# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

import os
import sys
import requests
from tqdm import tqdm

HF_REPO = "davidscripka/openwakeword_features"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

FILES = {
    "ACAV100M": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "validation": "validation_set_features.npy",
}


def download_file(url, dest_path, desc="Downloading"):
    resume_pos = 0
    if os.path.exists(dest_path):
        resume_pos = os.path.getsize(dest_path)

    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    response = requests.get(url, stream=True, headers=headers, timeout=30)
    total_size = int(response.headers.get("Content-Length", 0))
    if "Content-Range" in response.headers:
        total_size = int(response.headers["Content-Range"].split("/")[-1])

    mode = "ab" if resume_pos > 0 else "wb"
    initial = resume_pos

    with open(dest_path, mode) as f:
        with tqdm(
            total=total_size,
            initial=initial,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    actual_size = os.path.getsize(dest_path)
    if total_size > 0 and actual_size != total_size:
        print(f"[WARN] Expected {total_size} bytes, got {actual_size}. File may be incomplete.")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download openwakeword_features background data from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save downloaded .npy files (default: current dir)",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Also download the validation set features (185 MB)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Download main ACAV100M file (17.3 GB)
    url = f"{HF_BASE}/{FILES['ACAV100M']}"
    dest = os.path.join(args.output_dir, FILES["ACAV100M"])
    print(f"Downloading ACAV100M background features (~17.3 GB)...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")
    download_file(url, dest, desc="ACAV100M")

    # Optionally download validation set
    if args.include_validation:
        url_val = f"{HF_BASE}/{FILES['validation']}"
        dest_val = os.path.join(args.output_dir, FILES["validation"])
        print(f"\nDownloading validation set features (~185 MB)...")
        download_file(url_val, dest_val, desc="Validation")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
