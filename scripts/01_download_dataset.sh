#!/usr/bin/env bash
# =============================================================================
# 01_download_dataset.sh
# Download the Kaggle Ship Detection dataset
# =============================================================================
# Prerequisites:
#   1. pip install kaggle
#   2. Create a Kaggle account and go to Account → API → Create New Token
#      This downloads kaggle.json — place it at ~/.kaggle/kaggle.json
#      Then run: chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   bash scripts/01_download_dataset.sh
# =============================================================================

set -euo pipefail

DATASET="andrewmvd/ship-detection"
DEST="./raw_dataset"

echo "==> Checking for kaggle CLI..."
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: 'kaggle' not found. Run: pip install kaggle"
    exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found."
    echo "       Download your API token from https://www.kaggle.com/account"
    exit 1
fi

echo "==> Downloading dataset: $DATASET"
mkdir -p "$DEST"
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip

echo ""
echo "==> Download complete. Contents of $DEST:"
ls -lh "$DEST"

echo ""
echo "==> Expected structure after unzip:"
echo "    raw_dataset/"
echo "    ├── images/        (JPEG aerial images)"
echo "    └── annotations/   (Pascal VOC XML files)"
echo ""
echo "==> Next step: python scripts/02_convert_voc_to_yolo.py"
