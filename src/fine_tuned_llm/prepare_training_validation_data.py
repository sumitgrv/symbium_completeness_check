"""
Prepare training + validation data in one run.
"""
import json
import os
import sys
from pathlib import Path

# Ensure project root is importable when running file path directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.completeness_common import (
    DATASET_JSONL,
    DATASET_VALIDATION_JSONL,
    IMAGE_FOLDER,
    LABELS_JSON,
    LABELS_VAL_JSON,
    PDF_FOLDER,
    create_jsonl_entry,
    logger,
    parse_image_path,
    pdf_to_images,
)


def load_labels(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def build_jsonl_for_labels(all_image_paths, labels):
    rows = []
    for img_path in all_image_paths:
        pdf_name, page_1based = parse_image_path(img_path)
        if pdf_name not in labels:
            continue
        label = labels[pdf_name]
        page_list = label.get("page_list")
        if page_list is not None and page_1based not in page_list:
            continue
        stamp = page_1based in label.get("PE Stamp", [])
        north_arrow = page_1based in label.get("north_arrow", [])
        rows.append(create_jsonl_entry(img_path, stamp, north_arrow))
    return rows


logger.info("=== Prepare training + validation data ===")
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

labels_train = load_labels(LABELS_JSON)
labels_val = load_labels(LABELS_VAL_JSON)
logger.info("%s: %d PDF(s); %s: %d PDF(s)", LABELS_JSON, len(labels_train), LABELS_VAL_JSON, len(labels_val))

pdf_names_needed = set(labels_train.keys()) | set(labels_val.keys())
if not pdf_names_needed:
    logger.warning("No PDF keys in %s or %s", LABELS_JSON, LABELS_VAL_JSON)

all_images_by_path = {}
for pdf_file in os.listdir(PDF_FOLDER):
    if not pdf_file.lower().endswith(".pdf"):
        continue
    pdf_name = pdf_file[:-4]
    if pdf_name not in pdf_names_needed:
        continue
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    for image_path in pdf_to_images(pdf_path, output_folder=IMAGE_FOLDER):
        all_images_by_path[image_path] = image_path

all_images = list(all_images_by_path.keys())
logger.info("Images under %s: %d files from %d PDF(s)", IMAGE_FOLDER, len(all_images), len(pdf_names_needed))

train_rows = build_jsonl_for_labels(all_images, labels_train)
with open(DATASET_JSONL, "w", encoding="utf-8") as f:
    for row in train_rows:
        f.write(json.dumps(row) + "\n")
logger.info("%s: %d entries", DATASET_JSONL, len(train_rows))

val_rows = build_jsonl_for_labels(all_images, labels_val)
with open(DATASET_VALIDATION_JSONL, "w", encoding="utf-8") as f:
    for row in val_rows:
        f.write(json.dumps(row) + "\n")
logger.info("%s: %d entries", DATASET_VALIDATION_JSONL, len(val_rows))
