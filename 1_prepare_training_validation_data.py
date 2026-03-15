"""
Prepare training + validation data in one run.

- PDFs live in pdfs/
- labels.json  → which PDFs/pages are TRAINING → vision_train.jsonl
- labels_val.json → which PDFs/pages are VALIDATION → vision_validation.jsonl
- All page images: images/<pdf_name>/<pdf_name>_page_N.png (N = 1, 2, 3...; 1-based to match labels)

Convert each PDF at most once (union of train + val PDF names).
Train/val must not duplicate the same JSONL rows (prefer different PDFs or disjoint pages).
"""
import os
import json
from completeness_common import (
    logger,
    DATASET_JSONL,
    DATASET_VALIDATION_JSONL,
    LABELS_JSON,
    LABELS_VAL_JSON,
    PDF_FOLDER,
    IMAGE_FOLDER,
    pdf_to_images,
    parse_image_path,
    create_jsonl_entry,
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
        L = labels[pdf_name]
        pl = L.get("page_list")
        if pl is not None and page_1based not in pl:
            continue
        stamp = page_1based in L.get("PE Stamp", [])
        north_arrow = page_1based in L.get("north_arrow", [])
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

# Convert every PDF that appears in either label file (once → images/<name>/)
all_images_by_path = {}  # path -> path (dedupe)
for pdf_file in os.listdir(PDF_FOLDER):
    if not pdf_file.lower().endswith(".pdf"):
        continue
    pdf_name = pdf_file[:-4]
    if pdf_name not in pdf_names_needed:
        continue
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    for p in pdf_to_images(pdf_path, output_folder=IMAGE_FOLDER):
        all_images_by_path[p] = p

all_images = list(all_images_by_path.keys())
logger.info("Images under %s: %d files from %d PDF(s)", IMAGE_FOLDER, len(all_images), len(pdf_names_needed))

# Training JSONL
train_rows = build_jsonl_for_labels(all_images, labels_train)
with open(DATASET_JSONL, "w", encoding="utf-8") as f:
    for e in train_rows:
        f.write(json.dumps(e) + "\n")
logger.info("%s: %d entries", DATASET_JSONL, len(train_rows))

# Validation JSONL
val_rows = build_jsonl_for_labels(all_images, labels_val)
with open(DATASET_VALIDATION_JSONL, "w", encoding="utf-8") as f:
    for e in val_rows:
        f.write(json.dumps(e) + "\n")
logger.info("%s: %d entries", DATASET_VALIDATION_JSONL, len(val_rows))
