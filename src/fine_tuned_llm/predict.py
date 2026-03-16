"""
Prediction script for fine-tuned model.
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
    NEW_PDF_FOLDER,
    OUTPUT_FOLDER_FINETUNED,
    get_client,
    load_fine_tuned_model,
    logger,
    predict_from_pdf,
)

logger.info("=== Part 3: Prediction ===")

model_id = load_fine_tuned_model()
if not model_id:
    model_id = (os.getenv("FINE_TUNED_MODEL_ID") or os.getenv("MODEL_ID") or "").strip() or None
    if model_id:
        logger.info("Using model id from .env (FINE_TUNED_MODEL_ID or MODEL_ID)")
if not model_id:
    logger.error(
        "No fine-tuned model id. Run fine_tune.py first (writes fine_tuned_model.txt), "
        "or set FINE_TUNED_MODEL_ID in .env."
    )
    sys.exit(1)

client = get_client()
os.makedirs(OUTPUT_FOLDER_FINETUNED, exist_ok=True)

if len(sys.argv) > 1:
    pdf_paths = [sys.argv[1]]
else:
    if not os.path.isdir(NEW_PDF_FOLDER):
        logger.error(
            "NEW_PDF_FOLDER %s not found. Usage: python src/fine_tuned_llm/predict.py [path/to/file.pdf]",
            NEW_PDF_FOLDER,
        )
        sys.exit(1)
    pdf_paths = [
        os.path.join(NEW_PDF_FOLDER, f)
        for f in sorted(os.listdir(NEW_PDF_FOLDER))
        if f.lower().endswith(".pdf")
    ]
    if not pdf_paths:
        logger.error("No PDFs in %s. Pass a path: python src/fine_tuned_llm/predict.py path/to/file.pdf", NEW_PDF_FOLDER)
        sys.exit(1)
    logger.info("Predicting on %d PDF(s) in %s", len(pdf_paths), NEW_PDF_FOLDER)

for pdf_path in pdf_paths:
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(OUTPUT_FOLDER_FINETUNED, pdf_basename)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")

    logger.info("Running prediction on: %s", pdf_path)
    predictions = predict_from_pdf(client, pdf_path, model_id=model_id)

    pages = []
    for prediction in predictions:
        logger.info("[%s] %s", os.path.basename(pdf_path), prediction)
        row = {"image": prediction.get("image"), "prediction_raw": prediction.get("prediction")}
        try:
            row["prediction_parsed"] = json.loads(prediction.get("prediction") or "{}")
        except json.JSONDecodeError:
            row["prediction_parsed"] = None
        pages.append(row)

    payload = {"pdf_path": pdf_path, "pdf_name": pdf_basename, "model_id": model_id, "pages": pages}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", result_path)
