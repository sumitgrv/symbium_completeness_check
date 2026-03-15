"""
Part 3: Prediction.
- Loads the fine-tuned model from fine_tuned_model.txt or .env (FINE_TUNED_MODEL_ID).
- Runs detection on PDF(s): all PDFs in NEW_PDF_FOLDER by default, or pass one path as argument.
- Saves results to output/fine_tuned_llm/<pdf_name>/result.json per PDF.
Run after 2_fine_tune.py. Requires OPENAI_API_KEY.
"""
import json
import os
import sys
from completeness_common import (
    logger,
    get_client,
    NEW_PDF_FOLDER,
    OUTPUT_FOLDER_FINETUNED,
    load_fine_tuned_model,
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
        "No fine-tuned model id. Run 2_fine_tune.py (writes fine_tuned_model.txt), "
        "or set FINE_TUNED_MODEL_ID in .env (e.g. ft:gpt-4o-2024-08-06:org:...)."
    )
    sys.exit(1)

client = get_client()
os.makedirs(OUTPUT_FOLDER_FINETUNED, exist_ok=True)

if len(sys.argv) > 1:
    pdf_paths = [sys.argv[1]]
else:
    if not os.path.isdir(NEW_PDF_FOLDER):
        logger.error(
            "NEW_PDF_FOLDER %s not found. Usage: python 3_predict.py [path/to/file.pdf]",
            NEW_PDF_FOLDER,
        )
        sys.exit(1)
    pdf_paths = [
        os.path.join(NEW_PDF_FOLDER, f)
        for f in sorted(os.listdir(NEW_PDF_FOLDER))
        if f.lower().endswith(".pdf")
    ]
    if not pdf_paths:
        logger.error("No PDFs in %s. Pass a path: python 3_predict.py path/to/file.pdf", NEW_PDF_FOLDER)
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
    for p in predictions:
        logger.info("[%s] %s", os.path.basename(pdf_path), p)
        row = {
            "image": p.get("image"),
            "prediction_raw": p.get("prediction"),
        }
        try:
            row["prediction_parsed"] = json.loads(p.get("prediction") or "{}")
        except json.JSONDecodeError:
            row["prediction_parsed"] = None
        pages.append(row)

    payload = {
        "pdf_path": pdf_path,
        "pdf_name": pdf_basename,
        "model_id": model_id,
        "pages": pages,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", result_path)
