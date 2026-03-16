"""
Fine-tuning script.
"""
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable when running file path directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.completeness_common import (
    DATASET_JSONL,
    DATASET_VALIDATION_JSONL,
    FINE_TUNE_BASE_MODEL,
    FINE_TUNED_MODEL_FILE,
    get_client,
    logger,
    save_fine_tuned_model,
)

logger.info("=== Part 2: Fine-tuning ===")
client = get_client()

logger.info("Uploading training file: %s", DATASET_JSONL)
with open(DATASET_JSONL, "rb") as f:
    train_id = client.files.create(file=f, purpose="fine-tune").id
logger.info("Training file id: %s", train_id)

val_id = None
if os.path.isfile(DATASET_VALIDATION_JSONL) and os.path.getsize(DATASET_VALIDATION_JSONL) > 0:
    with open(DATASET_VALIDATION_JSONL, "rb") as f:
        first = f.readline()
    if first.strip():
        with open(DATASET_VALIDATION_JSONL, "rb") as f:
            val_id = client.files.create(file=f, purpose="fine-tune").id
        logger.info("Validation file uploaded: %s id=%s", DATASET_VALIDATION_JSONL, val_id)
    else:
        logger.info("Skipping validation: %s is empty", DATASET_VALIDATION_JSONL)
else:
    logger.info("No validation file or empty — train only")

kwargs = {
    "model": FINE_TUNE_BASE_MODEL,
    "training_file": train_id,
    "hyperparameters": {"n_epochs": 5},
}
if val_id:
    kwargs["validation_file"] = val_id

logger.info("Starting fine-tune job (model=%s, n_epochs=5)", FINE_TUNE_BASE_MODEL)
job_id = client.fine_tuning.jobs.create(**kwargs).id
logger.info("Fine-tune job started: %s", job_id)

logger.info("Waiting for job (polling every 60s)...")
while True:
    job = client.fine_tuning.jobs.retrieve(job_id)
    status = job.status
    logger.info("Job status: %s", status)
    if status == "succeeded":
        if job.fine_tuned_model:
            save_fine_tuned_model(job.fine_tuned_model)
        else:
            logger.warning("Job succeeded but fine_tuned_model empty.")
        break
    if status == "failed":
        logger.error("Fine-tune failed. Job id: %s", job_id)
        raise RuntimeError("Fine-tune job failed")
    time.sleep(60)

logger.info("Done. Model saved to %s", FINE_TUNED_MODEL_FILE)
