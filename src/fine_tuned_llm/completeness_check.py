"""
Completeness check using fine-tuned LLM.
Full pipeline: prepare data -> fine-tune -> predict.
"""
import sys
from pathlib import Path
import runpy

# Ensure project root is importable when running file path directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.completeness_common import logger


if __name__ == "__main__":
    logger.info("Starting fine-tuned completeness pipeline")
    runpy.run_module("src.fine_tuned_llm.prepare_training_validation_data", run_name="__main__")
    runpy.run_module("src.fine_tuned_llm.fine_tune", run_name="__main__")
    runpy.run_module("src.fine_tuned_llm.predict", run_name="__main__")
    logger.info("Fine-tuned completeness pipeline finished successfully")
