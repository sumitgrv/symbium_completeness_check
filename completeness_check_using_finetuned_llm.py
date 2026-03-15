"""
Completeness check using fine-tuned LLM.
Full pipeline: prepare data → fine-tune → predict.
  python 1_prepare_training_validation_data.py
  python 2_fine_tune.py
  python 3_predict.py [optional/path/to/file.pdf]
Results are saved to output/fine_tuned_llm/<pdf_name>/result.json
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("1_prepare_training_validation_data.py", run_name="__main__")
    runpy.run_path("2_fine_tune.py", run_name="__main__")
    runpy.run_path("3_predict.py", run_name="__main__")
