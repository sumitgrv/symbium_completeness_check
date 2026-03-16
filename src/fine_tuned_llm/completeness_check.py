"""
Completeness check using fine-tuned LLM.
Full pipeline: prepare data -> fine-tune -> predict.
"""
import runpy


if __name__ == "__main__":
    runpy.run_module("src.fine_tuned_llm.prepare_training_validation_data", run_name="__main__")
    runpy.run_module("src.fine_tuned_llm.fine_tune", run_name="__main__")
    runpy.run_module("src.fine_tuned_llm.predict", run_name="__main__")
