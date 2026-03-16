# Permit Plan Completeness Check

Permit plan completeness checking for:
- PE stamp detection
- North arrow detection

This repo supports two inference approaches:
- Fine-tuned vision model pipeline (`src/fine_tuned_llm`)
- Prompt-based standard LLM pipeline (`src/standard_llm`)

It also includes demo apps in `demo/`:
- FastAPI service
- Streamlit UI

## Project Structure

- `src/common/`: shared config, prompt rules, helpers
- `src/fine_tuned_llm/`: training data preparation, fine-tuning, batch prediction
- `src/standard_llm/`: prompt-based prediction + few-shot examples
- `demo/`: `app.py` (API) and `streamlit_app.py` (UI)
- `examples/`: few-shot reference images (stamp, north_direction)
- `pdfs/`: labeled training PDFs
- `new_pdfs/`: inference PDFs

## Installation Instructions

### 1) Python environment

Use Python 3.10 for reproducible local runs across scripts, and demos.
If multiple Python versions are installed, always use the same interpreter for both package install and execution.

```bash
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick interpreter check:

```bash
python -c "import sys; print(sys.executable)"
```

### 2) Poppler (required for PDF conversion)

`pdf2image` needs Poppler installed and available in PATH.

- Windows: install from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)

### 3) Environment variables

Create/update `.env`:

```env
OPENAI_API_KEY=your_key_here
FINE_TUNE_BASE_MODEL=gpt-4o-2024-08-06
# Optional:
# FINE_TUNED_MODEL_ID=ft:...
# OPENAI_BASE_MODELS=gpt-4.1,gpt-4o,gpt-4o-mini
# STAMP_EXAMPLES_PATH=absolute\path\to\examples\stamp
# NORTH_DIR_EXAMPLES_PATH=absolute\path\to\examples\north_direction
```

## Data Preparation Techniques (Fine-Tuned Flow)

Implemented in `src/fine_tuned_llm/prepare_training_validation_data.py`.

How data preparation works:
- Loads `labels.json` (train) and `labels_val.json` (validation)
- Uses union of referenced PDF names from both label files
- Converts each PDF once to page images under `images/<pdf_name>/`
- Generates:
  - `vision_train.jsonl`
  - `vision_validation.jsonl`
- Each JSONL row stores:
  - system prompt + user instruction
  - page image as base64 data URL
  - assistant target JSON: `{"stamp": bool, "north_arrow": bool}`

Label mapping details:
- `PE Stamp` list -> `stamp=True` on listed pages
- `north_arrow` list -> `north_arrow=True` on listed pages
- Optional `page_list` narrows labeled pages per PDF

### Template: `labels.json` and `labels_val.json`

Both files use the same schema.  
Top-level keys must match PDF file names **without** `.pdf`.

```json
{
  "PDF_NAME_WITHOUT_EXTENSION": {
    "page_list": [1, 2, 3],
    "PE Stamp": [1, 3],
    "north_arrow": [2, 3]
  },
  "ANOTHER_PDF_NAME": {
    "page_list": [1, 2, 3],
    "PE Stamp": [1],
    "north_arrow": []
  }
}
```

Field notes:
- `page_list` (optional): pages to consider for that PDF (1-based page numbers).
- `PE Stamp`: pages where PE stamp is present.
- `north_arrow`: pages where north arrow is present.

You can also add metadata keys prefixed with `_` (for comments/tracking), and they will be ignored by the loader.

Example:

```json
{
  "_note": "training labels v1",
  "ProjectA_SheetSet": {
    "page_list": [1, 2, 4],
    "PE Stamp": [1, 4],
    "north_arrow": [2]
  }
}
```

## Completeness Check Run Instructions

## Fine-Tuned LLM

### A) Full pipeline (prepare -> fine-tune -> predict)

```bash
python src/fine_tuned_llm/completeness_check.py
```

### B) Step-by-step

```bash
python src/fine_tuned_llm/prepare_training_validation_data.py
python src/fine_tuned_llm/fine_tune.py
python src/fine_tuned_llm/predict.py
```

Predict one PDF:

```bash
python src/fine_tuned_llm/predict.py path\to\file.pdf
```

Output folder:
- `output/fine_tuned_llm/<pdf_name>/result.json`

## Standard LLM (Prompt-Based)

Run all PDFs from `new_pdfs/`:

```bash
python src/standard_llm/completeness_check.py
```

Run one PDF:

```bash
python src/standard_llm/completeness_check.py path\to\file.pdf
```

Run one image:

```bash
python src/standard_llm/completeness_check.py path\to\image.png
```

Output folder:
- `output/standard_llm/<pdf_name>/result.json`

## Demo App Instructions

## 1) Streamlit UI

```bash
streamlit run demo/streamlit_app.py
```

What you can do in UI:
- Select approach: Prompt-Based or Fine-Tuned
- Choose base model (prompt-based)
- Select/paste fine-tuned model ID (fine-tuned)
- Upload image and view PE stamp + North arrow results

## 2) FastAPI Service

Run API:

```bash
python demo/app.py
```

or:

```bash
uvicorn demo.app:app --reload
```

Endpoints:
- `GET /` basic service info
- `GET /health` model configuration health
- `POST /predict` image upload prediction

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "image=@C:\path\to\sheet.png"
```