"""
Completeness check using standard (prompt-based) vision LLM (e.g. gpt-4o).
Reads all PDFs in new_pdfs, runs page-by-page stamp and north-arrow detection,
and saves results to output/standard_llm/<pdf_name>/result.json.

Usage:
  python completeness_check_using_standard_llm.py
      → process all PDFs in new_pdfs/
  python completeness_check_using_standard_llm.py path/to/file.pdf
      → process that PDF only
  python completeness_check_using_standard_llm.py path/to/image.png
      → single-image prediction (prints to stdout)

Requires: OPENAI_API_KEY. Optional: OPENAI_VISION_MODEL (default gpt-4o).
"""
import base64
import json
import os
import re
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from completeness_common import (
    logger,
    get_client,
    NEW_PDF_FOLDER,
    OUTPUT_FOLDER_STANDARD,
    pdf_to_images,
)

# Load few-shot examples from few_shot_examples (uses project examples/ folder)
try:
    from few_shot_examples.stamp_examples import STAMP_EXAMPLES
    from few_shot_examples.north_direction_examples import NORTH_DIR_EXAMPLES
except Exception as e:
    logger.warning("Few-shot examples not loaded: %s. Using no examples.", e)
    STAMP_EXAMPLES = []
    NORTH_DIR_EXAMPLES = []

# Default model for prompt-based vision (not the fine-tuned one)
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

# JSON schemas injected into prompts (PE stamp only; City stamp not checked)
STAMP_JSON_SCHEMA = {
    "checkStampPresence": "Yes or No",
    "CheckStampType": [
        {
            "ProfessionalEngineeringStamp": "Yes or No",
        }
    ],
}
NORTH_ARROW_JSON_SCHEMA = {
    "NorthDirectionSymbol": "Detected or Not Detected",
}

# ---------------------------------------------------------------------------
# Exact prompts (as provided)
# ---------------------------------------------------------------------------

STAMP_DETECTION_DESCRIPTION = """
### Definition of valid stamp (PE only)

#### Professional Engineer (PE) stamps
A **Professional Engineer (PE) stamp** demonstrates that a professional engineer placed his/her "registration seal" on the drawing or designs. These typically appear as an official seal or approval mark used for professional certification, company validation, or approval.

- Common Textual Elements found in professional engineer stamps:
    - **Profession Title**: Phrases like "LICENSED ENGINNER" or "PROFESSIONAL ENGINNER" or "LICENSED PROFESSIONAL ENGINNER".
    - **State**: A U.S. state or Canadian province, such as "STATE OF CALIFORNIA" or "STATE OF NEW YORK".
    - **License number**
    - **Expiration date**: e.g., "EXP 03/25")
    - **Descipline**: The engineer's field of specialization, such as "CIVIL", "STRUCTURAL", "MECHANICAL", etc.
- Design Characteristics of **Professional Engineer (PE) stamps**:
    - **Shape**: **circular, rectangular, or any other shape**.
    - **Structure**: Contains **printed, structured text** rather than handwriting.
- **Note**: Don't Considered an item as professional engineer stamps if **Professional Title** are not available inside it.

### Important Instructions
- Only detect structured, official **PE stamps** (do not consider City/AHJ approval stamps).
- Ignore handwritten marks or signatures by themselves; these are not stamps.
- Do not confuse logos, decorative symbols, or abstract shapes with stamps.
- A valid PE stamp may be partially covered by a signature—focus on the structured, printed portion.

--- 

You will be shown an image of a document. Based on the definition and characteristics above, determine whether the image contains a **valid PE stamp**.

---

### Output Format
Return a structured JSON response that follows this schema exactly:

{json_schema_str}

- Populate `checkStampPresence` with **"Yes"** if a **PE Stamp** is present on the sheet, and **"No"** otherwise.
- In `CheckStampType[0]`, set `ProfessionalEngineeringStamp` to **"Yes"** or **"No"** to indicate PE stamp detection.
"""

NORTH_ARROW_DETECTION_DESCRIPTION = """
Carefully examine the image and determine whether it contains a geographical **North direction Symbol** and/or a **Scale Indicator**.

---

### Definition of a North Direction Symbol:
- **Definitions**: A **North Direction Symbol** is a geographical directional symbol used to indicate geographic orientation in technical diagrams, such as floor plans or site plans.
- **Expected values**: `"Detected"` if present else `"Not Detected"`.
- **Visual Indicators**:
    - A **North Direction Symbol** is often available with label "North" pointing in one direction.
    - May include a full compass rose (N, NE, E, etc.) or a single geographical direction symbol with just the letter "N".
    - Is often labeled with the word **"NORTH"** or simply the letter **"N"**.
    - May appear **near the scale indicator**, often in the **corner or edge** of a drawing.

**Assumption**: If a North Direction Symbol is present, a scale is **likely to co-occur nearby** or on the same page.

---

### Before concluding detection, verify:
- Does the arrow explicitly show geographic direction (not flow or diagram arrows)?
- Is the arrow paired with "NORTH"?
- Is it in a logical architectural position (e.g., corner, title block)?

---

### Important Instructions:
- **Only detect the North Direction Symbol and its associated scale**, if visible.
- Look for **standard architectural or civil drawing conventions** — geographical compass or north direction symbol and scales.
- Do **not** treat direction-like symbols embedded inside the site diagram or near object labels (e.g., house, driveway) as valid North Arrows.
- Do **not** interpret "(N)" in equipment labels (e.g., "(N) Inverter", "(N) Panel", "(E) House", "(N) PV Sytem) as a North direction symbol. These are **installation phase indicators** (e.g., New, Existing), not geographic directions.
- Do **not** consider direction arrow/annotations pointing towards the diagram as North Direction Symbol.
- Do **not** confuse decorative arrows, flow arrows, or compass-like logos with a true North Direction Symbol.
- Absolutely **do not treat any label in parentheses**, such as "(N)", "(E)", "(R)", or "(P)", as a North Direction Arrow. These are **not geographic directional symbols** — they indicate project phases:
    - (N) = New
    - (E) = Existing
    - (R) = Relocated
    - (P) = Proposed
- **Never treat label (N)** found in front of any equipment name (e.g., "(N) Inverter", "(N) PV System", "(E) Battery) as a North direction marker — even if they are near arrows or architectural elements.
- A valid **North Arrow must be a graphic directional symbol**, with an arrow clearly pointing and labeled with "NORTH", **not embedded in parentheses**.

---

### Output Format:
Detect the North direction symbol and scale indicator and return a structured JSON-like response based on what is detected:

{json_schema_str}
"""

STAMP_SYSTEM_MESSAGE = (
    "You are a stamp detector assistant. Examine the page carefully and determine if it contains an official **Professional Engineer (PE) stamp** based on structured printed text (e.g. profession title, state, license). Do not consider City or AHJ approval stamps. Refer to the few-shot examples to see what a valid PE stamp looks like."
)
NORTH_ARROW_SYSTEM_MESSAGE = (
    "You are a helpful assistant specialized in detecting geographic symbols in architectural, site, and construction drawings. Your task is to visually analyze the layout and determine whether the image contains a North Direction Symbol and/or a Scale Indicator. Refer to the few-shot examples to understand what a valid North direction symbol looks like."
)


def encode_image(image_path_or_bytes) -> str:
    """Encode image to base64 data URL. image_path_or_bytes: path (str) or bytes."""
    if isinstance(image_path_or_bytes, bytes):
        b64 = base64.standard_b64encode(image_path_or_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"
    with open(image_path_or_bytes, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def extract_json_object(text: str) -> dict:
    """Try to extract a JSON object from model output (handles markdown code blocks)."""
    text = text.strip()
    for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"(\{[\s\S]*\})"):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _few_shot_stamp_messages():
    """Build list of user/assistant message pairs for stamp few-shot (like FewShotStampExamples().to_messages())."""
    messages = []
    for idx, ex in enumerate(STAMP_EXAMPLES):
        image_path = ex.get("image_path")
        if not image_path or not os.path.isfile(image_path):
            logger.debug("Skipping stamp example (file not found): %s", image_path)
            continue
        desc = ex.get("description", "")
        expected = ex.get("expected_response")
        if expected is None:
            continue
        image_url = encode_image(image_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Few-shot example {idx + 1}: {desc}"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        })
        messages.append({"role": "assistant", "content": json.dumps(expected)})
    return messages


def _few_shot_north_messages():
    """Build list of user/assistant message pairs for north direction few-shot (like FewShotNorthDirectionExamples().to_messages())."""
    messages = []
    for idx, ex in enumerate(NORTH_DIR_EXAMPLES):
        image_path = ex.get("image_path")
        if not image_path or not os.path.isfile(image_path):
            logger.debug("Skipping north direction example (file not found): %s", image_path)
            continue
        desc = ex.get("description", "")
        expected = ex.get("expected_response")
        if expected is None:
            continue
        image_url = encode_image(image_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Few-shot example {idx + 1}: {desc}"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        })
        messages.append({"role": "assistant", "content": json.dumps(expected)})
    return messages


def run_stamp_detection(client: OpenAI, image_data, model: str = DEFAULT_VISION_MODEL) -> dict:
    """Run stamp detection using prompt-based standard LLM with few-shot examples."""
    json_schema_str = json.dumps(STAMP_JSON_SCHEMA, indent=2)
    user_text = STAMP_DETECTION_DESCRIPTION.format(json_schema_str=json_schema_str)
    image_url = image_data if isinstance(image_data, str) and image_data.startswith("data:") else encode_image(image_data)
    messages = [
        {"role": "system", "content": STAMP_SYSTEM_MESSAGE},
        {"role": "user", "content": user_text},
        *_few_shot_stamp_messages(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Following is the actual input image."},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    raw = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return extract_json_object(raw)


def run_north_arrow_detection(client: OpenAI, image_data, model: str = DEFAULT_VISION_MODEL) -> dict:
    """Run north direction / scale detection using prompt-based standard LLM with few-shot examples."""
    json_schema_str = json.dumps(NORTH_ARROW_JSON_SCHEMA, indent=2)
    user_text = NORTH_ARROW_DETECTION_DESCRIPTION.format(json_schema_str=json_schema_str)
    image_url = image_data if isinstance(image_data, str) and image_data.startswith("data:") else encode_image(image_data)
    messages = [
        {"role": "system", "content": NORTH_ARROW_SYSTEM_MESSAGE},
        {"role": "user", "content": user_text},
        *_few_shot_north_messages(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Following is the actual input image."},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    raw = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return extract_json_object(raw)


def predict(client: OpenAI, image_path_or_bytes, model: str = None):
    """Run both stamp and north-arrow prompt-based detection. Returns dict with stamp_result and north_arrow_result."""
    model = model or DEFAULT_VISION_MODEL
    stamp_result = run_stamp_detection(client, image_path_or_bytes, model=model)
    north_arrow_result = run_north_arrow_detection(client, image_path_or_bytes, model=model)
    return {"stamp_result": stamp_result, "north_arrow_result": north_arrow_result}


def predict_from_pdf(client: OpenAI, pdf_path: str, model: str = None):
    """Run page-by-page stamp and north-arrow detection on a PDF. Returns list of {image, stamp_result, north_arrow_result}."""
    model = model or DEFAULT_VISION_MODEL
    logger.info("Running standard-LLM prediction on PDF: %s (model=%s)", pdf_path, model)
    image_paths = pdf_to_images(pdf_path)
    results = []
    for idx, img_path in enumerate(image_paths):
        logger.info("Page %d/%d: %s", idx + 1, len(image_paths), img_path)
        out = predict(client, img_path, model=model)
        results.append({
            "image": img_path,
            "stamp_result": out["stamp_result"],
            "north_arrow_result": out["north_arrow_result"],
        })
    logger.info("Standard-LLM prediction complete for %s: %d page(s)", pdf_path, len(results))
    return results


def run_pdfs(pdf_paths: list, model: str = None):
    """Process one or more PDFs and save result.json under output/standard_llm/<pdf_name>/."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = get_client()
    model = model or DEFAULT_VISION_MODEL
    os.makedirs(OUTPUT_FOLDER_STANDARD, exist_ok=True)

    for pdf_path in pdf_paths:
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        out_dir = os.path.join(OUTPUT_FOLDER_STANDARD, pdf_basename)
        os.makedirs(out_dir, exist_ok=True)
        result_path = os.path.join(out_dir, "result.json")

        predictions = predict_from_pdf(client, pdf_path, model=model)
        pages = []
        for p in predictions:
            row = {
                "image": p["image"],
                "stamp_result": p["stamp_result"],
                "north_arrow_result": p["north_arrow_result"],
            }
            pages.append(row)

        payload = {
            "pdf_path": pdf_path,
            "pdf_name": pdf_basename,
            "model_id": model,
            "pages": pages,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved results to %s", result_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
        if not os.path.isfile(path_arg):
            logger.error("File not found: %s", path_arg)
            sys.exit(1)
        if path_arg.lower().endswith(".pdf"):
            run_pdfs([path_arg])
        else:
            # Single image: print results to stdout
            client = get_client()
            result = predict(client, path_arg)
            print("Stamp detection:", json.dumps(result["stamp_result"], indent=2))
            print("North arrow / scale:", json.dumps(result["north_arrow_result"], indent=2))
    else:
        if not os.path.isdir(NEW_PDF_FOLDER):
            logger.error("NEW_PDF_FOLDER %s not found. Usage: python completeness_check_using_standard_llm.py [path/to/file.pdf]", NEW_PDF_FOLDER)
            sys.exit(1)
        pdf_paths = [
            os.path.join(NEW_PDF_FOLDER, f)
            for f in sorted(os.listdir(NEW_PDF_FOLDER))
            if f.lower().endswith(".pdf")
        ]
        if not pdf_paths:
            logger.error("No PDFs in %s. Pass a path: python completeness_check_using_standard_llm.py path/to/file.pdf", NEW_PDF_FOLDER)
            sys.exit(1)
        logger.info("Processing %d PDF(s) in %s", len(pdf_paths), NEW_PDF_FOLDER)
        run_pdfs(pdf_paths)
