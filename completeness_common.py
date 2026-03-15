"""
Shared config, logger, and helpers for training data preparation, fine-tuning, and prediction.
"""
import os
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import logging
import base64
from pdf2image import convert_from_path
from openai import OpenAI

# ----------------------------
# Logger
# ----------------------------
def setup_logger(name="completeness_check", level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

logger = setup_logger("completeness_check", log_file="completeness_check.log")

# ----------------------------
# Config
# ----------------------------
DATASET_JSONL = "vision_train.jsonl"
LABELS_JSON = "labels.json"
DATASET_VALIDATION_JSONL = "vision_validation.jsonl"
LABELS_VAL_JSON = "labels_val.json"
PDF_FOLDER = "pdfs/"
IMAGE_FOLDER = "images/"  # all PDF→image output (train + validation) under images/<pdf_name>/
NEW_PDF_FOLDER = "new_pdfs/"
OUTPUT_FOLDER = "output/"  # legacy; use OUTPUT_FOLDER_FINETUNED or OUTPUT_FOLDER_STANDARD
OUTPUT_FOLDER_FINETUNED = "output/fine_tuned_llm/"
OUTPUT_FOLDER_STANDARD = "output/standard_llm/"
FINE_TUNE_BASE_MODEL = os.getenv("FINE_TUNE_BASE_MODEL", "gpt-4o-2024-08-06")
FINE_TUNED_MODEL_FILE = "fine_tuned_model.txt"


def get_client():
    """OpenAI client (requires OPENAI_API_KEY in environment)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


# ----------------------------
# System prompt
# ----------------------------
SITE_PLAN_SYSTEM_PROMPT = """You are a site plan assistant. Every answer MUST follow the North arrow and PE Stamp (and City stamp) definitions below—do not use looser rules.

Output JSON only, no other text: {"stamp": true/false, "north_arrow": true/false}
- **north_arrow**: true = "Detected", false = "Not Detected" for a North Direction Symbol (per definitions below).
- **stamp**: true ONLY if a PE Stamp OR a qualifying City/AHJ stamp is present per the sections "PE Stamp" and "City / AHJ stamp".

---

### Definition of a North Direction Symbol
- **Definitions**: A **North Direction Symbol** is a geographical directional symbol used to indicate geographic orientation in technical diagrams, such as floor plans or site plans.
- **Expected values** (in JSON): **north_arrow** true if present ("Detected"), else false ("Not Detected").
- **Visual indicators**:
  - A **North Direction Symbol** is often available with label "North" pointing in one direction.
  - May include a full compass rose (N, NE, E, etc.) or a single geographical direction symbol with just the letter "N".
  - Is often labeled with the word **"NORTH"** or simply the letter **"N"**.
  - May appear **near the scale indicator**, often in the **corner or edge** of a drawing.

---

### Definition of Scale Indicator
- **Definition**: A **scale** is a reference graphic (such as a line or bar) labeled with units like "Scale: 1" = 20'-0"", "1:200", "N.T.S."
- **Visual indicators**:
  - Textual formats: "Scale: 1" = 20'-0"", "1:200", "N.T.S." (Not to Scale)
  - Or graphical bars labeled with distance (e.g., "10m" or "50 ft")
  - Most often appears near the **North Direction Symbol** or in **corners/title blocks**
- **Assumption**: If a North Direction Symbol is present, a scale is **likely to co-occur nearby** or on the same page. (You still output only north_arrow and stamp in JSON.)

---

### Before concluding north_arrow detection, verify:
- Does the arrow explicitly show geographic direction (not flow or diagram arrows)?
- Is the arrow paired with "NORTH" or clear compass/north convention?
- Is it in a logical architectural position (e.g., corner, title block)?

---

### North Direction Symbol — important instructions
- **Only detect the North Direction Symbol** (and use scale only as context for where to look).
- Look for **standard architectural or civil drawing conventions** — geographical compass or north direction symbol.
- Do **not** treat direction-like symbols embedded inside the site diagram or near object labels (e.g., house, driveway) as valid North arrows.
- Do **not** interpret "(N)" in equipment labels (e.g., "(N) Inverter", "(N) Panel", "(E) House", "(N) PV System") as a North direction symbol. These are **installation phase indicators** (e.g., New, Existing), not geographic directions.
- Do **not** consider direction arrows or annotations **pointing into the diagram** as a North Direction Symbol.
- Do **not** confuse decorative arrows, flow arrows, or compass-like logos with a true North Direction Symbol.
- Absolutely **do not treat any label in parentheses**, such as "(N)", "(E)", "(R)", or "(P)", as a North direction arrow. These are **not geographic directional symbols** — they indicate project phases:
  - (N) = New
  - (E) = Existing
  - (R) = Relocated
  - (P) = Proposed
- **Never treat "(N)"** in front of equipment names (e.g., "(N) Inverter", "(N) PV System", "(E) Battery") as a North direction marker — even if near arrows or architectural elements.
- A valid **North arrow must be a graphic directional symbol** with an arrow clearly pointing and labeled with "NORTH" or standard north/compass convention — **not embedded in parentheses** as a phase label.

---

## PE Stamp (Professional Engineer)
A **PE stamp** is an official engineer registration seal on the drawing.
- **Required**: Printed **professional title** such as LICENSED ENGINEER, PROFESSIONAL ENGINEER, or LICENSED PROFESSIONAL ENGINEER (if no such title, it is NOT a PE stamp).
- **Often also includes**: State/province (e.g. STATE OF CALIFORNIA), license number, expiration (e.g. EXP 03/25), discipline (CIVIL, STRUCTURAL, MECHANICAL).
- **Shape**: Circular, rectangular, or other; **structured printed text**, not handwriting alone.
- PE stamp may be partly covered by a signature—judge from the **printed** seal text.

---

## City / AHJ stamp (counts toward "stamp")
Official approval by city, county, district, or AHJ.
- **Required together**: (1) Approval wording in CAPS (APPROVED, APPROVAL, RECEIVED, ACCEPTED, ISSUED, CONDITIONAL APPROVAL); (2) Jurisdiction reference (e.g. CITY OF …, COUNTY, BUILDING DEPARTMENT); (3) Structured printed text.
- **Not** city stamps: company logos, contractor stamps, plain PE seals without approval+jurisdiction, title blocks without approval keywords.

---

## Rules
- Only structured official stamps count for **stamp**. Ignore standalone handwriting/signatures as stamps.
- Do not treat logos or decorative graphics as stamps or north arrows.
- Respond with JSON only: {"stamp": true/false, "north_arrow": true/false}."""

# User text used only at prediction time so the model explicitly applies north_arrow + PE stamp rules
PREDICTION_USER_INSTRUCTION = (
    "Analyze this plan sheet image. "
    "For north_arrow: apply ONLY the North Direction Symbol rules (reject (N)/(E) phase labels, flow arrows, diagram arrows). "
    "For stamp: apply ONLY the PE Stamp and City/AHJ stamp definitions. "
    "Reply with JSON only: {\"stamp\": true/false, \"north_arrow\": true/false}."
)

# ----------------------------
# Helpers
# ----------------------------
def load_fine_tuned_model():
    """Load fine-tuned model name from disk."""
    if os.path.isfile(FINE_TUNED_MODEL_FILE):
        with open(FINE_TUNED_MODEL_FILE, "r") as f:
            return f.read().strip() or None
    return None


def save_fine_tuned_model(model_name):
    """Persist fine-tuned model name to disk."""
    with open(FINE_TUNED_MODEL_FILE, "w") as f:
        f.write(model_name)
    logger.info("Saved fine-tuned model name to %s: %s", FINE_TUNED_MODEL_FILE, model_name)


def pdf_to_images(pdf_path, output_folder=None, dpi=200):
    """Save page images under IMAGE_FOLDER/<pdf name>/<pdf name>_page_N.png. N is 1-based (page 1, 2, 3...)."""
    output_folder = output_folder or IMAGE_FOLDER
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_dir = os.path.join(output_folder, pdf_basename)
    os.makedirs(pdf_dir, exist_ok=True)
    logger.info("Converting PDF to images: %s (dpi=%s) -> %s", pdf_path, dpi, pdf_dir)
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, page in enumerate(pages):
        page_num = i + 1  # 1-based to match labels.json (page 1, 2, 3...)
        image_file = os.path.join(pdf_dir, f"{pdf_basename}_page_{page_num}.png")
        page.save(image_file)
        image_paths.append(image_file)
    logger.debug("Created %d image(s) from %s", len(image_paths), pdf_path)
    return image_paths


def image_path_to_data_url(image_path):
    """Encode local image as base64 data URL for OpenAI API."""
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def parse_image_path(img_path):
    """From path like 'images/Some PDF Name/Some PDF Name_page_1.png' return (pdf_name, page_num). Page num is 1-based (matches labels.json)."""
    basename = os.path.basename(img_path)
    name_no_ext = basename[:-4] if basename.lower().endswith(".png") else basename
    if "_page_" not in name_no_ext:
        return None, 1
    pdf_name, page_part = name_no_ext.rsplit("_page_", 1)
    try:
        page_num = int(page_part)  # already 1-based in filename
        return pdf_name, page_num
    except ValueError:
        return pdf_name, 1


def create_jsonl_entry(image_path, stamp, north_arrow):
    """Build one JSONL training entry (uses SITE_PLAN_SYSTEM_PROMPT and base64 image)."""
    image_url = image_path_to_data_url(image_path)
    return {
        "messages": [
            {"role": "system", "content": SITE_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": PREDICTION_USER_INSTRUCTION},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
            {"role": "assistant", "content": json.dumps({"stamp": stamp, "north_arrow": north_arrow})},
        ]
    }


def predict_from_pdf(client, pdf_path, model_id=None):
    """Run detection on each page of a PDF. model_id defaults to saved fine-tuned model."""
    model_id = model_id or load_fine_tuned_model()
    if not model_id:
        raise ValueError("No fine-tuned model id; run 2_fine_tune.py first or set " + FINE_TUNED_MODEL_FILE)
    logger.info("Running prediction on PDF: %s (model=%s)", pdf_path, model_id)
    images = pdf_to_images(pdf_path)
    results = []
    for idx, img_path in enumerate(images):
        logger.debug("Predicting for image %d/%d: %s", idx + 1, len(images), img_path)
        image_url = image_path_to_data_url(img_path)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SITE_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": PREDICTION_USER_INSTRUCTION},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
            ],
        )
        output = response.choices[0].message.content
        results.append({"image": img_path, "prediction": output})
    logger.info("Prediction complete for %s: %d page(s) processed", pdf_path, len(results))
    return results


def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Convert raw image bytes to a data URL for the API."""
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def predict_single_image(client, image_data, model_id=None, mime=None):
    """
    Run detection on a single image. image_data: bytes or path (str).
    mime: optional, e.g. 'image/jpeg' when image_data is bytes (default image/png).
    Returns dict with prediction_raw, stamp, north_arrow (parsed), and prediction_parsed.
    """
    model_id = model_id or load_fine_tuned_model()
    if not model_id:
        raise ValueError("No fine-tuned model id; set " + FINE_TUNED_MODEL_FILE + " or FINE_TUNED_MODEL_ID")
    if isinstance(image_data, bytes):
        mime = mime or "image/png"
        image_url = image_bytes_to_data_url(image_data, mime)
    else:
        image_url = image_path_to_data_url(image_data)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SITE_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": PREDICTION_USER_INSTRUCTION},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
        ],
    )
    output = response.choices[0].message.content or "{}"
    parsed = None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        pass
    return {
        "prediction_raw": output,
        "prediction_parsed": parsed,
        "stamp": parsed.get("stamp") if isinstance(parsed, dict) else None,
        "north_arrow": parsed.get("north_arrow") if isinstance(parsed, dict) else None,
    }
