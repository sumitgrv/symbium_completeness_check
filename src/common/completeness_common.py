"""
Shared config, logger, and helpers for training data preparation, fine-tuning, and prediction.
"""
import base64
import json
import logging
import os
import time

from openai import OpenAI
from pdf2image import convert_from_path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


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

DATASET_JSONL = "vision_train.jsonl"
LABELS_JSON = "labels.json"
DATASET_VALIDATION_JSONL = "vision_validation.jsonl"
LABELS_VAL_JSON = "labels_val.json"
PDF_FOLDER = "pdfs/"
IMAGE_FOLDER = "images/"
NEW_PDF_FOLDER = "new_pdfs/"
OUTPUT_FOLDER = "output/"
OUTPUT_FOLDER_FINETUNED = "output/fine_tuned_llm/"
OUTPUT_FOLDER_STANDARD = "output/standard_llm/"
FINE_TUNE_BASE_MODEL = os.getenv("FINE_TUNE_BASE_MODEL", "gpt-4o-2024-08-06")
FINE_TUNED_MODEL_FILE = "fine_tuned_model.txt"

# Default pricing table in USD per 1M tokens. Values can be overridden via
# OPENAI_MODEL_PRICING_JSON env var.
_DEFAULT_MODEL_PRICING_USD_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def _extract_usage_tokens(usage):
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or (prompt + completion))
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion) or (prompt + completion))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _extract_pricing_model_key(model_id: str) -> str:
    if not model_id:
        return ""
    if model_id.startswith("ft:"):
        parts = [p for p in model_id.split(":") if p]
        for part in parts:
            if part.startswith("gpt-"):
                return part
    return model_id


def _get_model_pricing_table():
    table = dict(_DEFAULT_MODEL_PRICING_USD_PER_1M)
    raw = (os.getenv("OPENAI_MODEL_PRICING_JSON") or "").strip()
    if not raw:
        return table
    try:
        user_table = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid OPENAI_MODEL_PRICING_JSON. Falling back to default pricing table.")
        return table
    if not isinstance(user_table, dict):
        logger.warning("OPENAI_MODEL_PRICING_JSON must be a JSON object. Falling back to default pricing table.")
        return table
    for model_name, prices in user_table.items():
        if not isinstance(prices, dict):
            continue
        try:
            table[str(model_name)] = {
                "input": float(prices["input"]),
                "output": float(prices["output"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return table


def _resolve_model_pricing(model_id: str):
    table = _get_model_pricing_table()
    model_key = _extract_pricing_model_key(model_id)
    if model_key in table:
        return model_key, table[model_key]
    # Fallback: prefix match (ex: gpt-4o-2024-xx -> gpt-4o)
    for key in sorted(table.keys(), key=len, reverse=True):
        if model_key.startswith(key):
            return key, table[key]
    return model_key, None


def build_openai_request_metrics(model_id: str, usage, duration_ms: float, request_name: str = None):
    tokens = _extract_usage_tokens(usage)
    pricing_model, pricing = _resolve_model_pricing(model_id)

    input_cost = None
    output_cost = None
    total_cost = None
    if pricing:
        input_cost = (tokens["prompt_tokens"] / 1_000_000.0) * pricing["input"]
        output_cost = (tokens["completion_tokens"] / 1_000_000.0) * pricing["output"]
        total_cost = input_cost + output_cost

    metrics = {
        "model_id": model_id,
        "pricing_model_key": pricing_model or None,
        "duration_ms": round(float(duration_ms), 2),
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "input_cost_per_1m_usd": pricing["input"] if pricing else None,
        "output_cost_per_1m_usd": pricing["output"] if pricing else None,
        "estimated_input_cost_usd": round(input_cost, 8) if input_cost is not None else None,
        "estimated_output_cost_usd": round(output_cost, 8) if output_cost is not None else None,
        "estimated_total_cost_usd": round(total_cost, 8) if total_cost is not None else None,
        "cost_estimation_available": bool(pricing),
    }
    if request_name:
        metrics["request_name"] = request_name
    return metrics


SITE_PLAN_SYSTEM_PROMPT = """You are a site plan assistant. Every answer MUST follow the North arrow and PE Stamp definitions below—do not use looser rules.

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

## Rules
- Only structured official stamps count for **stamp**. Ignore standalone handwriting/signatures as stamps.
- Do not treat logos or decorative graphics as stamps or north arrows.
- Respond with JSON only: {"stamp": true/false, "north_arrow": true/false}."""

PREDICTION_USER_INSTRUCTION = (
    "Analyze this plan sheet image. "
    "For north_arrow: apply ONLY the North Direction Symbol rules (reject (N)/(E) phase labels, flow arrows, diagram arrows). "
    "For stamp: apply ONLY the PE Stamp and City/AHJ stamp definitions. "
    "Reply with JSON only: {\"stamp\": true/false, \"north_arrow\": true/false}."
)


def load_fine_tuned_model():
    if os.path.isfile(FINE_TUNED_MODEL_FILE):
        with open(FINE_TUNED_MODEL_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    return None


def save_fine_tuned_model(model_name):
    with open(FINE_TUNED_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(model_name)
    logger.info("Saved fine-tuned model name to %s: %s", FINE_TUNED_MODEL_FILE, model_name)


def pdf_to_images(pdf_path, output_folder=None, dpi=200):
    output_folder = output_folder or IMAGE_FOLDER
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_dir = os.path.join(output_folder, pdf_basename)
    os.makedirs(pdf_dir, exist_ok=True)
    logger.info("Converting PDF to images: %s (dpi=%s) -> %s", pdf_path, dpi, pdf_dir)
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, page in enumerate(pages):
        page_num = i + 1
        image_file = os.path.join(pdf_dir, f"{pdf_basename}_page_{page_num}.png")
        page.save(image_file)
        image_paths.append(image_file)
    logger.debug("Created %d image(s) from %s", len(image_paths), pdf_path)
    return image_paths


def image_path_to_data_url(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def parse_image_path(img_path):
    basename = os.path.basename(img_path)
    name_no_ext = basename[:-4] if basename.lower().endswith(".png") else basename
    if "_page_" not in name_no_ext:
        return None, 1
    pdf_name, page_part = name_no_ext.rsplit("_page_", 1)
    try:
        page_num = int(page_part)
        return pdf_name, page_num
    except ValueError:
        return pdf_name, 1


def create_jsonl_entry(image_path, stamp, north_arrow):
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
    model_id = model_id or load_fine_tuned_model()
    if not model_id:
        raise ValueError("No fine-tuned model id; run fine_tune.py first or set " + FINE_TUNED_MODEL_FILE)
    logger.info("Running prediction on PDF: %s (model=%s)", pdf_path, model_id)
    images = pdf_to_images(pdf_path)
    results = []
    for idx, img_path in enumerate(images):
        logger.debug("Predicting for image %d/%d: %s", idx + 1, len(images), img_path)
        image_url = image_path_to_data_url(img_path)
        start_ts = time.perf_counter()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SITE_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": PREDICTION_USER_INSTRUCTION},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
            ],
        )
        duration_ms = (time.perf_counter() - start_ts) * 1000.0
        request_metrics = build_openai_request_metrics(
            model_id=model_id,
            usage=getattr(response, "usage", None),
            duration_ms=duration_ms,
            request_name="fine_tuned_prediction",
        )
        output = response.choices[0].message.content
        results.append({"image": img_path, "prediction": output, "request_metrics": request_metrics})
    logger.info("Prediction complete for %s: %d page(s) processed", pdf_path, len(results))
    return results


def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def predict_single_image(client, image_data, model_id=None, mime=None):
    model_id = model_id or load_fine_tuned_model()
    if not model_id:
        raise ValueError("No fine-tuned model id; set " + FINE_TUNED_MODEL_FILE + " or FINE_TUNED_MODEL_ID")
    if isinstance(image_data, bytes):
        mime = mime or "image/png"
        image_url = image_bytes_to_data_url(image_data, mime)
    else:
        image_url = image_path_to_data_url(image_data)
    start_ts = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SITE_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": PREDICTION_USER_INSTRUCTION},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
        ],
    )
    duration_ms = (time.perf_counter() - start_ts) * 1000.0
    request_metrics = build_openai_request_metrics(
        model_id=model_id,
        usage=getattr(response, "usage", None),
        duration_ms=duration_ms,
        request_name="fine_tuned_prediction_single_image",
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
        "request_metrics": request_metrics,
    }
