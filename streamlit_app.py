"""
Streamlit demo: upload a plan sheet image and run completeness check.
Supports Prompt-Based (Standard LLM) or Fine-Tuned Model via sidebar configuration.
Run: streamlit run streamlit_app.py
"""
import os
import streamlit as st
from PIL import Image

from completeness_common import (
    get_client,
    load_fine_tuned_model,
    predict_single_image,
    logger,
)
from completeness_check_using_standard_llm import predict as predict_standard_llm

st.set_page_config(page_title="Site Plan Completeness", page_icon="📋", layout="centered")

# ---------------------------------------------------------------------------
# Sidebar: Model Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Configuration")

    approach = st.radio(
        "Inference approach",
        ["Prompt-Based (Standard LLM)", "Fine-Tuned Model"],
        index=0,
        help="Choose whether to use a standard vision model with prompts or a fine-tuned model.",
    )

    if approach == "Prompt-Based (Standard LLM)":
        base_models = [
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        # Allow env override for extra models: OPENAI_BASE_MODELS=gpt-4o,gpt-4o-mini,gpt-4.1
        env_models = os.getenv("OPENAI_BASE_MODELS", "").strip()
        if env_models:
            base_models = [m.strip() for m in env_models.split(",") if m.strip()]
        selected_base = st.selectbox(
            "Select Base Model",
            options=base_models,
            index=1 if "gpt-4o" in base_models else 0,
            help="OpenAI vision model used for prompt-based stamp and north-arrow detection.",
        )
        model_id = selected_base
    else:
        # Fine-Tuned Model: optional dropdown + text input
        client = None
        try:
            client = get_client()
            ft_models = []
            for m in client.models.list().data:
                if getattr(m, "id", "").startswith("ft:"):
                    ft_models.append(m.id)
            ft_models = sorted(ft_models)
        except Exception:
            ft_models = []

        if ft_models:
            dropdown_option = st.selectbox(
                "Available fine-tuned models",
                options=["— Paste manually —"] + ft_models,
                index=0,
                help="Select a fine-tuned model or choose to paste an ID below.",
            )
            if dropdown_option and dropdown_option != "— Paste manually —":
                model_id = dropdown_option
            else:
                default_ft_id = load_fine_tuned_model() or (os.getenv("FINE_TUNED_MODEL_ID") or os.getenv("MODEL_ID") or "").strip() or ""
                manual_ft_id = st.text_input(
                    "Enter Fine-Tuned Model ID",
                    value=default_ft_id,
                    placeholder="e.g. ft:gpt-4o-2024-08-06:org:...",
                    help="Paste your fine-tuned model ID if not listed above.",
                )
                model_id = (manual_ft_id or "").strip()
        else:
            default_ft_id = load_fine_tuned_model() or (os.getenv("FINE_TUNED_MODEL_ID") or os.getenv("MODEL_ID") or "").strip() or ""
            manual_ft_id = st.text_input(
                "Enter Fine-Tuned Model ID",
                value=default_ft_id,
                placeholder="e.g. ft:gpt-4o-2024-08-06:org:...",
                help="Paste your fine-tuned model ID (dropdown is empty if listing failed).",
            )
            model_id = (manual_ft_id or "").strip()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Permit Plan Completeness CheckDemo")
caption = "Upload a permit plan sheet image to detect **PE stamp** and **North arrow**."
if approach == "Prompt-Based (Standard LLM)":
    st.caption(caption + f" _(Prompt-based · {model_id})_")
else:
    st.caption(caption + " _(Fine-tuned model)_")

uploaded = st.file_uploader("Choose an image (PNG / JPEG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Upload an image to run prediction.")
    st.stop()

img = Image.open(uploaded)
img_bytes = uploaded.getvalue()
mime = uploaded.type or "image/png"
if mime not in ("image/png", "image/jpeg", "image/jpg"):
    mime = "image/png"

if approach == "Fine-Tuned Model" and not model_id:
    st.error("No fine-tuned model configured. Enter a Fine-Tuned Model ID in the sidebar or set `fine_tuned_model.txt` / `FINE_TUNED_MODEL_ID` in .env.")
    st.stop()

if st.button("Run prediction"):
    with st.spinner("Running prediction…"):
        try:
            client = get_client()
            if approach == "Prompt-Based (Standard LLM)":
                out = predict_standard_llm(client, img_bytes, model=model_id)
                # Normalize to same shape as fine-tuned output for display
                stamp_yes = (out.get("stamp_result") or {}).get("checkStampPresence", "").strip().lower() == "yes"
                north_yes = (out.get("north_arrow_result") or {}).get("NorthDirectionSymbol", "").strip().lower() == "detected"
                display = {
                    "stamp": stamp_yes,
                    "north_arrow": north_yes,
                    "prediction_parsed": {
                        "stamp": stamp_yes,
                        "north_arrow": north_yes,
                        "stamp_result": out.get("stamp_result"),
                        "north_arrow_result": out.get("north_arrow_result"),
                    },
                    "prediction_raw": None,
                }
            else:
                out = predict_single_image(client, img_bytes, model_id=model_id, mime=mime)
                display = {
                    "stamp": out.get("stamp"),
                    "north_arrow": out.get("north_arrow"),
                    "prediction_parsed": out.get("prediction_parsed"),
                    "prediction_raw": out.get("prediction_raw"),
                }
        except Exception as e:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {e}")
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_container_width=True, caption="Uploaded image")
    with col2:
        st.subheader("Result")
        stamp = display.get("stamp")
        north_arrow = display.get("north_arrow")
        st.metric("PE stamp", "Yes" if stamp else "No")
        st.metric("North arrow", "Yes" if north_arrow else "No")
    with st.expander("Raw model output"):
        st.json(display.get("prediction_parsed") or {"raw": display.get("prediction_raw")})
        raw = display.get("prediction_raw")
        if raw:
            st.caption("Raw response")
            st.code(raw)
        elif approach == "Prompt-Based (Standard LLM)" and out:
            st.caption("Full response")
            st.json({"stamp_result": out.get("stamp_result"), "north_arrow_result": out.get("north_arrow_result")})
