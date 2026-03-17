"""
Streamlit demo: upload a plan sheet image or PDF and run completeness check.
Supports Prompt-Based (Standard LLM), Fine-Tuned Model, or Both via sidebar configuration.
Run: streamlit run demo/streamlit_app.py
"""
import io
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure project root is importable when running `streamlit run demo/streamlit_app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.completeness_common import (
    FINE_TUNE_BASE_MODEL,
    get_client,
    load_fine_tuned_model,
    logger,
    pdf_to_images,
    predict_single_image,
)
from src.standard_llm.completeness_check import predict as predict_standard_llm

st.set_page_config(page_title="Completeness Check", page_icon="📋", layout="centered")

with st.sidebar:
    st.header("Model Configuration")
    approach = st.radio(
        "Inference approach",
        ["Prompt-Based (Standard LLM)", "Fine-Tuned Model", "Both"],
        index=0,
        help="Use standard LLM, fine-tuned model, or run both and compare results side by side.",
    )

    use_standard = approach in ("Prompt-Based (Standard LLM)", "Both")
    use_finetuned = approach in ("Fine-Tuned Model", "Both")

    model_id_standard = None
    model_id_ft = None

    if use_standard:
        base_models = ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]
        env_models = os.getenv("OPENAI_BASE_MODELS", "").strip()
        if env_models:
            base_models = [m.strip() for m in env_models.split(",") if m.strip()]
        selected_base = st.selectbox(
            "Select Base Model (Standard LLM)" if use_finetuned else "Select Base Model",
            options=base_models,
            index=1 if "gpt-4o" in base_models else 0,
            help="OpenAI vision model used for prompt-based stamp and north-arrow detection.",
        )
        model_id_standard = selected_base

    if use_finetuned:
        ft_base_model = (os.getenv("FINE_TUNE_BASE_MODEL") or FINE_TUNE_BASE_MODEL or "").strip()
        st.caption(f"Base model: {ft_base_model or 'Not set'}")

        try:
            client = get_client()
            ft_models = sorted([m.id for m in client.models.list().data if getattr(m, "id", "").startswith("ft:")])
        except Exception:
            ft_models = []

        if ft_models:
            dropdown_option = st.selectbox(
                "Available fine-tuned models" if use_standard else "Available fine-tuned models",
                options=["— Paste manually —"] + ft_models,
                index=0,
                help="Select a fine-tuned model or choose to paste an ID below.",
            )
            if dropdown_option and dropdown_option != "— Paste manually —":
                model_id_ft = dropdown_option
            else:
                default_ft_id = load_fine_tuned_model() or (os.getenv("FINE_TUNED_MODEL_ID") or os.getenv("MODEL_ID") or "").strip() or ""
                model_id_ft = st.text_input(
                    "Enter Fine-Tuned Model ID",
                    value=default_ft_id,
                    placeholder="e.g. ft:gpt-4o-2024-08-06:org:...",
                    help="Paste your fine-tuned model ID if not listed above.",
                ).strip()
        else:
            default_ft_id = load_fine_tuned_model() or (os.getenv("FINE_TUNED_MODEL_ID") or os.getenv("MODEL_ID") or "").strip() or ""
            model_id_ft = st.text_input(
                "Enter Fine-Tuned Model ID",
                value=default_ft_id,
                placeholder="e.g. ft:gpt-4o-2024-08-06:org:...",
                help="Paste your fine-tuned model ID (dropdown is empty if listing failed).",
            ).strip()
        st.caption(f"Model ID: {model_id_ft or 'Not set'}")

    # Single-mode compatibility
    if approach == "Prompt-Based (Standard LLM)":
        model_id = model_id_standard
    elif approach == "Fine-Tuned Model":
        model_id = model_id_ft
    else:
        model_id = None

st.title("Permit Plan Completeness Check Demo")
caption = "Upload a plan sheet **image** (PNG/JPEG) or **PDF** to detect **PE stamp** and **North arrow**."
if approach == "Prompt-Based (Standard LLM)":
    st.caption(caption + f" _(Prompt-based · {model_id})_")
elif approach == "Fine-Tuned Model":
    st.caption(caption + " _(Fine-tuned)_")
else:
    st.caption(caption + " _(Standard + Fine-tuned)_")

uploaded = st.file_uploader("Choose an image or PDF", type=["png", "jpg", "jpeg", "pdf"])
if not uploaded:
    st.info("Upload an image (PNG/JPEG) or a PDF to run prediction.")
    st.stop()

is_pdf = (uploaded.name or "").lower().endswith(".pdf")

# Cache pages_data in session_state so PDF conversion doesn't re-run on every interaction.
# Invalidate when the uploaded file changes (keyed by name + size).
_upload_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("_upload_key") != _upload_key:
    pages_data = []
    if is_pdf:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "uploaded.pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded.getvalue())
            image_paths = pdf_to_images(pdf_path, output_folder=tmpdir, dpi=200)
            for i, img_path in enumerate(image_paths):
                page_num = i + 1
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                pil_img = Image.open(io.BytesIO(img_bytes)).copy()
                pil_img.load()
                pages_data.append((page_num, img_bytes, pil_img, "image/png"))
    else:
        img_bytes = uploaded.getvalue()
        img = Image.open(uploaded)
        mime_val = uploaded.type or "image/png"
        if mime_val not in ("image/png", "image/jpeg", "image/jpg"):
            mime_val = "image/png"
        pages_data.append((1, img_bytes, img, mime_val))
    st.session_state["_upload_key"] = _upload_key
    st.session_state["_pages_data"] = pages_data
    st.session_state.pop("_prediction_results", None)
else:
    pages_data = st.session_state["_pages_data"]

if approach == "Fine-Tuned Model" and not model_id_ft:
    st.error("No fine-tuned model configured. Enter a Fine-Tuned Model ID in the sidebar or set `fine_tuned_model.txt` / `FINE_TUNED_MODEL_ID` in .env.")
    st.stop()
if approach == "Both" and (not model_id_standard or not model_id_ft):
    st.error("Configure both Standard LLM (base model) and Fine-Tuned Model ID in the sidebar for 'Both' mode.")
    st.stop()

def _run_batch(fn_per_page, pages_data, max_workers=8):
    """Run fn_per_page(page_num, img_bytes, pil_img, mime) for all pages in parallel.
    Returns dict {page_num: result} and wall_elapsed_ms.
    """
    t0 = time.perf_counter()
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(pages_data), max_workers)) as pool:
        futures = {}
        for page_num, img_bytes, pil_img, mime in pages_data:
            future = pool.submit(fn_per_page, page_num, img_bytes, pil_img, mime)
            futures[future] = page_num
        for future in as_completed(futures):
            page_num = futures[future]
            results[page_num] = future.result()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return results, wall_ms


if st.button("Run prediction"):
    with st.spinner("Running prediction..."):
        try:
            client = get_client()
            page_results = []
            total_cost_usd = None
            std_wall_total_ms = 0.0
            ft_wall_total_ms = 0.0
            wall_elapsed_ms = 0.0

            std_results_by_page = {}
            ft_results_by_page = {}

            if approach in ("Prompt-Based (Standard LLM)", "Both"):
                def _run_std(page_num, img_bytes, pil_img, mime):
                    return predict_standard_llm(client, img_bytes, model=model_id_standard)
                std_results_by_page, std_wall_total_ms = _run_batch(_run_std, pages_data)

            if approach in ("Fine-Tuned Model", "Both"):
                def _run_ft(page_num, img_bytes, pil_img, mime):
                    return predict_single_image(client, img_bytes, model_id=model_id_ft, mime=mime)
                ft_results_by_page, ft_wall_total_ms = _run_batch(_run_ft, pages_data)

            if approach == "Both":
                wall_elapsed_ms = std_wall_total_ms + ft_wall_total_ms
            else:
                wall_elapsed_ms = std_wall_total_ms + ft_wall_total_ms

            all_page_nums = sorted({p for p, *_ in pages_data})
            for page_num, img_bytes, pil_img, mime in pages_data:
                out_standard = std_results_by_page.get(page_num)
                out_finetuned = ft_results_by_page.get(page_num)
                page_cost = None

                if out_standard is not None:
                    page_total = (out_standard.get("request_metrics") or {}).get("page_total", {})
                    c = page_total.get("estimated_total_cost_usd")
                    if c is not None:
                        page_cost = (page_cost or 0.0) + float(c)

                if out_finetuned is not None:
                    rm = out_finetuned.get("request_metrics") or {}
                    c = rm.get("estimated_total_cost_usd")
                    if c is not None:
                        page_cost = (page_cost or 0.0) + float(c)

                if page_cost is not None:
                    total_cost_usd = (total_cost_usd or 0.0) + page_cost

                if approach == "Prompt-Based (Standard LLM)":
                    out = out_standard
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
                elif approach == "Fine-Tuned Model":
                    out = out_finetuned
                    display = {
                        "stamp": out.get("stamp"),
                        "north_arrow": out.get("north_arrow"),
                        "prediction_parsed": out.get("prediction_parsed"),
                        "prediction_raw": out.get("prediction_raw"),
                    }
                else:
                    display = None

                page_results.append({
                    "page_num": page_num,
                    "pil_img": pil_img,
                    "display": display,
                    "out_standard": out_standard,
                    "out_finetuned": out_finetuned,
                })

            st.session_state["_prediction_results"] = {
                "page_results": page_results,
                "total_cost_usd": total_cost_usd,
                "wall_elapsed_ms": wall_elapsed_ms,
                "std_wall_total_ms": std_wall_total_ms,
                "ft_wall_total_ms": ft_wall_total_ms,
                "approach": approach,
            }
        except Exception as e:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# Render results from session state (persists across reruns / widget changes)
# ---------------------------------------------------------------------------
_res = st.session_state.get("_prediction_results")
if _res and _res.get("approach") == approach:
    page_results = _res["page_results"]
    total_cost_usd = _res["total_cost_usd"]
    wall_elapsed_ms = _res["wall_elapsed_ms"]
    std_wall_total_ms = _res.get("std_wall_total_ms", 0.0)
    ft_wall_total_ms = _res.get("ft_wall_total_ms", 0.0)

    num_pages = len(page_results)
    multi_page = num_pages > 1

    if multi_page:
        st.caption(f"Processed **{num_pages}** pages.")

    # Compute per-model totals for PDF-level averages
    def _aggregate_model_costs(page_results, num_pages):
        std_total_cost = None
        ft_total_cost = None
        for pr in page_results:
            if pr.get("out_standard"):
                pt = (pr["out_standard"].get("request_metrics") or {}).get("page_total", {})
                c = pt.get("estimated_total_cost_usd")
                if c is not None:
                    std_total_cost = (std_total_cost or 0.0) + float(c)
            if pr.get("out_finetuned"):
                rm = pr["out_finetuned"].get("request_metrics") or {}
                c = rm.get("estimated_total_cost_usd")
                if c is not None:
                    ft_total_cost = (ft_total_cost or 0.0) + float(c)
        std_avg_cost = (std_total_cost / num_pages) if std_total_cost is not None else None
        ft_avg_cost = (ft_total_cost / num_pages) if ft_total_cost is not None else None
        return std_total_cost, ft_total_cost, std_avg_cost, ft_avg_cost

    if approach == "Both" and multi_page:
        std_total_cost, ft_total_cost, std_avg_cost, ft_avg_cost = _aggregate_model_costs(page_results, num_pages)
        std_avg_time = std_wall_total_ms / num_pages
        ft_avg_time = ft_wall_total_ms / num_pages

        st.subheader("PDF prediction summary")
        col_std, col_ft = st.columns(2)
        with col_std:
            st.markdown("**Standard LLM**")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total cost", f"${std_total_cost:.6f}" if std_total_cost is not None else "—")
                st.metric("Total time", f"{std_wall_total_ms / 1000:.2f} s")
            with m2:
                st.metric("Avg cost / page", f"${std_avg_cost:.6f}" if std_avg_cost is not None else "—")
                st.metric("Avg time / page", f"{std_avg_time / 1000:.2f} s")
        with col_ft:
            st.markdown("**Fine-tuned LLM**")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total cost", f"${ft_total_cost:.6f}" if ft_total_cost is not None else "—")
                st.metric("Total time", f"{ft_wall_total_ms / 1000:.2f} s")
            with m2:
                st.metric("Avg cost / page", f"${ft_avg_cost:.6f}" if ft_avg_cost is not None else "—")
                st.metric("Avg time / page", f"{ft_avg_time / 1000:.2f} s")
        st.divider()

    elif approach != "Both" and multi_page:
        if approach == "Prompt-Based (Standard LLM)":
            std_total_cost, _, std_avg_cost, _ = _aggregate_model_costs(page_results, num_pages)
            model_total_cost = std_total_cost
            model_avg_cost = std_avg_cost
        else:
            _, ft_total_cost, _, ft_avg_cost = _aggregate_model_costs(page_results, num_pages)
            model_total_cost = ft_total_cost
            model_avg_cost = ft_avg_cost
        avg_time = wall_elapsed_ms / num_pages

        st.subheader("PDF prediction summary")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total cost", f"${model_total_cost:.6f}" if model_total_cost is not None else "—")
        with m2:
            st.metric("Total time", f"{wall_elapsed_ms / 1000:.2f} s")
        with m3:
            st.metric("Avg cost / page", f"${model_avg_cost:.6f}" if model_avg_cost is not None else "—")
        with m4:
            st.metric("Avg time / page", f"{avg_time / 1000:.2f} s")
        st.divider()

    elif approach != "Both":
        st.subheader("Request metrics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            if total_cost_usd is not None:
                st.metric("Total OpenAI cost", f"${total_cost_usd:.6f}")
            else:
                st.metric("Total OpenAI cost", "— (pricing unknown)")
        with metric_col2:
            st.metric("Prediction time", f"{wall_elapsed_ms / 1000:.2f} s")
        st.divider()

    def _show_page_content(pr, _approach, _multi_page, _std_wall_ms, _ft_wall_ms):
        """Render image + results for one page."""
        display = pr["display"]
        out_standard = pr["out_standard"]
        out_finetuned = pr["out_finetuned"]
        pil_img = pr["pil_img"]

        if _approach == "Both":
            col_img, col_std, col_ft = st.columns(3)
            with col_img:
                st.image(pil_img, use_container_width=True, caption=f"Page {pr['page_num']}")
            with col_std:
                st.subheader("Standard LLM")
                out = out_standard
                stamp_yes = (out.get("stamp_result") or {}).get("checkStampPresence", "").strip().lower() == "yes"
                north_yes = (out.get("north_arrow_result") or {}).get("NorthDirectionSymbol", "").strip().lower() == "detected"
                st.metric("PE stamp", "Yes" if stamp_yes else "No")
                st.metric("North arrow", "Yes" if north_yes else "No")
                page_total = (out.get("request_metrics") or {}).get("page_total", {})
                std_cost = page_total.get("estimated_total_cost_usd")
                st.metric("Prediction time", f"{_std_wall_ms / 1000:.2f} s")
                st.metric("OpenAI cost", f"${std_cost:.6f}" if std_cost is not None else "—")
            with col_ft:
                st.subheader("Fine-tuned LLM")
                out = out_finetuned
                st.metric("PE stamp", "Yes" if out.get("stamp") else "No")
                st.metric("North arrow", "Yes" if out.get("north_arrow") else "No")
                rm = out.get("request_metrics") or {}
                ft_cost = rm.get("estimated_total_cost_usd")
                st.metric("Prediction time", f"{_ft_wall_ms / 1000:.2f} s")
                st.metric("OpenAI cost", f"${ft_cost:.6f}" if ft_cost is not None else "—")
            with st.expander("Raw outputs"):
                st.write("**Standard LLM**")
                st.json({"stamp_result": out_standard.get("stamp_result"), "north_arrow_result": out_standard.get("north_arrow_result")})
                st.write("**Fine-tuned LLM**")
                st.json(out_finetuned.get("prediction_parsed") or {"raw": out_finetuned.get("prediction_raw")})
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, use_container_width=True, caption=f"Page {pr['page_num']}" if _multi_page else "Uploaded image")
            with col2:
                st.subheader("Result")
                st.metric("PE stamp", "Yes" if display.get("stamp") else "No")
                st.metric("North arrow", "Yes" if display.get("north_arrow") else "No")
            with st.expander("Raw model output"):
                st.json(display.get("prediction_parsed") or {"raw": display.get("prediction_raw")})
                raw = display.get("prediction_raw")
                if raw:
                    st.caption("Raw response")
                    st.code(raw)
                elif _approach == "Prompt-Based (Standard LLM)" and out_standard:
                    st.caption("Full response")
                    st.json({"stamp_result": out_standard.get("stamp_result"), "north_arrow_result": out_standard.get("north_arrow_result")})

    if multi_page:
        st.subheader("Summary by page")
        rows = []
        for pr in page_results:
            if approach == "Both":
                out_std = pr["out_standard"]
                out_ft = pr["out_finetuned"]
                std_stamp = (out_std.get("stamp_result") or {}).get("checkStampPresence", "").strip().lower() == "yes"
                std_north = (out_std.get("north_arrow_result") or {}).get("NorthDirectionSymbol", "").strip().lower() == "detected"
                rows.append({
                    "Page": pr["page_num"],
                    "Std: PE stamp": "Yes" if std_stamp else "No",
                    "Std: North arrow": "Yes" if std_north else "No",
                    "FT: PE stamp": "Yes" if out_ft.get("stamp") else "No",
                    "FT: North arrow": "Yes" if out_ft.get("north_arrow") else "No",
                })
            else:
                d = pr["display"]
                rows.append({
                    "Page": pr["page_num"],
                    "PE stamp": "Yes" if d.get("stamp") else "No",
                    "North arrow": "Yes" if d.get("north_arrow") else "No",
                })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.divider()
        selected_page_num = st.selectbox(
            "View details for page",
            options=[pr["page_num"] for pr in page_results],
            format_func=lambda x: f"Page {x}",
        )
        selected = next(pr for pr in page_results if pr["page_num"] == selected_page_num)
        _show_page_content(selected, approach, multi_page, std_wall_total_ms, ft_wall_total_ms)
    else:
        _show_page_content(page_results[0], approach, multi_page, std_wall_total_ms, ft_wall_total_ms)
