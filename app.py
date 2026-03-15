"""
API for site plan completeness (stamp / north arrow) using the fine-tuned model.
POST /predict: upload an image file, get JSON with stamp, north_arrow, prediction_raw.
Run: python app.py  (or uvicorn app:app --reload)
"""
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from completeness_common import (
    get_client,
    load_fine_tuned_model,
    predict_single_image,
    logger,
)

app = FastAPI(
    title="Site Plan Completeness API",
    description="Detect PE stamp and North arrow on plan sheet images using fine-tuned GPT-4o vision.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_model_id():
    model_id = load_fine_tuned_model()
    if not model_id:
        model_id = (os.getenv("FINE_TUNED_MODEL_ID") or os.getenv("MODEL_ID") or "").strip()
    if not model_id:
        raise HTTPException(
            status_code=503,
            detail="No fine-tuned model configured. Set fine_tuned_model.txt or FINE_TUNED_MODEL_ID in .env",
        )
    return model_id


@app.get("/")
def root():
    return {"service": "site-plan-completeness", "predict": "POST /predict with image file"}


@app.get("/health")
def health():
    try:
        _get_model_id()
        return {"status": "ok", "model": "configured"}
    except HTTPException:
        return {"status": "degraded", "model": "not configured"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Upload a plan sheet image (PNG/JPEG). Returns stamp, north_arrow (bool), and raw model output."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. image/png, image/jpeg)")
    try:
        body = await image.read()
    except Exception as e:
        logger.exception("Failed to read upload")
        raise HTTPException(status_code=400, detail="Failed to read file")
    if not body:
        raise HTTPException(status_code=400, detail="Empty file")
    mime = image.content_type or "image/png"
    if mime not in ("image/png", "image/jpeg", "image/jpg"):
        mime = "image/png"
    model_id = _get_model_id()
    client = get_client()
    try:
        out = predict_single_image(client, body, model_id=model_id, mime=mime)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=502, detail=str(e))
    return {
        "stamp": out.get("stamp"),
        "north_arrow": out.get("north_arrow"),
        "prediction_raw": out.get("prediction_raw"),
        "prediction_parsed": out.get("prediction_parsed"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
