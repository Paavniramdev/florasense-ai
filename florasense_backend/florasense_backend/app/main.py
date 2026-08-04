from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.model_service import flora_model
from app.schemas import HealthResponse, PredictResponse, TopPrediction

app = FastAPI(
    title="FloraSense AI - Vision Service",
    description="Flower species classification with Grad-CAM explainability and calibrated confidence.",
    version="0.1.0",
)

# Wide-open CORS for local dev with the Streamlit frontend. Tighten this
# (specific origins) before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup():
    flora_model.load()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if flora_model.loaded else "model not loaded",
        model_loaded=flora_model.loaded,
        num_classes=len(flora_model.idx_to_name),
        temperature=flora_model.temperature,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not flora_model.loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    raw_bytes = await file.read()
    try:
        pil_image = Image.open(BytesIO(raw_bytes))
        pil_image.load()  # force decode now, so corrupt files fail here, not later
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    try:
        species, confidence, is_confident, top5, gradcam_b64 = flora_model.predict(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        predicted_species=species,
        confidence=confidence,
        is_confident=is_confident,
        top5=[TopPrediction(species=s, confidence=c) for s, c in top5],
        gradcam_image_base64=gradcam_b64,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
