from typing import List

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    species: str
    confidence: float = Field(..., description="Calibrated confidence, 0-1")


class PredictResponse(BaseModel):
    predicted_species: str
    confidence: float = Field(..., description="Calibrated confidence for the top prediction, 0-1")
    is_confident: bool = Field(..., description="True if confidence is above the reliability threshold")
    top5: List[TopPrediction]
    gradcam_image_base64: str = Field(..., description="PNG image (base64), original + Grad-CAM heatmap side by side")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    temperature: float
