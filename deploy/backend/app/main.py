from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model_service import ModelService


class PredictRequest(BaseModel):
    eeg_features: list[float] = Field(..., description="128-d EEG feature vector")
    speech_features: list[float] = Field(..., description="128-d speech feature vector")


class PredictResponse(BaseModel):
    model_type: str
    label: str
    confidence: float
    confidences: dict[str, float]


app = FastAPI(title="AMERS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelService(config_path=Path(__file__).resolve().parents[3] / "config" / "default.yaml")
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"

if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dir), name="frontend-assets")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": service.model_type}


@app.get("/")
def index() -> FileResponse:
    if not frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend files not found.")
    return FileResponse(frontend_dir / "index.html")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        out = service.predict(payload.eeg_features, payload.speech_features)
        return PredictResponse(**out)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
