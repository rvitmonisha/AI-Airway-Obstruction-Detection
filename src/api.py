import os
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
from tensorflow.keras.models import load_model

app = FastAPI(
    title="🫁 Bio-Inspired AI Airway Diagnostics API Core",
    description=(
        "Production-grade, high-performance REST microservice. "
        "Ingests digital lung acoustics and clinical metadata parameters to evaluate "
        "pathological airway obstruction risks using a decoupled deep learning architecture."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = ["Both (Crackles & Wheezes)", "Crackles", "Normal", "Wheezes"]
MODEL_PATH = "models/airway_cnn_model.h5"

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None


# =========================================================================
# WARNING-FREE VALIDATION SCHEMAS (Pydantic V2 Compliant)
# =========================================================================
class BAAISMetricsResponse(BaseModel):
    baais_coefficient: float = Field(..., description="Calculated scaling coefficient based on compliance.", json_schema_extra={"example": 0.72})
    base_compliance: float = Field(..., description="Estimated volumetric elasticity baseline tracking.", json_schema_extra={"example": 0.45})
    overall_risk_percentage: float = Field(..., description="Aggregated risk projection metric.", json_schema_extra={"example": 46.6})

class TimelineSliceResponse(BaseModel):
    time_start: float = Field(..., description="Window segment initiation timestamp in seconds.", json_schema_extra={"example": 0.0})
    time_end: float = Field(..., description="Window segment termination timestamp in seconds.", json_schema_extra={"example": 3.0})
    classification: str = Field(..., description="Acoustic prediction condition classification tag.", json_schema_extra={"example": "Crackles"})
    raw_confidence: float = Field(..., description="Raw probability distribution value from the neural net.", json_schema_extra={"example": 88.5})
    baais_optimized_confidence: float = Field(..., description="Predictive tuning value modulated by BAAIS matrix.", json_schema_extra={"example": 92.1})

class DiagnosticMatrixResponse(BaseModel):
    status: str = Field(..., description="Execution pipeline completion confirmation state.", json_schema_extra={"example": "success"})
    computed_diagnostics: BAAISMetricsResponse
    timeline_analysis: List[TimelineSliceResponse]


def calculate_baais_metrics(age: int, gender: str, asthma: str, copd: str, aqi: int):
    gender_scalar = 1.05 if gender.lower() == "male" else 0.95
    base_compliance = max(0.4, (100 - age) / 100.0 * gender_scalar)
    
    vagal_reflex_index = 1.0
    if asthma.lower() == "yes": vagal_reflex_index += 0.35
    if copd.lower() == "yes": vagal_reflex_index += 0.25
        
    aqi_stressor = max(1.0, aqi / 100.0)
    baais_coefficient = max(0.3, min(1.5, round(base_compliance / (aqi_stressor * (vagal_reflex_index ** 0.5)), 2)))
    
    base_risk = (age / 120.0) * 20.0 
    calculated_risk = base_risk + (25.0 if asthma.lower() == "yes" else 0.0) + (30.0 if copd.lower() == "yes" else 0.0) + ((aqi / 300.0) * 15.0)
    if baais_coefficient < 0.65: calculated_risk += 10.0 
        
    return {
        "baais_coefficient": baais_coefficient,
        "base_compliance": round(base_compliance, 2),
        "overall_risk_percentage": round(max(5.0, min(98.5, calculated_risk)), 1)
    }


@app.get("/health", tags=["Infrastructure Monitoring"])
def health_check():
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model weights uninitialized.")
    return {"status": "healthy", "model_loaded": True}


@app.post("/api/v1/diagnose", response_model=DiagnosticMatrixResponse, tags=["Clinical Diagnostics Processing Core"])
async def diagnose_airway_stream(
    file: UploadFile = File(..., description="Binary .wav digital clinical audio lung recording stream payload."),
    age: int = Form(..., description="Patient chronological age in years.", ge=1, le=120, examples=[55]),
    gender: Literal["Male", "Female", "Other"] = Form(..., description="Biological sex classifications."),
    asthma: Literal["No", "Yes"] = Form(..., description="Pre-existing clinical background diagnostic confirmation for asthma."),
    copd: Literal["No", "Yes"] = Form(..., description="Pre-existing chronic obstructive pulmonary disease validation tracker.")
):
    if model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Predictive CNN engine uninitialized.")
        
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Requires structured .wav audio payloads.")

    try:
        temp_path = f"stream_cache_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
            
        signal, sr = librosa.load(temp_path, sr=None)
        if os.path.exists(temp_path): os.remove(temp_path)
        
        metrics = calculate_baais_metrics(age, gender, asthma, copd, aqi=48)
        
        window_size = int(3.0 * sr)
        step_size = int(2.5 * sr)
        start_idx = 0
        slices_results = []
        
        while start_idx + window_size <= len(signal):
            segment = signal[start_idx : start_idx + window_size]
            mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
            matrix = librosa.power_to_db(mel, ref=np.max)
            
            target_width = 128
            if matrix.shape[1] < target_width:
                pad_width = target_width - matrix.shape[1]
                matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
            else:
                matrix = matrix[:, :target_width]
                
            matrix_scaled = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)
            input_tensor = np.expand_dims(matrix_scaled, axis=(0, -1))
            
            preds = model.predict(input_tensor, verbose=0)
            pred_idx = np.argmax(preds[0])
            
            raw_conf = float(preds[0][pred_idx] * 100)
            calibrated_conf = min(100.0, raw_conf * (1.0 + (1.0 - metrics["baais_coefficient"]) * 0.15))
            
            slices_results.append({
                "time_start": round(start_idx / sr, 1),
                "time_end": round((start_idx + window_size) / sr, 1),
                "classification": LABELS[pred_idx],
                "raw_confidence": round(raw_conf, 1),
                "baais_optimized_confidence": round(calibrated_conf, 1)
            })
            start_idx += step_size

        return {
            "status": "success",
            "computed_diagnostics": metrics,
            "timeline_analysis": slices_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Pipeline processing anomaly: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8001)