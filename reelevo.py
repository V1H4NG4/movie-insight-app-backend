from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi import HTTPException

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("syncdicator.pkl")

class MovieInput(BaseModel):
    Budget: float
    Runtime: float
    Rating: float  # Rating in percentage, like 85

@app.post("/predictDC")
def predict_box_office(data: MovieInput):
    input_data = np.array([[data.Budget, data.Runtime, data.Rating]])
    prediction = model.predict(input_data)
    return {"predicted_box_office": prediction[0]}

# --- Load new 6-feature RF pipeline ---
# Expects a Pipeline saved as 'nemesis_rf.pkl' that can handle string genres
try:
    model_nemesis = joblib.load("nemesis_rf.pkl")
    nemesis_loaded = True
except Exception as e:
    model_nemesis = None
    nemesis_loaded = False
    nemesis_load_error = str(e)

class MovieInput6(BaseModel):
    budget: float = Field(..., description="Budget in millions")
    runtime: float
    rating: float = Field(..., description="Rating in percentage (e.g., 85 for 85%)")
    popularity: float
    genre_1: str
    genre_2: Optional[str] = None  # allow None if your pipeline supports it

@app.post("/predictNemesis")
def predict_box_office_6f(data: MovieInput6):
    """
    Build a single-row DataFrame that matches the model's training columns exactly.
    Model expects:
      - numeric columns (budget, runtime, rating, popularity)
      - one-hot columns for genres (e.g., genre_1_Action, genre_2_Comedy, ...)
    """
    if model_nemesis is None:
        raise HTTPException(status_code=503, detail=f"Nemesis model not loaded: {nemesis_load_error}")

    expected = getattr(model_nemesis, "feature_names_in_", None)
    if expected is None:
        raise HTTPException(
            status_code=500,
            detail="Model has no feature_names_in_. Re-export using a DataFrame so column names persist."
        )

    try:
        # Start all-zeros row for every expected column
        row = {col: 0.0 for col in expected}

        # Fill numeric columns if present in training
        if "budget" in row:      row["budget"] = float(data.budget)
        if "runtime" in row:     row["runtime"] = float(data.runtime)
        if "rating" in row:      row["rating"] = float(data.rating)
        if "popularity" in row:  row["popularity"] = float(data.popularity)

        # One-hot: set only the columns that exist in training
        g1 = (data.genre_1 or "").strip()
        g2 = (data.genre_2 or "").strip()
        if g1:
            k1 = f"genre_1_{g1}"
            if k1 in row: row[k1] = 1.0
        if g2:
            k2 = f"genre_2_{g2}"
            if k2 in row: row[k2] = 1.0

        # IMPORTANT: do NOT add raw 'genre_1' or 'genre_2' keys; the model never saw them.
        X = pd.DataFrame([row], columns=expected)

        yhat = float(model_nemesis.predict(X)[0])
        return {"predicted_box_office": yhat}

    except Exception as e:
        # Return a 400 so the frontend shows the real message instead of "failed to fetch"
        raise HTTPException(status_code=400, detail=str(e))
    

# --- Load Marketing RF (budget, genre_1, genre_2, box_office -> marketing) ---
try:
    marketing_model = joblib.load("marketor_rf.pkl")
    genre1_enc = joblib.load("genre1_encoder.pkl")
    genre2_enc = joblib.load("genre2_encoder.pkl")
    marketing_loaded = True
    marketing_load_error = None
except Exception as e:
    marketing_model = None
    genre1_enc = None
    genre2_enc = None
    marketing_loaded = False
    marketing_load_error = str(e)


class MarketingInput(BaseModel):
    budget: float = Field(..., description="Budget in millions (same scale you trained with)")
    box_office: float = Field(..., description="Box office in millions (same scale)")
    genre_1: str = Field(..., description="Primary genre (e.g., Action, Drama)")
    genre_2: Optional[str] = Field(None, description="Secondary genre (optional)")


def _safe_le_transform(le, value: Optional[str]) -> int:
    """
    Safely transform a genre string with a LabelEncoder.
    - If the value is unseen, map to 'Unknown' if present, else first class (0).
    """
    if value is None:
        value = ""
    value = value.strip()

    # If already known to the encoder
    classes_list = le.classes_.tolist()
    if value in classes_list:
        return int(le.transform([value])[0])

    # Fallbacks for unseen labels
    if "Unknown" in classes_list:
        return int(le.transform(["Unknown"])[0])

    # Last resort: map to the first class (index 0)
    return 0


@app.post("/predictMarketing")
def predict_marketing(data: MarketingInput):
    """
    Predicts marketing spend from budget, genres, and box_office using the trained RF.
    Expectation: You trained with LabelEncoders (genre_1, genre_2) and saved them as pkl.
    """
    if not marketing_loaded or marketing_model is None or genre1_enc is None or genre2_enc is None:
        raise HTTPException(status_code=503, detail=f"Marketing model not loaded: {marketing_load_error}")

    try:
        g1 = _safe_le_transform(genre1_enc, data.genre_1)
        g2 = _safe_le_transform(genre2_enc, data.genre_2)

        X = pd.DataFrame([{
            "budget": float(data.budget),
            "genre_1": g1,
            "genre_2": g2,
            "box_office": float(data.box_office),
        }], columns=["budget", "genre_1", "genre_2", "box_office"])

        yhat = float(marketing_model.predict(X)[0])
        return {"predicted_marketing": yhat}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "dc_model": {
            "name": "syncdicator.pkl",
            "inputs": ["Budget (millions)", "Runtime", "Rating (percent)"],
            "endpoint": "/predictDC"
        },
        "nemesis_model": {
            "name": "nemesis_rf.pkl",
            "loaded": nemesis_loaded,
            "inputs": ["budget", "runtime", "rating (percent)", "popularity", "genre_1", "genre_2"],
            "endpoint": "/predictNemesis"
        },
        "marketing_model": {
            "name": "marketor_rf.pkl",
            "loaded": marketing_loaded,
            "inputs": ["budget", "box_office", "genre_1", "genre_2"],
            "endpoint": "/predictMarketing"
        }
    }