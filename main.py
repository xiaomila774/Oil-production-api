from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np, pandas as pd, io, json, joblib
from keras.models import load_model
import pathlib
import logging

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading model and scaler...")

BASE_DIR = pathlib.Path(__file__).resolve().parent
model   = load_model(BASE_DIR / "model.h5")
scaler  = joblib.load(BASE_DIR / "scaler.gz")
meta    = json.load(open(BASE_DIR / "meta.json", encoding="utf-8"))

FEATURE_COLS = meta["feature_cols"]
WINDOW_SIZE  = meta["window_size"]

logger.info("Model and scaler loaded.")
logger.info(f"Features: {FEATURE_COLS}")

class Features(BaseModel):
    CHOKE_SIZE_t:     float
    ON_STREAM_HRS_t:  float
    Pmin_t:           float
    Pmax_t:           float
    Tmin_t:           float
    Tmax_t:           float
    BORE_water_VOL_t: float
    BORE_GAS_VOL_t:   float

app = FastAPI(title="FY102HF Forecast API")

@app.get("/")
def root():
    return {"msg": "FY102HF API ready. See /docs"}

@app.post("/predict")
def predict(f: Features):
    try:
        X = np.array([[f.dict()[c] for c in FEATURE_COLS]], dtype=float)
        X_scaled = scaler.transform(X)
        y = model.predict(X_scaled)[0][0]
        return {"pred_oil_rate": float(y)}
    except Exception as e:
        logger.exception("Single prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
            mime, suf = "text/csv", ".csv"
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file.file)
            mime, suf = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")
        else:
            raise HTTPException(status_code=400, detail="只接受 CSV / Excel")

        if not all(c in df.columns for c in FEATURE_COLS):
            miss = [c for c in FEATURE_COLS if c not in df.columns]
            raise HTTPException(status_code=400, detail=f"缺少列: {miss}")

        X_scaled = scaler.transform(df[FEATURE_COLS].astype(float))
        df["pred_oil_rate"] = model.predict(X_scaled).flatten()

        buf = io.BytesIO()
        if suf == ".csv":
            df.to_csv(buf, index=False, encoding="utf-8-sig")
        else:
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name="Prediction")
        buf.seek(0)
        out_name = file.filename.replace(suf, f"_pred{suf}")
        return StreamingResponse(buf, media_type=mime,
                                 headers={"Content-Disposition": f"attachment;filename={out_name}"})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))
