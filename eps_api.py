import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import pandas as pd
from datetime import datetime

from prediction.query_script.query_dataset import building_dataset
from prediction.query_script.query_prediction import predict_ticker

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class TickerRequest(BaseModel):
    tickers: List[str]


@app.post("/predict")
def predict(request: TickerRequest):
    if not request.tickers:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty")

    # build dataset
    builder = building_dataset()
    df = builder.build_latest_eps_dataset(request.tickers)

    if df.empty:
        return {"message": "No data could be fetched for the provided tickers"}

    # save dataset
    csv_path = "prediction\\query_script\\ticker_ds.csv"
    df.to_csv(csv_path, index=False)

    # load predictor and predict
    predictor = predict_ticker()
    result_df = predictor.predict_from_csv(csv_path=csv_path)

    # predictions with SHAP explanations
    predictions_response = result_df[[
        'Ticker', 'Earnings Date', 'prediction', 'confidence', 'shap_explanation'
    ]].to_dict(orient='records')

    # global feature importance (same for all)
    feature_importance_response = predictor.feature_importance_global

    return {
        "predictions": predictions_response,
        "feature_importance": feature_importance_response
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
