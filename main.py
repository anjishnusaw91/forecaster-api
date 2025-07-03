import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

app = FastAPI()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LOOKBACK = 100
HIST_WINDOW = 30

class ForecastRequest(BaseModel):
    symbol: str
    days: int = 4

def fetch_data(symbol, lookback=LOOKBACK, hist_window=HIST_WINDOW, extra=30):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=lookback + hist_window + extra)
    df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)[['Close']]
    df.dropna(inplace=True)
    return df

def prepare_data(closes, lookback=LOOKBACK):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(closes)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def get_model_path(symbol):
    safe_symbol = symbol.replace('.', '_')
    return os.path.join(MODEL_DIR, f"{safe_symbol}_lstm.h5")

def get_scaler_path(symbol):
    safe_symbol = symbol.replace('.', '_')
    return os.path.join(MODEL_DIR, f"{safe_symbol}_scaler.npy")

def train_and_save(symbol, closes):
    X, y, scaler = prepare_data(closes)
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    model.save(get_model_path(symbol))
    np.save(get_scaler_path(symbol), scaler.scale_)
    return model, scaler

def load_model_and_scaler(symbol, closes):
    model_path = get_model_path(symbol)
    scaler_path = get_scaler_path(symbol)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = tf.keras.models.load_model(model_path)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(closes)
        scaler.scale_ = np.load(scaler_path)
        return model, scaler
    else:
        return train_and_save(symbol, closes)

@app.post("/predict")
async def predict(req: ForecastRequest):
    symbol = req.symbol
    days = req.days
    try:
        df = fetch_data(symbol)
        if len(df) < LOOKBACK + days:
            return {"success": False, "error": f"Not enough data for {symbol} to make predictions."}
        closes = df['Close'].values.reshape(-1, 1)
        model, scaler = load_model_and_scaler(symbol, closes)
        scaled_closes = scaler.transform(closes)
        # Historical prediction
        x_hist, y_hist = [], []
        for i in range(len(scaled_closes) - HIST_WINDOW, len(scaled_closes)):
            if i - LOOKBACK < 0:
                continue
            x_hist.append(scaled_closes[i-LOOKBACK:i])
            y_hist.append(closes[i][0])
        x_hist = np.array(x_hist)
        y_hist = np.array(y_hist)
        if len(x_hist) == 0:
            return {"success": False, "error": "Not enough data for historical prediction."}
        y_pred_hist = model.predict(x_hist)
        y_pred_hist_rescaled = scaler.inverse_transform(y_pred_hist).flatten()
        # Metrics
        rmse = float(np.sqrt(mean_squared_error(y_hist, y_pred_hist_rescaled)))
        r2 = float(r2_score(y_hist, y_pred_hist_rescaled))
        accuracy = float(np.mean(np.abs(y_pred_hist_rescaled - y_hist) / y_hist <= 0.02) * 100)
        # Future prediction
        last_seq = scaled_closes[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        future_preds_scaled = []
        for _ in range(days):
            pred_scaled = model.predict(last_seq)[0, 0]
            future_preds_scaled.append(pred_scaled)
            last_seq = np.append(last_seq[:, 1:, :], [[[pred_scaled]]], axis=1)
        future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
        future_preds = scaler.inverse_transform(future_preds_scaled).flatten().tolist()
        # Dates
        last_date = df.index[-1]
        forecast_dates = [(last_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(days)]
        hist_dates = df.index[-HIST_WINDOW:].strftime('%Y-%m-%d').tolist()
        actual_last30 = closes[-HIST_WINDOW:].flatten().tolist()
        pred_last30 = y_pred_hist_rescaled[-HIST_WINDOW:].tolist()
        return {
            "success": True,
            "forecast": {
                "dates": forecast_dates,
                "prices": future_preds
            },
            "historical": {
                "dates": hist_dates,
                "prices": actual_last30,
                "predictions": pred_last30
            },
            "metrics": {
                "accuracy": accuracy,
                "rmse": rmse,
                "r2": r2
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}