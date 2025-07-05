# <-------------------------Prophet--------------------------->
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math

from pandas.tseries.offsets import BDay  # Business days offset

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockpoint.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    symbol: str
    days: int = 5

@app.post("/api/generalForecaster")
async def general_forecaster(req: ForecastRequest):
    try:
        ticker = req.symbol.upper()
        prediction_days = max(1, min(req.days, 30))

        # Step 1: Get historical data
        data = yf.download(ticker, period="2y", interval="1d")
        if data.empty or 'Close' not in data:
            return {"success": False, "error": f"No data found for {ticker}"}

        df = data.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df.dropna(inplace=True)

        # Step 2: Train/test split
        if len(df) < prediction_days + 30:
            return {"success": False, "error": f"Not enough data for {ticker}"}

        train_df = df[:-prediction_days]
        test_df = df[-prediction_days:]

        # Step 3: Train Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(train_df)

        # Step 4: Create future dates: forecast next N *trading* days
        last_date = df['ds'].max()
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
        all_future = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days*2)

        # Prophet needs a wider date range to generate yhat values, then we filter
        future = model.make_future_dataframe(periods=prediction_days*2)
        forecast = model.predict(future)

        forecast_filtered = forecast[forecast['ds'].isin(future_dates)].copy()
        forecast_filtered = forecast_filtered[['ds', 'yhat']].head(prediction_days)

        # Step 5: Training prediction and metrics
        train_predictions = model.predict(train_df[['ds']])
        actual = train_df['y'].values
        predicted = train_predictions['yhat'].values

        rmse = math.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        accuracy = round(max(0.0, min(r2 * 100, 100.0)), 2)
        confidence = round(100 - (rmse / np.mean(actual) * 100), 2)

        return {
            "success": True,
            "forecast": {
                "dates": forecast_filtered['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "prices": forecast_filtered['yhat'].round(2).tolist()
            },
            "historical": {
                "dates": df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "prices": df['y'].round(2).tolist(),
                "predictions": train_predictions['yhat'].round(2).tolist()
            },
            "metrics": {
                "accuracy": accuracy,
                "confidence": confidence,
                "rmse": round(rmse, 2),
                "predictionDays": prediction_days
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Welcome! Stock Forecaster API is live. Use /ping to check status."

@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "API is live"
# ===============================================

# <-------------LSTM------------------------------------>
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import math
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import MinMaxScaler
# from datetime import timedelta
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://stockpoint.vercel.app"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ForecastRequest(BaseModel):
#     symbol: str
#     days: int = 5

# # PyTorch LSTM Model
# class StockLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(StockLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         _, (h_n, _) = self.lstm(x)
#         out = self.fc(h_n.squeeze(0))
#         return out

# @app.post("/api/generalForecaster")
# async def general_forecaster(req: ForecastRequest):
#     try:
#         symbol = req.symbol.upper()
#         days = max(1, min(req.days, 30))
#         lookback = 60
#         hidden_size = 64
#         epochs = 20

#         df = yf.download(symbol, period="2y", interval="1d")
#         if df.empty or 'Close' not in df:
#             return {"success": False, "error": f"No data found for {symbol}"}

#         df = df[['Close']].dropna().reset_index()
#         df.columns = ['ds', 'y']
#         df['ds'] = pd.to_datetime(df['ds'])

#         if len(df) < lookback + days + 30:
#             return {"success": False, "error": f"Not enough data for {symbol}"}

#         scaler = MinMaxScaler()
#         scaled = scaler.fit_transform(df[['y']])

#         X, y = [], []
#         for i in range(lookback, len(scaled) - days):
#             X.append(scaled[i-lookback:i])
#             y.append(scaled[i:i+days].flatten())

#         X, y = np.array(X), np.array(y)
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.float32)

#         dataset = TensorDataset(X_tensor, y_tensor)
#         loader = DataLoader(dataset, batch_size=16, shuffle=True)

#         model = StockLSTM(input_size=1, hidden_size=hidden_size, output_size=days)
#         criterion = nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#         for epoch in range(epochs):
#             for batch_X, batch_y in loader:
#                 batch_X = batch_X.view(-1, lookback, 1)
#                 optimizer.zero_grad()
#                 outputs = model(batch_X)
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()

#         with torch.no_grad():
#             last_seq = torch.tensor(scaled[-lookback:], dtype=torch.float32).view(1, lookback, 1)
#             future_scaled = model(last_seq).view(-1, 1).numpy()
#             future = scaler.inverse_transform(future_scaled).flatten()

#         forecast_start = df['ds'].iloc[-1] + timedelta(days=1)
#         forecast_dates = [(forecast_start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]

#         pred_on_train = model(X_tensor).detach().numpy()
#         pred_rescaled = scaler.inverse_transform(pred_on_train.reshape(-1, 1)).flatten()
#         y_true_rescaled = scaler.inverse_transform(y_tensor.reshape(-1, 1)).flatten()

#         r2 = round(r2_score(y_true_rescaled, pred_rescaled), 2)
#         rmse = round(math.sqrt(mean_squared_error(y_true_rescaled, pred_rescaled)), 2)
#         confidence = round(100 - (rmse / np.mean(df['y'].values[-(days+lookback):]) * 100), 2)
#         accuracy = round(np.mean(np.abs(pred_rescaled - y_true_rescaled) / y_true_rescaled <= 0.02) * 100, 2)

#         # Historical prediction line
#         last30 = df[-30:]
#         last30_scaled = scaler.transform(last30[['y']])
#         X_hist = []
#         for i in range(lookback, len(last30_scaled)):
#             X_hist.append(last30_scaled[i-lookback:i])
#         X_hist = torch.tensor(X_hist, dtype=torch.float32).view(-1, lookback, 1)
#         hist_preds = model(X_hist).detach().numpy()
#         predicted_hist = scaler.inverse_transform(hist_preds)[:, -1]

#         return {
#             "success": True,
#             "forecast": {
#                 "dates": forecast_dates,
#                 "prices": [round(val, 2) for val in future.tolist()]
#             },
#             "historical": {
#                 "dates": last30['ds'].dt.strftime('%Y-%m-%d').tolist(),
#                 "prices": last30['y'].round(2).tolist(),
#                 "predictions": predicted_hist.round(2).tolist()
#             },
#             "metrics": {
#                 "accuracy": accuracy,
#                 "confidence": confidence,
#                 "rmse": rmse,
#                 "predictionDays": days
#             }
#         }

#     except Exception as e:
#         return {"success": False, "error": str(e)}
