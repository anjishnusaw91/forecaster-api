# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import yfinance as yf
# import pandas as pd
# from prophet import Prophet
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np
# import math

# app = FastAPI()

# # CORS for local dev and Vercel
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://stockpoint.vercel.app"],  # Replace with Vercel domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ForecastRequest(BaseModel):
#     symbol: str
#     days: int = 5

# @app.post("/api/generalForecaster")
# async def general_forecaster(req: ForecastRequest):
#     try:
#         ticker = req.symbol.upper()
#         prediction_days = max(1, min(req.days, 30))

#         # Step 1: Download historical stock data
#         data = yf.download(ticker, period="2y", interval="1d")
#         if data.empty or 'Close' not in data:
#             return {"success": False, "error": f"No data found for {ticker}"}

#         df = data.reset_index()[['Date', 'Close']]
#         df.columns = ['ds', 'y']
#         df.dropna(inplace=True)

#         # Step 2: Split into train/test
#         if len(df) < prediction_days + 30:
#             return {"success": False, "error": f"Not enough data for {ticker}"}

#         train_df = df[:-prediction_days]
#         test_df = df[-prediction_days:]

#         # Step 3: Train model
#         model = Prophet(daily_seasonality=True)
#         model.fit(train_df)

#         # Step 4: Forecast future
#         future = model.make_future_dataframe(periods=prediction_days)
#         forecast = model.predict(future)

#         # Step 5: Prepare responses
#         forecast_future = forecast[['ds', 'yhat']].tail(prediction_days)
#         train_predictions = model.predict(train_df[['ds']])

#         # Step 6: Calculate metrics
#         actual = train_df['y'].values
#         predicted = train_predictions['yhat'].values
#         rmse = math.sqrt(mean_squared_error(actual, predicted))
#         r2 = r2_score(actual, predicted)
#         accuracy = round(max(0.0, min(r2 * 100, 100.0)), 2)
#         confidence = round(100 - (rmse / np.mean(actual) * 100), 2)

#         return {
#             "success": True,
#             "forecast": {
#                 "dates": forecast_future['ds'].dt.strftime('%Y-%m-%d').tolist(),
#                 "prices": forecast_future['yhat'].round(2).tolist()
#             },
#             "historical": {
#                 "dates": df['ds'].dt.strftime('%Y-%m-%d').tolist(),
#                 "prices": df['y'].round(2).tolist(),
#                 "predictions": train_predictions['yhat'].round(2).tolist()
#             },
#             "metrics": {
#                 "accuracy": accuracy,
#                 "confidence": confidence,
#                 "rmse": round(rmse, 2),
#                 "predictionDays": prediction_days
#             }
#         }

#     except Exception as e:
#         return {"success": False, "error": str(e)}



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import timedelta

app = FastAPI()

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
        symbol = req.symbol.upper()
        days = max(1, min(req.days, 30))
        lookback = 60

        # STEP 1: Load data
        df = yf.download(symbol, period="2y", interval="1d")
        if df.empty or 'Close' not in df:
            return {"success": False, "error": f"No data found for {symbol}"}

        df = df[['Close']].dropna().reset_index()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])

        if len(df) < lookback + days + 30:
            return {"success": False, "error": f"Not enough data for {symbol}"}

        # STEP 2: Normalize
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['y']])

        X, y = [], []
        for i in range(lookback, len(scaled) - days):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i:i+days].flatten())

        X = np.array(X)
        y = np.array(y)

        # STEP 3: Train LSTM
        model = Sequential()
        model.add(LSTM(64, return_sequences=False, input_shape=(lookback, 1)))
        model.add(Dense(days))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        # STEP 4: Predict future
        last_seq = scaled[-lookback:].reshape(1, lookback, 1)
        future_scaled = model.predict(last_seq)[0].reshape(-1, 1)
        future = scaler.inverse_transform(future_scaled).flatten()

        forecast_start = df['ds'].iloc[-1] + timedelta(days=1)
        forecast_dates = [(forecast_start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]

        # STEP 5: Backtest on recent actuals
        recent_actuals = df['y'].values[-(days + lookback):]
        pred_on_train = model.predict(X)
        pred_rescaled = scaler.inverse_transform(pred_on_train)
        y_true = scaler.inverse_transform(y)

        r2 = round(r2_score(y_true.flatten(), pred_rescaled.flatten()), 2)
        rmse = round(math.sqrt(mean_squared_error(y_true.flatten(), pred_rescaled.flatten())), 2)
        confidence = round(100 - (rmse / np.mean(recent_actuals) * 100), 2)
        accuracy = round(np.mean(np.abs(pred_rescaled - y_true) / y_true <= 0.02) * 100, 2)

        # STEP 6: Add training predictions
        last30 = df[-30:]
        last30_scaled = scaler.transform(last30[['y']])
        X_hist = []
        for i in range(lookback, len(last30_scaled)):
            X_hist.append(last30_scaled[i-lookback:i])
        X_hist = np.array(X_hist)
        hist_preds = model.predict(X_hist)
        hist_preds_rescaled = scaler.inverse_transform(hist_preds)
        predicted_hist = hist_preds_rescaled[:, -1]

        return {
            "success": True,
            "forecast": {
                "dates": forecast_dates,
                "prices": [round(val, 2) for val in future.tolist()]
            },
            "historical": {
                "dates": last30['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "prices": last30['y'].round(2).tolist(),
                "predictions": predicted_hist.round(2).tolist()
            },
            "metrics": {
                "accuracy": accuracy,
                "confidence": confidence,
                "rmse": rmse,
                "predictionDays": days
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
