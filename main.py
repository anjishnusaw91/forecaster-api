# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import yfinance as yf
# import pandas as pd
# from prophet import Prophet

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def root():
#     return {"message": "Stock Predictor API is Live"}

# @app.get("/simulate")
# def simulate_prediction():
#     ticker = "RELIANCE.NS"
#     future_days = 5

#     try:
#         # Download data
#         data = yf.download(ticker, period="1y")
        
#         if data.empty or 'Close' not in data.columns:
#             return {"error": f"No historical data found for {ticker}"}

#         # Clean and rename
#         df = data.reset_index()[['Date', 'Close']]
#         df.columns = ['ds', 'y']
        
#         # Ensure 'ds' is datetime and 'y' is numeric
#         df['ds'] = pd.to_datetime(df['ds'])
#         df['y'] = pd.to_numeric(df['y'], errors='coerce')
#         df.dropna(inplace=True)

#         if df.empty or df.shape[0] < 30:
#             return {"error": f"Insufficient data to train model for {ticker}"}

#         # Prophet model
#         model = Prophet()
#         model.fit(df)

#         future = model.make_future_dataframe(periods=future_days)
#         forecast = model.predict(future)

#         # Only future predictions
#         predicted = forecast[['ds', 'yhat']].tail(future_days)

#         # Round values
#         predicted['yhat'] = predicted['yhat'].round(2)

#         return {
#             "ticker": ticker,
#             "predictions": predicted.to_dict(orient="records")
#         }

#     except Exception as e:
#         return {"error": str(e)}





from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math

app = FastAPI()

# CORS for local dev and Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Vercel domain in production
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

        # Step 1: Download historical stock data
        data = yf.download(ticker, period="2y", interval="1d")
        if data.empty or 'Close' not in data:
            return {"success": False, "error": f"No data found for {ticker}"}

        df = data.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df.dropna(inplace=True)

        # Step 2: Split into train/test
        if len(df) < prediction_days + 30:
            return {"success": False, "error": f"Not enough data for {ticker}"}

        train_df = df[:-prediction_days]
        test_df = df[-prediction_days:]

        # Step 3: Train model
        model = Prophet(daily_seasonality=True)
        model.fit(train_df)

        # Step 4: Forecast future
        future = model.make_future_dataframe(periods=prediction_days)
        forecast = model.predict(future)

        # Step 5: Prepare responses
        forecast_future = forecast[['ds', 'yhat']].tail(prediction_days)
        train_predictions = model.predict(train_df[['ds']])

        # Step 6: Calculate metrics
        actual = train_df['y'].values
        predicted = train_predictions['yhat'].values
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        accuracy = round(max(0.0, min(r2 * 100, 100.0)), 2)
        confidence = round(100 - (rmse / np.mean(actual) * 100), 2)

        return {
            "success": True,
            "forecast": {
                "dates": forecast_future['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "prices": forecast_future['yhat'].round(2).tolist()
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
