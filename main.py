# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import yfinance as yf
# import pandas as pd
# from prophet import Prophet

# app = FastAPI()

# # Allow all origins for now (you can restrict later)
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
#         data = yf.download(ticker, period="1y")
#         if data.empty:
#             return {"error": "No data found"}

#         df = data[['Close']].reset_index()
#         df = df.rename(columns={"Date": "ds", "Close": "y"})

#         model = Prophet()
#         model.fit(df)

#         future = model.make_future_dataframe(periods=future_days)
#         forecast = model.predict(future)
#         predicted = forecast[['ds', 'yhat']].tail(future_days)

#         return {
#             "ticker": ticker,
#             "predictions": predicted.to_dict(orient="records")
#         }

#     except Exception as e:
#         return {"error": str(e)}



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from prophet import Prophet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Stock Predictor API is Live"}

@app.get("/simulate")
def simulate_prediction():
    ticker = "RELIANCE.NS"
    future_days = 5

    try:
        # Download data
        data = yf.download(ticker, period="1y")
        
        if data.empty or 'Close' not in data.columns:
            return {"error": f"No historical data found for {ticker}"}

        # Clean and rename
        df = data.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Ensure 'ds' is datetime and 'y' is numeric
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df.dropna(inplace=True)

        if df.empty or df.shape[0] < 30:
            return {"error": f"Insufficient data to train model for {ticker}"}

        # Prophet model
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=future_days)
        forecast = model.predict(future)

        # Only future predictions
        predicted = forecast[['ds', 'yhat']].tail(future_days)

        # Round values
        predicted['yhat'] = predicted['yhat'].round(2)

        return {
            "ticker": ticker,
            "predictions": predicted.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}
