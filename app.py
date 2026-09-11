from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
import pandas as pd
import joblib

app = FastAPI(
    title="Smart Irrigation API",
    description="Predict recommended irrigation amount in millimeters",
    version="1.0"
)

# Load trained model
model = joblib.load("smart_irrigation_random_forest.joblib")
features = joblib.load("smart_irrigation_features.joblib")


# Define the data expected from the user/sensor
class IrrigationInput(BaseModel):
    date: date
    air_pressure_hpa: float
    average_temperature_c: float
    maximum_temperature_c: float
    minimum_temperature_c: float
    precipitation_mm: float
    relative_humidity_pct: float
    wind_velocity_ms: float
    sunshine_hours: float
    et0: float
    water_demand: float


@app.get("/")
def home():
    return {
        "message": "Smart Irrigation API is running"
    }


@app.post("/predict")
def predict(data: IrrigationInput):

    # Automatically derive temporal features
    month = data.date.month
    day_of_year = data.date.timetuple().tm_yday

    # Map API inputs to the exact variables expected by the ML model
    input_data = pd.DataFrame([{
        "Air pressure on the ground(hPa)": data.air_pressure_hpa,
        "Average temperatures(℃)": data.average_temperature_c,
        "Maximum temperature(℃)": data.maximum_temperature_c,
        "Minimum temperature(℃)": data.minimum_temperature_c,
        "Precipitation(mm)": data.precipitation_mm,
        "Relative humidity(%)": data.relative_humidity_pct,
        "Wind velocity(m/s)": data.wind_velocity_ms,
        "Hours of sunshine(h)": data.sunshine_hours,
        "ET0": data.et0,
        "Water demand": data.water_demand,
        "Month": month,
        "Day_of_year": day_of_year
    }])

    # Ensure exact training feature order
    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    # Irrigation cannot be negative
    recommended_irrigation = max(0, float(prediction))

    return {
        "date": str(data.date),
        "recommended_irrigation_mm": round(recommended_irrigation, 3)
    }

