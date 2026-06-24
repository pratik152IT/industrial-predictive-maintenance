import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the FastAPI app with meta-data for Eaton's standards
app = FastAPI(
    title="Eaton Intelligent Power - Predictive Maintenance API",
    description="An MLOps microservice to predict industrial machine failures based on sensor telemetries.",
    version="1.0.0"
)

# Define paths to our downloaded cloud artifacts
MODEL_PATH = os.path.join("models", "model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Load the model and scaler when the server starts up
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("🚀 Cloud model and scaler successfully loaded into production server!")
else:
    raise RuntimeError("Artifacts missing! Ensure model.pkl and scaler.pkl are inside the 'models' directory.")

# Define the exact structure of incoming real-time sensor data using Pydantic
class SensorInput(BaseModel):
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    Type_L: int  # 1 if Low quality variant, else 0
    Type_M: int  # 1 if Medium quality variant, else 0

@app.get("/")
def home():
    return {"status": "Healthy", "message": "Eaton ML Maintenance Service is running."}

@app.post("/predict")
def predict_failure(data: SensorInput):
    try:
        # 1. Convert the incoming JSON payload into a dictionary
        input_dict = data.model_dump()
        
        # 2. Re-introduce original spaces matching raw dataset feature headers expected by the model
        formatted_dict = {
            "Air temperature [K]": input_dict["Air_temperature_K"],
            "Process temperature [K]": input_dict["Process_temperature_K"],
            "Rotational speed [rpm]": input_dict["Rotational_speed_rpm"],
            "Torque [Nm]": input_dict["Torque_Nm"],
            "Tool wear [min]": input_dict["Tool_wear_min"],
            "Type_L": input_dict["Type_L"],
            "Type_M": input_dict["Type_M"]
        }
        
        # 3. Transform data into a single-row Pandas DataFrame
        input_df = pd.DataFrame([formatted_dict])
        
        # 4. Use the saved cloud scaler to normalize the features perfectly 
        scaled_features = scaler.transform(input_df)
        
        # 5. Execute model inference
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        # 6. Return response payload back to the client
        return {
            "machine_failure_predicted": int(prediction),
            "failure_probability": float(np.round(probability, 4)),
            "action_required": "Schedule Immediate Maintenance" if prediction == 1 else "None. Machine is stable."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")