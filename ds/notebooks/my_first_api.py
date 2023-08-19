# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("my_first_api")

# Create input/output pydantic models
input_model = create_model("my_first_api_input", **{'flat_type': '4 ROOM', 'storey_range': '01 TO 03', 'floor_area_sqm': 84.0, 'flat_model': 'Simplified', 'lease_commence_date': 1985, 'latitude': 1.4346189498901367, 'longitude': 103.84058380126953, 'cbd_dist': 16807.609375, 'min_dist_mrt': 844.3861694335938})
output_model = create_model("my_first_api_output", prediction=255000.0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
