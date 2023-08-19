from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('my_third_pipeline')
cols = ['flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'latitude', 'longitude',
        'cbd_dist', 'min_dist_mrt']


@app.route('/')
def home():
    return render_template("home.html")


# @app.route('/predict', methods=['POST'])
# def predict():
#     int_features = [x for x in request.form.values()]
#     final = np.array(int_features)
#     data_unseen = pd.DataFrame([final], columns=cols)
#     prediction = predict_model(model, data=data_unseen, round=0)
#     prediction = int(prediction.Label[0])
#     return render_template('home.html', pred='Expected Bill will be {}'.format(prediction))


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    
    # Access the prediction value using the 'prediction_label' column
    prediction_value = prediction['prediction_label'][0]
    
    return render_template('home.html', pred='Estimated Price of house will be {:.2f}'.format(prediction_value))



@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
