import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from upload_data import upload_file
from train_data import handle_null_values
from train_data import detect_outlier
from train_data import binning_data
from train_data import train_data

app = Flask(__name__)

model_bawang_merah = joblib.load("modelBawangMerah.pkl")
model_cabai_merah = joblib.load("modelCabaiMerah.pkl")
model_cabai_rawit = joblib.load("modelCabaiMerah.pkl")

@app.route('/predict', methods = ["POST"])
def predict():
    komoditas = request.json["komoditas"]
    x1 = request.json["x1"]
    x2 = request.json["x2"]

    X = np.array([x1, x2]).reshape(1, -1)

    if komoditas == "Bawang Merah":
        # make prediction using loaded regresi model
        y_pred = model_bawang_merah.predict(X)
        
        data = {
            "prediction": int(y_pred[0])
        }

        response = {
            "message": "Success",
            "data": data
        }

        # return prediction as JSON response
        return jsonify(response)
    elif komoditas == "Cabai Merah":
        # make prediction using loaded regresi model
        y_pred = model_cabai_merah.predict(X)
        
        data = {
            "prediction": int(y_pred[0])
        }

        response = {
            "message": "Success",
            "data": data
        }

        # return prediction as JSON response
        return jsonify(response)
    elif komoditas == "Cabai Rawit":
        # make prediction using loaded regresi model
        y_pred = model_cabai_rawit.predict(X)
        
        data = {
            "prediction": int(y_pred[0])
        }

        response = {
            "message": "Success",
            "data": data
        }

        # return prediction as JSON response
        return jsonify(response)
    
    response = {
        "message": "Failed",
        "data": ""
    }

    # return JSON response
    return jsonify(response)

@app.route('/uploader', methods = ['POST'])
def upload():
    komoditas = request.form.get("komoditas")
    file = request.files["file"]
    print(file)
    return upload_file(file, komoditas)

@app.route('/data-cleaning')
def cleaning():
    komoditas = request.args.get('komoditas')
    df = pd.read_csv("data/DataTraining" + komoditas.replace(" ", "") + ".csv")
    
    outliers = {}
    null_data = {}
    columns_to_check = ['Curah Hujan', 'Harga', 'Produksi']
    for column in columns_to_check:
        row = detect_outlier(df, column)
        if row is not None:
            outliers = row.to_dict(orient="records")

    null_value = handle_null_values(df)

    if null_value is not None:
        null_data = null_value.to_dict(orient="records")
        response = {
            "message": "Success get nulls value",
            "data": null_data
        }
    elif outliers is not None:
        response = {
            "message": "Success get outliers",
            "data": outliers
        }
    else:
        response = {
            "message": "Success",
            "data": df.to_dict(orient="records")
        }

    return response

@app.route('/binning-data')
def binning():
    response = {}
    komoditas = request.args.get('komoditas')
    df = pd.read_csv("data/DataTraining" + komoditas.replace(" ", "") + ".csv")
    columns_to_check = ['Curah Hujan', 'Harga', 'Produksi']
    
    for column in columns_to_check:
      binning_data(df, column)
    
    response["data"] = df.to_dict(orient='records')
    return jsonify(response), 200

@app.route('/train-model')
def train():
    komoditas = request.args.get('komoditas')
    return train_data(komoditas)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')