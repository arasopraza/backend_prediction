from flask import Flask, jsonify, request
from upload_data import upload_file
import joblib
import numpy as np

app = Flask(__name__)

model_bawang_merah = joblib.load("komoditas.pkl")
model_cabai_merah = joblib.load("komoditas.pkl")
model_cabai_rawit = joblib.load("komoditas.pkl")

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
    return upload_file(file, komoditas)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')