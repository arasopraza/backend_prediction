from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)  

model = joblib.load("komoditas.pkl")

@app.route('/predict', methods = ["POST"])
def predict():
    x1 = request.json["x1"]
    x2 = request.json["x2"]

    X = np.array([x1, x2]).reshape(1, -1)

    # make prediction using loaded regresi model
    y_pred = model.predict(X)

    data = {
        "prediction": int(y_pred[0])
    }

    response = {
        "message": "Success",
        "data": data
    }

    # return prediction as JSON response
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')