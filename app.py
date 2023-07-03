from flask import Flask, jsonify
import joblib
import numpy as np

app = Flask(__name__)  

model = joblib.load("komoditas.pkl")

@app.route('/predict')
def index():
    X = np.array([1222, 0]).reshape(1, -1)
    # make prediction using loaded KNN model
    y_pred = model.predict(X)
    # return prediction as JSON response
    return jsonify({"prediction": int(y_pred[0])})
