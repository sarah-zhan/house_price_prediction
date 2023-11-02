import pickle
from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

modle_file = "model.bin"

with open(modle_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("predict_house_price")

@app.route("/predict", methods=["POST"])
def predict():
    house = request.get_json()

    X = dv.transform([house])
    dx = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
    prediction = model.predict(dx)
    price = np.expm1(prediction)[0]

    result = {
        "price": float(price)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)




