import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

#
st.write("""# House Price Prediction
This app predicts the house price in **Ames, Iowa**""")
st.write('---')

st.sidebar.header("Specify Your Parameters")

# customed features input function
def user_input_features():
    yearbuilt = st.sidebar.slider("Year Built", min_value=1800, max_value=2024, value=2000, step=1)
    grlivarea = st.sidebar.slider("Above ground living space square feet", min_value=300, max_value=7000, value=2000, step=100)
    garagecars = st.sidebar.slider("Garage size (Fit how many cars?)", min_value=0, max_value=4, value=2, step=1)
    centralair = st.sidebar.slider('Central air conditioning? Yes->"1"; No->"0', min_value=0, max_value=1, value=1)
    totalbsmtsf = st.sidebar.slider("Total square feet of basement area", min_value=0, max_value=7000, value=500, step=100)
    bsmtfinsf1 = st.sidebar.slider("Good Living Quarters square feet", min_value=0, max_value=7000, value=500, step=100)
    firstflrsf = st.sidebar.slider("First Floor square feet", min_value=300, max_value=1200, value=500, step=100)
    lotarea = st.sidebar.slider("Lot size in square feet", min_value=1300, max_value=8000, value=3000, step=100)
    masvnrarea = st.sidebar.slider("Masonry veneer area in square feet", min_value=0, max_value=500, value=100, step=10)
    garageyrblt = st.sidebar.slider("Year garage was built", min_value=1800, max_value=2024, value=2000, step=1)

    data = {
        "yearbuilt": yearbuilt,
        "grlivarea": grlivarea,
        "garagecars": garagecars,
        "centralair=y": centralair,
        "totalbsmtsf": totalbsmtsf,
        "bsmtfinsf1": bsmtfinsf1,
        "1stflrsf": firstflrsf,
        "lotarea": lotarea,
        "masvnrarea": masvnrarea,
        "garageyrblt": garageyrblt
    }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
# main panel
st.header("Specify Your Parameters")
st.write(df)
st.write("---")

# load model
modle_file = "model.bin"

with open(modle_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

def predict():
    X = dv.transform(df.to_dict(orient="records"))
    dx = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
    prediction = model.predict(dx)
    price = np.expm1(prediction)[0]
    result = {
        "price": float(price)
    }

    return result["price"]

st.subheader("Estimated house price")
st.write("👇👇👇👇👇👇👇👇👇👇")
st.write(predict())
st.write("👆👆👆👆👆👆👆👆👆👆")