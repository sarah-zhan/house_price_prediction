import streamlit as st
import pandas as pd
import shap
import pickle
import xgboost as xgb
import numpy as np

# load model
modle_file = "model.bin"

with open(modle_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

X = pd.DataFrame(dv.get_feature_names_out())


"""
main features:
yearbuilt: Original construction date
grlivarea: Above grade (ground) living area square feet
garagecars: Size of garage in car capacity
centralair=y: Central air conditioning
totalbsmtsf: Total square feet of basement area
bsmtfinsf1: Type 1 finished square feet
1stflrsf: First Floor square feet
lotarea: Lot size in square feet
masvnrarea: Masonry veneer area in square feet
garageyrblt: Year garage was built
"""

st.write("""# House Price Prediction
This app predicts the house price in **Ames, Iowa**""")
st.write('---')

st.sidebar.header("Specify Your Parameters")

# customed features input function
def user_input_features():
    yearbuilt = st.sidebar.slider("Year Built", X.yearbuilt.min(), X.yearbuilt.max(), X.yearbuilt.mean())
    grlivarea = st.sidebar.slider("Above ground living space", X.grlivarea.min(), X.grlivarea.max(), X.grlivarea.mean())
    garagecars = st.sidebar.slider("Garage size (hold how many car?)", X.garagecars.min(), X.garagecars.max(), X.garagecars.mean())
    centralair = st.sidebar.slider("Central air conditioning", X["centralair=y"].min(), X["centralair=y"].max())
    totalbsmtsf = st.sidebar.slider("Total square feet of basement area", X.totalbsmtsf.min(), X.totalbsmtsf.max(), X.totalbsmtsf.mean())
    bsmtfinsf1 = st.sidebar.slider("Good Living Quarters square feet", X.bsmtfinsf1.min(), X.bsmtfinsf1.max(), X.bsmtfinsf1.mean())
    firstflrsf = st.sidebar.slider("First Floor square feet", X["1stflrsf"].min(), X["1stflrsf"].max(), X["1stflrsf"].mean())
    lotarea = st.sidebar.slider("Lot size in square feet", X["lotarea"].min(), X["lotarea"].max(), X["lotarea"].mean())
    masvnrarea = st.sidebar.slider("Masonry veneer area in square feet", X["masvnrarea"].min(), X["masvnrarea"].max(), X["masvnrarea"].mean())
    garageyrblt = st.sidebar.slider("Year garage was built", X["garageyrblt"].min(), X["garageyrblt"].max(), X["garageyrblt"].mean())

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


