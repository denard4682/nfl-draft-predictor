import streamlit as st
from src.predict import load_model, predict_player
from src.config import CLASSIFIER_PATH, REGRESSOR_PATH
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

'''st.title("🏈 NFL Combine Draft Predictor")
if st.button("Load Model"):
    st.write("Loading...")

clf, reg = load_model(CLASSIFIER_PATH, REGRESSOR_PATH)

height = st.number_imput("Height")
weight = st.number_input("Weight")
dash = st.number_input("40-yd Dash")
vertical = st.number_input("Vertical Jump")
bench = st.number_input("Bench Press")
broad = st.number_input("Broad Jump")
cone = st.number_input("3-Cone Drill")
shuttle = st.number_input("20-yd Shuttle")
position = st.number_input("Position(encoded)")


if st.button("predict"):
    features = [height, weight, dash, bench, vertical, broad, shuttle, cone, position]
    drafted, pick = predict_player(clf, reg, features)

    if drafted:
        st.success(f"Drafte Estimated Pick: {pick} ")
    else:
        st.error("Not Drafted")'''

st.title("🏈 NFL Draft Predictor")

st.write("App is running successfully")

