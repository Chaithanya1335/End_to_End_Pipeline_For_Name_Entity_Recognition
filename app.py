import streamlit as st
from src.Components.Predict import Predict

st.set_page_config("Named Entity Recognition")

st.header("Named Entity Recognition")

text = st.text_input("Enter text",placeholder="Enter Text to extract Name Entity's")

if st.button("predict"):

    predictions  = Predict().predict_data(text=text)

    st.write(f"Predicted Name Entitys {predictions}")