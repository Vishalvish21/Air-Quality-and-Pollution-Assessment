import pickle
import streamlit as st
import os
import numpy as np  

# Streamlit App Title
st.title("Air Quality Checker")

# Load Model
file_path = os.path.join("model", "final_model.pkl")

# Check if file exists before loading
if not os.path.exists(file_path):
    st.error("Model file not found! Please ensure 'final_model.pkl' is in the 'model' directory.")
else:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)

p1 = st.number_input("Temperature", min_value=0.0)
p2 = st.number_input("Humidity", min_value=0.0)
p3 = st.number_input("PM2.5", min_value=0.0)
p4 = st.number_input("PM10", min_value=0.0)
p5 = st.number_input("NO2", min_value=0.0)
p6 = st.number_input("SO2", min_value=0.0)
p7 = st.number_input("CO", min_value=0.0)
p8 = st.number_input("Proximity to Industrial Areas", min_value=0.0)
p9 = st.number_input("Population Density", min_value=0)

# Prediction
if st.button("Predict"):
    if 'model' in locals():  # Check if model is loaded
        input_data = np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9]])
        prediction = model.predict(input_data)
        st.write("Air quality is:", prediction[0])
    else:
        st.error("Model not loaded. Check the file path.")
