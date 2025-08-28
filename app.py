import streamlit as st
import pandas as pd
import joblib

lr = joblib.load("linear_model.joblib")
st.title("Tip Prediction App")

total_bill = st.number_input("Total Bill", min_value=0.0, value=20.0)
size = st.number_input("Table Size", min_value=1, value=2)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

def preprocess(total_bill, size, sex, smoker, day, time):
    data = {
        "total_bill": [total_bill],
        "size": [size],
        "sex_Male": [1 if sex == "Male" else 0],
        "smoker_Yes": [1 if smoker == "Yes" else 0],
        "day_Sat": [1 if day == "Sat" else 0],
        "day_Sun": [1 if day == "Sun" else 0],
        "day_Thur": [1 if day == "Thur" else 0],
        "time_Lunch": [1 if time == "Lunch" else 0],
    }
    return pd.DataFrame(data)

input_df = preprocess(total_bill, size, sex, smoker, day, time)


if st.button("Predict Tip"):
    prediction = lr.predict(input_df)[0]
    st.success(f"Predicted Tip: ${prediction:.2f}")
