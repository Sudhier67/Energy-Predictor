import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_energy_model.pkl")

st.set_page_config(page_title="Energy Predictor", layout="centered")

st.title("🏠 Appliance Energy Consumption Predictor")
st.write("Predict household energy usage (Wh) using a Random Forest model")

st.sidebar.header("Input Features")

important_features = [
    "T2", "T6", "T_out", "RH_1", "T1", "RH_2", "hour", "T3", "RH_3", "Windspeed", "T8", "T9", "RH_9"
]

def user_inputs():
    T2 = st.sidebar.number_input("Living Room Temp (T2)", 0.0, 50.0, 20.0)
    T6 = st.sidebar.number_input("Outside North Temp (T6)", 0.0, 50.0, 7.0)
    T_out = st.sidebar.number_input("Outside Temp (T_out)", -20.0, 50.0, 5.0)
    RH_1 = st.sidebar.number_input("Kitchen Humidity (RH_1)", 0.0, 100.0, 40.0)
    T1 = st.sidebar.number_input("Kitchen Temp (T1)", 0.0, 50.0, 20.0)
    RH_2 = st.sidebar.number_input("Living Room Humidity (RH_2)", 0.0, 100.0, 40.0)
    hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
    T3 = st.sidebar.number_input("Laundry Room Temp (T3)", 0.0, 50.0, 20.0)
    RH_3 = st.sidebar.number_input("Laundry Room Humidity (RH_3)", 0.0, 100.0, 40.0)
    Windspeed = st.sidebar.number_input("Windspeed", 0.0, 50.0, 5.0)
    T8 = st.sidebar.number_input("Teenager Room Temp (T8)", 0.0, 50.0, 20.0)
    T9 = st.sidebar.number_input("Parents Room Temp (T9)", 0.0, 50.0, 20.0)
    RH_9 = st.sidebar.number_input("Parents Room Humidity (RH_9)", 0.0, 100.0, 40.0)

    data = {
        "T2": T2, "T6": T6, "T_out": T_out, "RH_1": RH_1, "T1": T1,
        "RH_2": RH_2, "hour": hour, "T3": T3, "RH_3": RH_3, "Windspeed": Windspeed, "T8": T8,
        "T9": T9, "RH_9": RH_9
    }
    return pd.DataFrame([data])

input_df = user_inputs()

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Predict Energy Consumption"):
    # Ensure all required columns for the model, fill missing with 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)
    st.success(f"🔋 Predicted Energy Consumption: {prediction[0]:.2f} Wh")
