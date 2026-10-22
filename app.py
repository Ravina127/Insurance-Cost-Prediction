
import pickle
import streamlit as st
import pandas as pd

# Load saved model files
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("columns.pkl", "rb") as file:
    model_columns = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Insurance Cost Prediction")
st.write(
    "Predict estimated medical insurance charges "
    "using Machine Learning."
)

st.divider()

# User inputs
st.subheader("Enter Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Prediction button
if st.button("Predict Insurance Cost"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    # One-hot encoding, matching notebook approach
    categorical_features = ["sex", "smoker", "region"]

    input_data = pd.get_dummies(
        input_data,
        columns=categorical_features,
        drop_first=True
    )

    # Match training feature columns
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale numeric features
    numeric_features = ["age", "bmi", "children"]

    input_data[numeric_features] = scaler.transform(
        input_data[numeric_features]
    )

    # Predict insurance charges
    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Insurance Charges: ₹{prediction:,.2f}"
    )

    st.caption(
        "This is a model-based estimate, not an actual insurance quote."
    )
