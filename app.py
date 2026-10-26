import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------

preprocessor = joblib.load("preprocessor.pkl")
feature_selector = joblib.load("feature_selector.pkl")
model = joblib.load("xgb_model.pkl")


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.set_page_config(
    page_title="Swiggy Delivery Time Prediction",
    page_icon="🍔",
    layout="wide"
)

st.title("🍔 Swiggy Delivery Time Prediction")
st.write("Enter the order details below to predict the delivery time.")


# --------------------------------------------------
# INPUTS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)


# -------- Numerical inputs --------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    ratings = st.number_input(
        "Ratings",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

    vehicle_condition = st.selectbox(
        "Vehicle Condition",
        [0, 1, 2, 3, 4, 5],
        index=2
    )

    multiple_deliveries = st.selectbox(
        "Multiple Deliveries",
        [0, 1, 2, 3],
        index=0
    )

    distance = st.number_input(
        "Distance (km)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1
    )

    pickup_time_minutes = st.number_input(
        "Pickup Time (minutes)",
        min_value=0,
        max_value=120,
        value=15
    )


# -------- Categorical inputs --------

with col2:

    weather = st.selectbox(
        "Weather",
        [
            "sunny",
            "stormy",
            "sandstorms",
            "sindy",
            "cloudy",
            "fog"
        ]
    )

    traffic = st.selectbox(
        "Traffic",
        [
            "low",
            "medium",
            "high",
            "jam"
        ]
    )

    type_of_order = st.selectbox(
        "Type of Order",
        [
            "snack",
            "meal",
            "drinks",
            "buffet"
        ]
    )

    type_of_vehicle = st.selectbox(
        "Type of Vehicle",
        [
            "motorcycle",
            "scooter",
            "electric_scooter",
            "bicycle"
        ]
    )

    festival = st.selectbox(
        "Festival",
        [
            "no",
            "yes"
        ]
    )

    city_type = st.selectbox(
    "City Type",
    ["urban", "metropolitian", "semi-urban"]
    )
    city_name = st.selectbox(
    "City Name",
    [
        "INDO",
        "BANG",
        "COIMB",
        "CHEN",
        "HYD",
        "RANCHI",
        "MYS",
        "DEH",
        "KOC",
        "PUNE",
        "LUDH",
        "KNP",
        "MUM",
        "KOL",
        "JAP",
        "SUR",
        "GOA",
        "AURG",
        "AGR",
        "VAD",
        "ALH",
        "BHP"
    ]
)

# -------- Date / time inputs --------

with col3:

    order_day = st.number_input(
        "Order Day",
        min_value=1,
        max_value=31,
        value=15
    )

    order_month = st.number_input(
        "Order Month",
        min_value=1,
        max_value=12,
        value=3
    )

    order_day_of_week = st.selectbox(
        "Order Day of Week",
        [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"
        ]
    )

    is_weekend = st.selectbox(
        "Is Weekend?",
        [0, 1]
    )

    order_time_hour = st.number_input(
        "Order Time Hour",
        min_value=0,
        max_value=23,
        value=18
    )

    order_time_of_day = st.selectbox(
        "Order Time of Day",
        [
            "after_midnight",
            "morning",
            "afternoon",
            "evening",
            "night"
        ]
    )


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.write("")

if st.button("🚀 Predict Delivery Time"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "ratings": [ratings],
        "weather": [weather],
        "traffic": [traffic],
        "vehicle_condition": [vehicle_condition],
        "type_of_order": [type_of_order],
        "type_of_vehicle": [type_of_vehicle],
        "multiple_deliveries": [multiple_deliveries],
        "festival": [festival],
        "city_type": [city_type],
        "city_name": [city_name],
        "order_day": [order_day],
        "order_month": [order_month],
        "order_day_of_week": [order_day_of_week],
        "is_weekend": [is_weekend],
        "pickup_time_minutes": [pickup_time_minutes],
        "order_time_hour": [order_time_hour],
        "order_time_of_day": [order_time_of_day],
        "distance": [distance]
    })

    try:

        # Preprocessing
        processed_data = preprocessor.transform(input_data)

        # Feature selection
        selected_data = feature_selector.transform(processed_data)

        # Prediction
        prediction = model.predict(selected_data)[0]

        # Display result
        st.success(
            f"Estimated Delivery Time: {prediction:.1f} minutes"
        )

        st.info(
            f"Approximately {round(prediction)} minutes"
        )

    except Exception as e:

        st.error("Prediction could not be made.")

        st.write("Error:", e)