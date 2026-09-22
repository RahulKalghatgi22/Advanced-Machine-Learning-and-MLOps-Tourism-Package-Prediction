import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package - Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the
newly introduced **Wellness Tourism Package**, based on their profile and
how they were pitched. Enter the customer's details below to get a
prediction.
""")

# --- Customer profile ---
age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=200000, value=20000, step=500)

# --- Trip / household details ---
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
num_children_visiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
num_trips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=30, value=3)
preferred_property_star = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
passport = st.selectbox("Holds a Valid Passport?", ["No", "Yes"])
own_car = st.selectbox("Owns a Car?", ["No", "Yes"])

# --- Sales interaction details ---
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=180, value=15)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Assemble input into DataFrame (column names/order must match training data)
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': 1 if passport == "Yes" else 0,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': 1 if own_car == "Yes" else 0,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"This customer is **likely to purchase** the Wellness Tourism Package (probability: {probability:.1%}).")
    else:
        st.info(f"This customer is **unlikely to purchase** the Wellness Tourism Package (probability: {probability:.1%}).")
