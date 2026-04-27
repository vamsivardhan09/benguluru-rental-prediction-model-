import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and feature info
@st.cache_resource
def load_model():
    model = joblib.load('rent_prediction_model.pkl')
    feature_info = joblib.load('model_features.pkl')
    return model, feature_info

model, feature_info = load_model()

# Title
st.title("🏠 Bangalore Rent Prediction Model")
st.markdown("Predict monthly rental prices for properties in Bangalore based on property features.")

# Sidebar for inputs
st.sidebar.header("Property Details")

# Input fields
property_size = st.sidebar.number_input("Property Size (sq ft)", min_value=100, max_value=10000, value=1000)
bathroom = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5], index=1)
floor = st.sidebar.number_input("Floor Number", min_value=0, max_value=50, value=2)
total_floor = st.sidebar.number_input("Total Floors in Building", min_value=1, max_value=50, value=4)
property_age = st.sidebar.number_input("Property Age (years)", min_value=0, max_value=50, value=2)

# Amenities
st.sidebar.subheader("Amenities")
gym = st.sidebar.checkbox("Gym")
lift = st.sidebar.checkbox("Lift")
swimming_pool = st.sidebar.checkbox("Swimming Pool")

# Categorical features
property_type = st.sidebar.selectbox("Property Type", ['BHK1', 'BHK2', 'BHK3', 'BHK4', 'RK1', 'BHK4PLUS'])
furnishing = st.sidebar.selectbox("Furnishing", ['SEMI_FURNISHED', 'FULLY_FURNISHED', 'NOT_FURNISHED'])
locality = st.sidebar.selectbox("Locality", [
    'Bellandur', 'Brookefield', 'Electronic_City', 'K.R Puram',
    'Kaggadasapura', 'Varthur', 'Whitefield', 'Yelahanka'
])
lease_type = st.sidebar.selectbox("Lease Type", ['FAMILY', 'ANYONE', 'BACHELOR', 'COMPANY'])
parking = st.sidebar.selectbox("Parking", ['BOTH', 'TWO_WHEELER', 'FOUR_WHEELER', 'NONE'])

# Convert amenities to 0/1
gym_val = 1 if gym else 0
lift_val = 1 if lift else 0
swimming_pool_val = 1 if swimming_pool else 0

# Create input dataframe
input_data = pd.DataFrame({
    'property_size': [property_size],
    'bathroom': [bathroom],
    'floor': [floor],
    'total_floor': [total_floor],
    'property_age': [property_age],
    'gym': [gym_val],
    'lift': [lift_val],
    'swimming_pool': [swimming_pool_val],
    'type': [property_type],
    'furnishing': [furnishing],
    'locality': [locality],
    'lease_type': [lease_type],
    'parking': [parking]
})

# Predict button
if st.sidebar.button("Predict Rent"):
    try:
        prediction = model.predict(input_data)[0]

        # Display result
        st.success(f"🏠 Predicted Monthly Rent: ₹{prediction:,.0f}")

        # Additional info
        st.info(f"""
        **Property Summary:**
        - Type: {property_type}
        - Size: {property_size} sq ft
        - Bathrooms: {bathroom}
        - Location: {locality}
        - Furnishing: {furnishing}
        - Amenities: {'Gym, ' if gym else ''}{'Lift, ' if lift else ''}{'Swimming Pool' if swimming_pool else ''}
        """)

        # Model info
        st.markdown("---")
        st.markdown("**Model Information:**")
        st.write("• Algorithm: Linear Regression")
        st.write("• Training Data: 2,054 properties")
        st.write("• R² Score: 0.59")
        st.write("• Mean Absolute Error: ₹4,548")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and scikit-learn*")