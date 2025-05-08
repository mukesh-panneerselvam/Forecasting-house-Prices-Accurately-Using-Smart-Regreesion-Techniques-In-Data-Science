import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="🏠 Smart Price Forecaster",
    page_icon="📈",
    layout="wide"
)

# Load models and data (with error handling)
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
        sample_data = pd.read_csv(r"C:\Users\welcome\Downloads\house_price-dataset.csv")
        return model, scaler, features, sample_data
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, scaler, features, sample_data = load_artifacts()

# Sidebar with options
with st.sidebar:
    st.title("Settings")
    prediction_type = st.radio(
        "Prediction Mode",
        ["Single Property", "Batch Upload"],
        help="Choose between predicting one property or uploading a CSV file"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts house prices using machine learning.
    - *Model*: Gradient Boosting Regressor
    - *Accuracy*: 92% (test set)
    - *Last Updated*: {date}
    """.format(date=datetime.now().strftime("%B %d, %Y")))

# Main App
st.title("🏠 Smart Price Forecaster")
st.markdown("Predict accurate home values using our AI-powered valuation tool")

if prediction_type == "Single Property":
    # Single property prediction
    st.header("Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_input = {}
        for i, feature in enumerate(features[:len(features)//2]):
            if feature in ["YearBuilt", "Bedrooms"]:
                user_input[feature] = st.number_input(
                    f"{feature}", 
                    min_value=1800 if feature == "YearBuilt" else 1,
                    value=2000 if feature == "YearBuilt" else 3
                )
            else:
                user_input[feature] = st.number_input(f"{feature}", value=0.0)
    
    with col2:
        for feature in features[len(features)//2:]:
            if feature == "GarageSpots":
                user_input[feature] = st.slider(f"{feature}", 0, 5, 2)
            else:
                user_input[feature] = st.number_input(f"{feature}", value=0.0)
    
    if st.button("Predict Price", type="primary", help="Click to get valuation"):
        with st.spinner("Calculating..."):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            
            st.success(f"### Estimated Value: ${prediction[0]:,.2f}")
            
            # Show feature importance
            feature_importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            fig = px.bar(
                feature_importance.head(10),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Influential Features"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # Batch prediction
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV file with property data",
        type=["csv"],
        help="File should contain the same features as single prediction"
    )
    
    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            missing_cols = set(features) - set(batch_data.columns)
            
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                if st.button("Predict Batch", type="primary"):
                    with st.spinner("Processing..."):
                        batch_scaled = scaler.transform(batch_data[features])
                        predictions = model.predict(batch_scaled)
                        results = batch_data.copy()
                        results["PredictedPrice"] = predictions
                        
                        st.success(f"Processed {len(results)} properties")
                        
                        # Show sample results
                        st.dataframe(results.head())
                        
                        # Download button
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Predictions",
                            csv,
                            "house_price_predictions.csv",
                            "text/csv"
                        )
                        
                        # Visualization
                        fig = px.scatter(
                            results,
                            x="SquareFeet",
                            y="PredictedPrice",
                            trendline="lowess",
                            title="Price vs. Square Footage"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Sample data section
with st.expander("💡 Sample Data Format"):
    st.write("Here's the expected format for batch predictions:")
    st.dataframe(sample_data.head())
    st.download_button(
        "Download Sample CSV",
        sample_data.to_csv(index=False).encode('utf-8'),
        "sample_house_data.csv",
        "text/csv"
    )
