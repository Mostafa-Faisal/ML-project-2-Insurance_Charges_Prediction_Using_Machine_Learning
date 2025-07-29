import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Set page configuration
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 20px 0;
    }
    .feature-importance {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏥 Insurance Charges Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
Predict insurance charges based on personal attributes using machine learning models
</div>
""", unsafe_allow_html=True)

# Function to load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to load the saved model
        if os.path.exists('models/best_insurance_model.pkl'):
            with open('models/best_insurance_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            with open('models/feature_columns.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
            
            with open('models/model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
            
            return model, feature_columns, model_info
        else:
            st.error("❌ Model files not found! Please run the training notebook first.")
            return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# Function to prepare input data
def prepare_input_data(age, sex, bmi, children, smoker, region, feature_columns):
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Apply one-hot encoding
    input_encoded = pd.get_dummies(input_data, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    return input_encoded

# Load the model
model, feature_columns, model_info = load_model()

if model is not None:
    # Display model information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", model_info['model_name'])
    
    with col2:
        st.metric("R² Score", f"{model_info['r2_score']:.4f}")
    
    with col3:
        st.metric("Training Date", model_info['training_date'])
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Personal Information</h2>', unsafe_allow_html=True)
        
        # Input fields
        age = st.slider("Age", min_value=18, max_value=100, value=30, help="Age of the person")
        
        sex = st.selectbox("Gender", ["male", "female"], help="Gender of the person")
        
        bmi = st.slider("BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, 
                       help="Body Mass Index - Normal range is 18.5-24.9")
        
        children = st.slider("Number of Children", min_value=0, max_value=10, value=0, 
                           help="Number of children/dependents covered by insurance")
        
        smoker = st.selectbox("Smoking Status", ["no", "yes"], help="Does the person smoke?")
        
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], 
                            help="Geographic region")
        
        # BMI category display
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "#74c0fc"
        elif bmi < 25:
            bmi_category = "Normal"
            bmi_color = "#51cf66"
        elif bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "#ffd43b"
        else:
            bmi_category = "Obese"
            bmi_color = "#ff6b6b"
        
        st.markdown(f"""
        <div style="background-color: {bmi_color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <strong>BMI Category: {bmi_category}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🔮 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Make prediction button
        if st.button("🚀 Predict Insurance Charges", type="primary", use_container_width=True):
            try:
                # Prepare input data
                input_data = prepare_input_data(age, sex, bmi, children, smoker, region, feature_columns)
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: #1f77b4; margin-bottom: 15px;">💰 Predicted Insurance Charges</h3>
                    <h1 style="color: #e74c3c; text-align: center; font-size: 3rem; margin: 20px 0;">
                        ${prediction:,.2f}
                    </h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk assessment
                if prediction < 5000:
                    risk_level = "Low Risk"
                    risk_color = "#27ae60"
                    risk_icon = "✅"
                elif prediction < 15000:
                    risk_level = "Medium Risk"
                    risk_color = "#f39c12"
                    risk_icon = "⚠️"
                else:
                    risk_level = "High Risk"
                    risk_color = "#e74c3c"
                    risk_icon = "🚨"
                
                st.markdown(f"""
                <div style="background-color: {risk_color}20; padding: 15px; border-radius: 8px; border-left: 4px solid {risk_color};">
                    <h4 style="color: {risk_color}; margin: 0;">{risk_icon} Risk Assessment: {risk_level}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 📊 Insights")
                insights = []
                
                if smoker == "yes":
                    insights.append("🚬 Smoking significantly increases insurance costs")
                
                if bmi > 30:
                    insights.append("⚖️ High BMI may contribute to higher charges")
                
                if age > 50:
                    insights.append("👴 Age is a factor in insurance pricing")
                
                if children > 2:
                    insights.append("👨‍👩‍👧‍👦 Multiple dependents affect coverage costs")
                
                if not insights:
                    insights.append("✅ Profile shows relatively standard risk factors")
                
                for insight in insights:
                    st.write(f"• {insight}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    # Feature importance section (if Random Forest model)
    if hasattr(model, 'feature_importances_'):
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📈 Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(importance_df.set_index('Feature')['Importance'])
        
        with col2:
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            st.markdown("**Top 5 Most Important Features:**")
            for i, (_, row) in enumerate(importance_df.head().iterrows()):
                percentage = row['Importance'] * 100
                st.write(f"{i+1}. {row['Feature']}: {percentage:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information
    st.markdown("---")
    st.markdown("### 📋 About This Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Information:**
        - Built using scikit-learn and XGBoost
        - Trained on insurance dataset
        - Predicts charges based on personal attributes
        - Uses one-hot encoding for categorical variables
        """)
    
    with col2:
        st.markdown("""
        **Important Notes:**
        - Predictions are estimates based on historical data
        - Actual charges may vary based on specific insurance policies
        - This tool is for educational/demonstration purposes
        - Consult with insurance professionals for accurate quotes
        """)
    
    # Sidebar with additional options
    with st.sidebar:
        st.markdown("### 🔧 Model Options")
        
        st.info(f"**Current Model:** {model_info['model_name']}")
        st.info(f"**Model Accuracy:** {model_info['r2_score']:.1%}")
        
        st.markdown("### 📊 Quick Statistics")
        st.write("**Average Insurance Charges by Smoking Status:**")
        st.write("• Non-smokers: ~$8,400")
        st.write("• Smokers: ~$32,000")
        
        st.write("**BMI Impact:**")
        st.write("• Normal BMI: Lower charges")
        st.write("• High BMI: Higher charges")
        
        st.markdown("### 💡 Tips")
        st.write("• Maintain healthy BMI")
        st.write("• Avoid smoking")
        st.write("• Consider preventive care")
        st.write("• Compare insurance plans")

else:
    st.error("""
    ❌ **Model not found!**
    
    Please follow these steps:
    1. Run the Insurance_Charges_Prediction.ipynb notebook first
    2. Make sure all cells are executed successfully
    3. Ensure the 'models' directory is created with the saved model files
    4. Then run this Streamlit app again
    """)
    
    st.markdown("### 🔄 Alternative: Use Sample Model")
    if st.button("Create Sample Model for Demo"):
        st.info("This would create a sample model for demonstration purposes...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Built with ❤️ using Streamlit | Insurance Charges Prediction ML Project
</div>
""", unsafe_allow_html=True)
