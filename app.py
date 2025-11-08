import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

# Page configuration
st.set_page_config(
    page_title="Sleep Disorder Prediction",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .none-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .apnea-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .insomnia-box {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('sleep_disorder_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.error("Model file not found. Please train the model first using the Colab notebook.")
        return None

model = load_model()

# Title and description
st.title("😴 Sleep Disorder Prediction System")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #1f77b4;'>Welcome to the Sleep Disorder Prediction System</h3>
        <p style='font-size: 16px;'>
            This system uses Machine Learning (Decision Tree Algorithm) to predict whether a patient has a sleep disorder.
            It can identify three outcomes:
            <ul>
                <li><b>None:</b> No sleep disorder detected</li>
                <li><b>Sleep Apnea:</b> A serious sleep disorder where breathing repeatedly stops and starts</li>
                <li><b>Insomnia:</b> Difficulty falling asleep or staying asleep</li>
            </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 About the Model")
    st.info("""
    **Model:** Decision Tree Classifier
    
    **Top Important Features:**
    1. Quality of Sleep
    2. Sleep Duration
    3. Stress Level
    4. Blood Pressure (Systolic/Diastolic)
    5. Heart Rate
    6. Daily Steps
    
    **Accuracy:** ~85-90% (varies based on data split)
    """)
    
    st.header("🎯 How to Use")
    st.markdown("""
    1. Enter patient information in the form
    2. Click "Predict Sleep Disorder"
    3. View the prediction and recommendations
    """)

# Main content
if model is not None:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📈 Model Insights", "ℹ️ Information"])
    
    with tab1:
        st.header("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Details")
            age = st.slider("Age", 20, 70, 35, help="Patient's age in years")
            gender = st.selectbox("Gender", ["Female", "Male"], help="Patient's gender")
            gender_encoded = 0 if gender == "Female" else 1
            
            st.subheader("Sleep Metrics")
            sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 9.0, 7.0, 0.1, 
                                      help="Average hours of sleep per night")
            quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7, 
                                        help="Self-rated sleep quality")
        
        with col2:
            st.subheader("Health Metrics")
            heart_rate = st.slider("Heart Rate (bpm)", 50, 100, 70, 
                                  help="Resting heart rate")
            systolic_pressure = st.slider("Systolic Blood Pressure (mmHg)", 100, 150, 120, 
                                         help="Upper blood pressure number")
            diastolic_pressure = st.slider("Diastolic Blood Pressure (mmHg)", 60, 100, 80, 
                                          help="Lower blood pressure number")
        
        with col3:
            st.subheader("Lifestyle Factors")
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5, 
                                    help="Self-rated stress level")
            daily_steps = st.slider("Daily Steps", 1000, 15000, 7000, 100, 
                                   help="Average daily step count")
            physical_activity = st.slider("Physical Activity Level (min/day)", 0, 120, 45, 5, 
                                         help="Minutes of physical activity per day")
        
        # Create prediction button
        st.markdown("---")
        predict_button = st.button("🔮 Predict Sleep Disorder", use_container_width=True)
        
        if predict_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'Age': [age],
                'Sleep Duration': [sleep_duration],
                'Quality of Sleep': [quality_of_sleep],
                'Stress Level': [stress_level],
                'Heart Rate': [heart_rate],
                'Systolic Pressure': [systolic_pressure],
                'Diastolic Pressure': [diastolic_pressure],
                'Daily Steps': [daily_steps],
                'Gender': [gender_encoded],
                'Physical Activity Level': [physical_activity]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            # Prediction box with color coding
            if prediction == "None":
                st.markdown(f"""
                    <div class='prediction-box none-box'>
                        ✅ No Sleep Disorder Detected
                    </div>
                    """, unsafe_allow_html=True)
                st.success("Great news! The model predicts no sleep disorder.")
            elif prediction == "Sleep Apnea":
                st.markdown(f"""
                    <div class='prediction-box apnea-box'>
                        ⚠️ Sleep Apnea Detected
                    </div>
                    """, unsafe_allow_html=True)
                st.warning("The model predicts Sleep Apnea. Please consult a healthcare professional.")
            else:  # Insomnia
                st.markdown(f"""
                    <div class='prediction-box insomnia-box'>
                        😴 Insomnia Detected
                    </div>
                    """, unsafe_allow_html=True)
                st.warning("The model predicts Insomnia. Consider consulting a sleep specialist.")
            
            # Probability visualization
            st.subheader("📊 Prediction Confidence")
            
            prob_df = pd.DataFrame({
                'Disorder': model.classes_,
                'Probability': prediction_proba * 100
            })
            
            fig = px.bar(prob_df, x='Disorder', y='Probability', 
                        color='Probability',
                        color_continuous_scale='RdYlGn_r',
                        text=prob_df['Probability'].round(1))
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(
                title="Prediction Probability Distribution",
                xaxis_title="Sleep Disorder Type",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if prediction == "None":
                st.markdown("""
                    - ✅ Maintain your current sleep habits
                    - ✅ Continue regular physical activity
                    - ✅ Keep stress levels manageable
                    - ✅ Maintain a consistent sleep schedule
                    """)
            elif prediction == "Sleep Apnea":
                st.markdown("""
                    - 🏥 **Consult a sleep specialist immediately**
                    - 📋 Consider a sleep study (polysomnography)
                    - 💪 Maintain a healthy weight
                    - 🚭 Avoid alcohol and smoking before bedtime
                    - 😴 Sleep on your side instead of your back
                    - 🩺 Regular monitoring of blood pressure
                    """)
            else:  # Insomnia
                st.markdown("""
                    - 🏥 **Consult a healthcare professional**
                    - 🛌 Establish a regular sleep schedule
                    - 😌 Practice relaxation techniques (meditation, deep breathing)
                    - 🚫 Avoid caffeine and screens before bedtime
                    - 🏃‍♂️ Increase physical activity during the day
                    - 📝 Consider cognitive behavioral therapy for insomnia (CBT-I)
                    """)
            
            # Risk factors identified
            st.subheader("⚠️ Risk Factors Identified")
            risk_factors = []
            
            if sleep_duration < 6:
                risk_factors.append("⚠️ Insufficient sleep duration")
            if quality_of_sleep < 6:
                risk_factors.append("⚠️ Poor sleep quality")
            if stress_level > 7:
                risk_factors.append("⚠️ High stress level")
            if systolic_pressure > 130 or diastolic_pressure > 85:
                risk_factors.append("⚠️ Elevated blood pressure")
            if heart_rate > 80:
                risk_factors.append("⚠️ Elevated resting heart rate")
            if daily_steps < 5000:
                risk_factors.append("⚠️ Low physical activity")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("✅ No significant risk factors identified")
    
    with tab2:
        st.header("📈 Model Insights")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        feature_names = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level', 
                        'Heart Rate', 'Systolic Pressure', 'Diastolic Pressure', 
                        'Daily Steps', 'Gender', 'Physical Activity Level']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title="Feature Importance in Sleep Disorder Prediction")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model statistics
        st.subheader("Model Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Algorithm", "Decision Tree")
        with col2:
            st.metric("Max Depth", model.max_depth)
        with col3:
            st.metric("Features Used", len(feature_names))
    
    with tab3:
        st.header("ℹ️ About Sleep Disorders")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌙 Sleep Apnea")
            st.markdown("""
                **What is it?**
                Sleep apnea is a serious sleep disorder where breathing repeatedly stops 
                and starts during sleep.
                
                **Common Symptoms:**
                - Loud snoring
                - Episodes of breathing cessation
                - Gasping for air during sleep
                - Morning headache
                - Difficulty staying asleep
                - Excessive daytime sleepiness
                
                **Risk Factors:**
                - Obesity
                - High blood pressure
                - Age (40+)
                - Male gender
                - Neck circumference
                """)
        
        with col2:
            st.subheader("😴 Insomnia")
            st.markdown("""
                **What is it?**
                Insomnia is a sleep disorder characterized by difficulty falling asleep, 
                staying asleep, or both.
                
                **Common Symptoms:**
                - Difficulty falling asleep
                - Waking up during the night
                - Waking up too early
                - Daytime tiredness
                - Irritability or depression
                - Difficulty concentrating
                
                **Risk Factors:**
                - High stress levels
                - Poor sleep habits
                - Mental health disorders
                - Medications
                - Medical conditions
                """)
        
        st.markdown("---")
        st.info("⚕️ **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.")

else:
    st.error("⚠️ Model not loaded. Please ensure 'sleep_disorder_model.pkl' is in the same directory.")
    st.info("Train the model using the provided Colab notebook first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Sleep Disorder Prediction System | Built with Streamlit & Scikit-learn</p>
        <p>🌙 Better Sleep, Better Health 🌙</p>
    </div>
    """, unsafe_allow_html=True)