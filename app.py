import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import re
from model import MedicineRecommender

# Set page configuration
st.set_page_config(
    page_title="AI Medicine Recommender",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
    /* Base Reset */
    html, body {
        margin: 0;
        padding: 0;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #f0f4f8;
        color: #1a1a1a;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1565c0;
        text-align: center;
        margin: 1rem 0 0.5rem;
        animation: popup 0.5s ease-in-out;
    }

    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #546e7a;
        margin-bottom: 2rem;
        animation: popup 0.7s ease-in-out;
    }

    .disclaimer {
        background: #fff8e1;
        border-left: 6px solid #fbc02d;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        animation: popup 0.9s ease-in-out;
    }
    
    .disclaimer-text {
        color: #5d4037;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        animation: popup 1s ease-in-out;
    }

    .medicine-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #64b5f6;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: popup 1.2s ease-in-out;
    }

    .medicine-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }

    .critical-warning {
        background-color: #ffebee;
        border-left: 6px solid #f44336;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        animation: popup 1.5s ease-in-out;
    }

    .footer {
        text-align: center;
        margin-top: 4rem;
        color: #9e9e9e;
        font-size: 0.85rem;
        padding-bottom: 2rem;
    }

    /* Animations */
    @keyframes popup {
        0% {
            transform: scale(0.9);
            opacity: 0;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }

    /* Responsive Layouts */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }

        .sub-header {
            font-size: 1rem;
        }

        .disclaimer, .result-card, .medicine-card, .critical-warning {
            padding: 1rem;
        }

        .medicine-name {
            font-size: 1.1rem;
        }

        .medicine-desc, .medicine-dosage {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize the recommender model
@st.cache_resource
def load_model():
    return MedicineRecommender()

recommender = load_model()

# Centered Header
st.markdown('<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">AI Medicine Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter your symptoms to get personalized medicine recommendations</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Medical Disclaimer
st.markdown("""
<div class="disclaimer">
    <p class="disclaimer-text"><strong>Medical Disclaimer:</strong> This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read on this website.</p>
</div>
""", unsafe_allow_html=True)




# Create two columns for layout
col1, col2 = st.columns([2, 1])
# Center the content in the middle column
with col1:
    st.markdown('<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">', unsafe_allow_html=True)
    
    # User input
    st.markdown('<div style="text-align: center; max-width: 600px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">Describe Your Symptoms</h3>', unsafe_allow_html=True)
    symptoms_text = st.text_area(
        "Please describe what you're experiencing in detail:",
        height=150,
        placeholder="Example: I have a headache and slight fever since yesterday. The pain is concentrated on the front of my head.",
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional questions to improve recommendation
    with st.expander("Additional Information (Optional)", expanded=False):
        duration = st.selectbox(
            "How long have you been experiencing these symptoms?",
            ["Less than 24 hours", "1-3 days", "4-7 days", "More than a week"]
        )
        
        severity = st.slider(
            "On a scale of 1-10, how severe are your symptoms?",
            1, 10, 5
        )
        
        previous_conditions = st.multiselect(
            "Do you have any pre-existing medical conditions?",
            ["None", "Diabetes", "High Blood Pressure", "Heart Disease", "Asthma", "Allergies", "Other"]
        )
        
        medications = st.text_input(
            "Are you currently taking any medications?",
            placeholder="List any current medications"
        )

    # Submit button
    if st.button("Get Recommendations", type="primary"):
        if not symptoms_text:
            st.error("Please describe your symptoms before submitting.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                # Process the input and get recommendations
                result = recommender.recommend(
                    symptoms_text, 
                    {
                        'duration': duration,
                        'severity': severity,
                        'conditions': previous_conditions,
                        'medications': medications
                    }
                )
                
                # Display results
                if result['is_critical']:
                    st.markdown(f"""
                    <div class="critical-warning">
                        <p class="critical-text">⚠️ Medical Attention Recommended</p>
                        <p>{result['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card" style="text-align: center;">', unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Analysis:</strong> {result['recommendation']}</p>", unsafe_allow_html=True)
                    
                    if result['medicines']:
                        st.markdown("<h3>Recommended Medicines</h3>", unsafe_allow_html=True)
                        for medicine in result['medicines']:
                            st.markdown(f"""
                            <div class="medicine-card">
                                <p class="medicine-name">{medicine['name']}</p>
                                <p class="medicine-dosage">Dosage: {medicine['dosage']}</p>
                                <p class="medicine-desc">{medicine['description']}</p>
                                <a href="{medicine['url']}" target="_blank" style="color: #1565c0; text-decoration: underline;">More Information</a>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Information section below the main content
st.markdown("---")  # Add a horizontal line for separation
# How It Works Section
st.markdown('<h2 style="text-align: center; color: #1565c0;">How It Works</h2>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1rem; line-height: 1.8; color: #546e7a; max-width: 600px; margin: 0 auto;">
    <p>This AI system uses advanced natural language processing and machine learning to analyze your symptoms and provide relevant over-the-counter medicine recommendations.</p>
    <br>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
        <div style="text-align: left; max-width: 280px;">
            <strong>Key Features:</strong>
            <ul style="list-style-type: disc; padding-left: 20px;">
                <li>Identifies common symptoms</li>
                <li>Detects potential critical conditions</li>
                <li>Recommends appropriate medications</li>
                <li>Provides dosage information</li>
            </ul>
        </div>
        <div style="text-align: left; max-width: 280px;">
            <strong>Important Notes:</strong>
            <ul style="list-style-type: disc; padding-left: 20px;">
                <li>This is not a replacement for professional medical advice</li>
                <li>If symptoms are severe or persistent, consult a doctor</li>
                <li>Always read medication labels carefully</li>
            </ul>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Common symptoms section
st.markdown("---")  # Add another horizontal line for separation
st.markdown('<div style="text-align: center;"><h2>Common Symptoms</h2></div>', unsafe_allow_html=True)
common_symptoms = [
    "Headache", "Fever", "Cold", "Cough", 
    "Sore throat", "Stomach pain", "Nausea",
    "Joint pain", "Muscle pain", "Allergies"
]

# Display as clickable chips
symptom_html = '<div style="text-align: center;">'
for symptom in common_symptoms:
    symptom_html += f'<span style="display: inline-block; background-color: #1565c0; color: white; padding: 8px 16px; margin: 6px; border-radius: 20px; font-size: 0.95rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.backgroundColor=\'#64b5f6\'" onmouseout="this.style.backgroundColor=\'#1565c0\'">{symptom}</span>'
symptom_html += '</div>'

st.markdown(symptom_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>AI Medicine Recommendation System © 2023 | For educational purposes only</p>
    <p>Not for actual medical use | Always consult healthcare professionals</p>
</div>
""", unsafe_allow_html=True)
