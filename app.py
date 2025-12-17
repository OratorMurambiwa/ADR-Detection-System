import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="ADR Detection System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    """Load the trained federated model and tokenizer"""
    try:
        # Path to your model directory
        model_path = "."  # Current directory since files are at root
        
        # Check if model files exist
        if not os.path.exists("config.json"):
            st.error("Model files not found. Please ensure config.json and model files are in the same directory as app.py")
            return None, None, None
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure all model files (config.json, tokenizer files, model weights) are in the app directory")
        return None, None, None

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\;\:\!\?\-\/\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_adr(review_text, model, tokenizer, device):
    """Make prediction on review text"""
    # Preprocess
    cleaned = clean_text(review_text)
    
    # Tokenize
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    # Determine risk level
    if pred == 1:  # ADR detected
        if confidence > 0.9:
            risk = "High"
        elif confidence > 0.7:
            risk = "Medium"
        else:
            risk = "Low"
    else:
        risk = "Low"
    
    return {
        "has_adr": bool(pred),
        "confidence": confidence,
        "risk_level": risk,
        "adr_probability": probs[0][1].item(),
        "no_adr_probability": probs[0][0].item()
    }

# Header
st.markdown('<p class="main-header">ADR Detection System</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    AI-Powered Adverse Drug Reaction Analysis using Federated Learning
</div>
""", unsafe_allow_html=True)

# Load model
model, tokenizer, device = load_model()

if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses a **Federated Learning** approach to detect adverse drug reactions 
    from patient reviews while preserving privacy.
    
    **Model Details:**
    - Architecture: DistilBERT
    - Training: Federated across 3 clients
    - Classes: ADR / No ADR
    """)
    
    st.divider()
    
    st.header("Settings")
    show_probabilities = st.checkbox("Show detailed probabilities", value=True)
    show_recommendations = st.checkbox("Show recommendations", value=True)
    
    st.divider()
    
    device_status = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    **Device:** {device_status}
    
    **Status:** Model Loaded Successfully
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Review Input")
    
    # Input form
    review_text = st.text_area(
        "Enter drug review:",
        height=200,
        placeholder="Example: I've been taking this medication for 3 weeks and experiencing severe headaches and nausea...",
        help="Enter the patient's review or description of their experience with the medication"
    )
    
    drug_name = st.text_input(
        "Drug Name (Optional):",
        placeholder="e.g., Aspirin, Metformin"
    )
    
    condition = st.text_input(
        "Medical Condition (Optional):",
        placeholder="e.g., Hypertension, Diabetes"
    )
    
    analyze_button = st.button("Analyze Review", type="primary", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    
    if analyze_button:
        if not review_text.strip():
            st.warning("Please enter a review to analyze")
        else:
            with st.spinner("Analyzing review..."):
                # Make prediction
                result = predict_adr(review_text, model, tokenizer, device)
                
                # Store in session state for history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'drug': drug_name or "Not specified",
                    'has_adr': result['has_adr'],
                    'confidence': result['confidence']
                })
                
                # Display results
                if result['has_adr']:
                    st.error("**ADR DETECTED**")
                    st.markdown("Potential adverse drug reaction identified")
                else:
                    st.success("**NO ADR DETECTED**")
                    st.markdown("Review suggests positive medication experience")
                
                st.divider()
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.1f}%",
                        delta=None
                    )
                
                with col_b:
                    st.metric(
                        "Risk Level",
                        result['risk_level'],
                        delta=None
                    )
                
                with col_c:
                    classification = "ADR" if result['has_adr'] else "No ADR"
                    st.metric(
                        "Classification",
                        classification,
                        delta=None
                    )
                
                # Progress bar for confidence
                st.progress(result['confidence'])
                
                # Detailed probabilities
                if show_probabilities:
                    st.divider()
                    st.markdown("**Probability Breakdown:**")
                    prob_df = pd.DataFrame({
                        'Class': ['No ADR', 'ADR'],
                        'Probability': [
                            result['no_adr_probability'],
                            result['adr_probability']
                        ]
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                
                # Recommendations
                if show_recommendations:
                    st.divider()
                    st.markdown("**Recommendations:**")
                    
                    if result['has_adr']:
                        st.markdown("""
                        - Consult with healthcare provider immediately
                        - Document all symptoms and timing
                        - Consider alternative medications
                        - Report to FDA MedWatch if severe
                        - Do not stop medication without medical advice
                        """)
                    else:
                        st.markdown("""
                        - Continue medication as prescribed
                        - Monitor for any new symptoms
                        - Maintain regular check-ups
                        - Report any changes to your doctor
                        """)
    else:
        st.info("Enter a review and click 'Analyze Review' to see results")

# Analysis History
if 'history' in st.session_state and len(st.session_state.history) > 0:
    st.divider()
    st.subheader("Recent Analysis History")
    
    history_df = pd.DataFrame(st.session_state.history)
    history_df['Result'] = history_df['has_adr'].apply(lambda x: "ADR" if x else "No ADR")
    history_df['Confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    history_df = history_df[['timestamp', 'drug', 'Result', 'Confidence']]
    history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(history_df, use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <strong>Disclaimer:</strong> This is a research tool and should not replace professional medical advice.
    Always consult healthcare professionals for medical decisions.
</div>
""", unsafe_allow_html=True)