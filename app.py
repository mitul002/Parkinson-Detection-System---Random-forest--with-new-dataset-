"""
Enhanced Streamlit App for Parkinson's Disease Detection
with SHAP and LIME explanations
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import shap
import lime
import lime.lime_tabular
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
from feature_extraction import SpiralFeatureExtractor

# Page configuration
st.set_page_config(
    page_title="Parkinson's Detection System",
    page_icon="🌀",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_model_files():
    try:
        # Ensure we're using absolute paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(base_path, "random_forest_model.pkl"))
        scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
        imputer = joblib.load(os.path.join(base_path, "imputer.pkl"))
        label_encoder = joblib.load(os.path.join(base_path, "label_encoder.pkl"))
        feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))
        return model, scaler, imputer, label_encoder, feature_names
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        raise e

# Initialize feature extractor
@st.cache_resource
def get_feature_extractor():
    return SpiralFeatureExtractor(use_cnn=True)

# SHAP explanation
def get_shap_explanation(model, features, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Determine predicted class
    pred_proba = model.predict_proba(features)[0]
    class_idx = int(np.argmax(pred_proba))
    
    # Select SHAP values for predicted class
    if isinstance(shap_values, list):
        # Old API: list of arrays per class
        vec = np.array(shap_values[class_idx][0])
    elif shap_values.ndim == 3:
        # Multiclass: (n_samples, n_classes, n_features)
        vec = np.array(shap_values[0, class_idx, :])
    elif shap_values.ndim == 2:
        # Binary: (n_samples, n_features)
        vec = np.array(shap_values[0])
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    
    # Create custom bar chart of top 10 features
    imp = np.abs(vec)
    top_idx = np.argsort(-imp)[:15]  # Get top 15 candidates
    top_features = [feature_names[i] for i in top_idx]
    top_values = vec[top_idx]
    
    # Filter out near-zero values (keep only meaningful contributions)
    threshold = np.max(imp) * 0.005  # Keep features with at least 0.5% of max importance (lowered from 1%)
    meaningful_mask = np.abs(top_values) >= threshold
    top_features_filtered = [f for f, m in zip(top_features, meaningful_mask) if m]
    top_values_filtered = top_values[meaningful_mask]
    
    # Ensure at least 5 features are shown if available
    if len(top_features_filtered) < 5 and len(top_features) >= 5:
        top_features_filtered = top_features[:5]
        top_values_filtered = top_values[:5]
    
    # Determine number of features to show
    n_features = len(top_features_filtered)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_values_filtered]  # Green for positive, Red for negative
    ax.barh(list(reversed(top_features_filtered)), list(reversed(top_values_filtered)), color=list(reversed(colors)))
    ax.set_title(f'Top {n_features} SHAP Contributions')
    ax.set_xlabel('SHAP Value')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Positive (Supports)'),
                      Patch(facecolor='#e74c3c', label='Negative (Against)')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    
    return fig, shap_values

# LIME explanation
def get_lime_explanation(model, features, feature_names, class_names):
    # Generate synthetic training data for LIME based on feature statistics
    np.random.seed(42)
    n_samples = 500
    # Create more varied training data
    mean_vals = np.mean(features, axis=0)
    std_vals = np.std(features, axis=0)
    # Replace zero std with small value to avoid division by zero
    std_vals = np.where(std_vals == 0, 0.1, std_vals)
    training_data = np.random.randn(n_samples, features.shape[1]) * std_vals + mean_vals
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data,
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode='classification',
        discretize_continuous=False,
        random_state=42
    )
    
    # Get predicted class
    pred_class = int(model.predict(features)[0])
    
    exp = explainer.explain_instance(
        features[0],
        model.predict_proba,
        num_features=10,
        labels=(pred_class,)  # Specify which label to explain
    )
    
    return exp, pred_class

def process_single_image(uploaded_file, model, scaler, imputer, label_encoder, feature_names, feature_extractor):
    """Process a single image and return results"""
    # Save temporary file for feature extraction
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    features = feature_extractor.extract_features(temp_path)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    if features is None:
        return None
    
    # Prepare feature vector
    feature_dict = {}
    for i, value in enumerate(features['cnn_features']):
        feature_dict[f'cnn_feat_{i}'] = value
    feature_dict.update(features['handcrafted_features'])
    
    feature_vector = pd.DataFrame([feature_dict])
    
    # Handle skeleton feature aliases
    alias_pairs = [
        ("Skel_Endpoints", "skel_endpoints"),
        ("Skel_Branchpoints", "skel_branchpoints"),
        ("Skel_Length", "skel_length"),
    ]
    for a, b in alias_pairs:
        if a in feature_names and a not in feature_vector.columns and b in feature_vector.columns:
            feature_vector[a] = feature_vector[b]
        if b in feature_names and b not in feature_vector.columns and a in feature_vector.columns:
            feature_vector[b] = feature_vector[a]
    
    # Select and order features
    feature_vector = feature_vector[feature_names]
    
    # Preprocess
    X = imputer.transform(feature_vector)
    X = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    
    return {
        'prediction': predicted_class,
        'probabilities': probabilities,
        'confidence': probabilities[prediction] * 100,
        'features': features,
        'X': X
    }

def main():
    # Load model and components
    try:
        model, scaler, imputer, label_encoder, feature_names = load_model_files()
        feature_extractor = get_feature_extractor()
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return
    
    # Title and description
    st.title("🌀 Parkinson's Disease Detection System")
    st.markdown("""
    This system analyzes spiral drawings to detect signs of Parkinson's disease using advanced machine learning techniques.
    Upload spiral drawing images to get predictions and detailed explanations.
    """)
    
    # Mode selection
    st.sidebar.header("Upload Mode")
    upload_mode = st.sidebar.radio(
        "Choose upload mode:",
        ["Single Image", "Batch Upload"]
    )
    
    if upload_mode == "Single Image":
        # Upload method selection
        st.subheader("📤 Upload Image")
        upload_method = st.radio(
            "Choose upload method:",
            ["📁 Upload from Files", "📷 Take Photo with Camera"],
            horizontal=True
        )
        
        uploaded_file = None
        
        if upload_method == "📁 Upload from Files":
            uploaded_file = st.file_uploader(
                "Upload a spiral drawing image...", 
                type=["jpg", "jpeg", "png"],
                key="file_uploader"
            )
        else:  # Camera option
            camera_photo = st.camera_input("Take a photo of spiral drawing", key="camera_input")
            if camera_photo:
                uploaded_file = camera_photo
        
        if uploaded_file:
            # Display original image
            col1, col2 = st.columns([1, 1.3])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, width=400)
            
            # Extract features and predict (need results before showing explanation)
            with st.spinner("Analyzing image..."):
                result = process_single_image(uploaded_file, model, scaler, imputer, 
                                             label_encoder, feature_names, feature_extractor)
                
                if result is None:
                    st.error("Could not extract features from the image. Please try another image.")
                    return
            
            # Add explanation under the image in col1
            with col1:
                st.markdown("---")
                st.subheader("🔍 Why this prediction?")
                
                handcrafted = result['features']['handcrafted_features']
                
                if result['prediction'] == "Parkinson":
                    # Analyze actual feature values for THIS specific image
                    jerkiness = handcrafted.get('Jerkiness', 0)
                    wobble = handcrafted.get('Wobble', 0)
                    curvature_std = handcrafted.get('Curv_Std', 0)
                    spiral_r2 = handcrafted.get('Spiral_R2', 1.0)
                    stroke_width_std = handcrafted.get('Std_Thickness', 0)
                    loop_var = handcrafted.get('Loop_Variation', 0)
                    path_rmse = handcrafted.get('Path_RMSE', 0)
                    
                    # Build specific reasons based on THIS image's features
                    reasons = []
                    
                    # Shakiness analysis
                    if jerkiness > 0.015:
                        reasons.append("**extremely shaky and jerky** lines")
                    elif jerkiness > 0.010:
                        reasons.append("**very shaky** drawing")
                    elif jerkiness > 0.006:
                        reasons.append("**noticeable shakiness**")
                    elif jerkiness > 0.003:
                        reasons.append("**slight shakiness**")
                    
                    # Wobble analysis
                    if wobble > 0.20:
                        reasons.append("**severe wobbling** throughout")
                    elif wobble > 0.12:
                        reasons.append("**excessive wobbling**")
                    elif wobble > 0.08:
                        reasons.append("**noticeable wobbling**")
                    elif wobble > 0.04:
                        reasons.append("**some wobbling**")
                    
                    # Loop variation analysis
                    if loop_var > 50:
                        reasons.append("**very inconsistent loop sizes**")
                    elif loop_var > 30:
                        reasons.append("**irregular loop patterns**")
                    elif loop_var > 20:
                        reasons.append("**uneven loops**")
                    
                    # Curvature analysis
                    if curvature_std > 0.0008:
                        reasons.append("**highly irregular curves**")
                    elif curvature_std > 0.0005:
                        reasons.append("**irregular curves**")
                    elif curvature_std > 0.0003:
                        reasons.append("**somewhat irregular curves**")
                    
                    # Spiral shape analysis
                    if spiral_r2 < 0.3:
                        reasons.append("**very poorly formed spiral**")
                    elif spiral_r2 < 0.5:
                        reasons.append("**poorly formed spiral**")
                    elif spiral_r2 < 0.7:
                        reasons.append("**uneven spiral shape**")
                    
                    # Line width analysis
                    if stroke_width_std > 3.0:
                        reasons.append("**extremely unsteady line thickness**")
                    elif stroke_width_std > 2.0:
                        reasons.append("**very unsteady lines**")
                    elif stroke_width_std > 1.2:
                        reasons.append("**unsteady line thickness**")
                    
                    # Path error analysis
                    if path_rmse > 30:
                        reasons.append("**very erratic path**")
                    elif path_rmse > 20:
                        reasons.append("**erratic drawing path**")
                    
                    # If no specific reasons found, use general ones
                    if not reasons:
                        reasons.append("**subtle tremor patterns**")
                        reasons.append("**irregular movement**")
                    
                    # Build natural explanation
                    explanation_text = "This image shows **Parkinson's disease** because the drawing has "
                    if len(reasons) == 1:
                        explanation_text += reasons[0]
                    elif len(reasons) == 2:
                        explanation_text += f"{reasons[0]} and {reasons[1]}"
                    else:
                        explanation_text += ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
                    explanation_text += "."
                    
                    st.warning(explanation_text)
                    st.info("💡 These signs indicate tremors and difficulty with fine motor control, typical of Parkinson's patients.")
                
                elif result['prediction'] == "Healthy":
                    # Analyze actual feature values for THIS specific image
                    jerkiness = handcrafted.get('Jerkiness', 1)
                    wobble = handcrafted.get('Wobble', 1)
                    spiral_r2 = handcrafted.get('Spiral_R2', 0)
                    curvature_std = handcrafted.get('Curv_Std', 1)
                    stroke_width_std = handcrafted.get('Std_Thickness', 10)
                    loop_var = handcrafted.get('Loop_Variation', 100)
                    circularity = handcrafted.get('Circularity', 0)
                    
                    # Build specific reasons based on THIS image's features
                    reasons = []
                    
                    # Smoothness analysis
                    if jerkiness < 0.002:
                        reasons.append("**exceptionally smooth and steady**")
                    elif jerkiness < 0.004:
                        reasons.append("**very smooth with minimal shakiness**")
                    elif jerkiness < 0.006:
                        reasons.append("**smooth drawing**")
                    elif jerkiness < 0.008:
                        reasons.append("**reasonably smooth**")
                    
                    # Steadiness analysis
                    if wobble < 0.03:
                        reasons.append("**extremely steady with no wobbling**")
                    elif wobble < 0.06:
                        reasons.append("**very steady**")
                    elif wobble < 0.10:
                        reasons.append("**steady hand movement**")
                    elif wobble < 0.15:
                        reasons.append("**mostly steady**")
                    
                    # Loop consistency analysis
                    if loop_var < 12:
                        reasons.append("**highly consistent loop sizes**")
                    elif loop_var < 18:
                        reasons.append("**consistent loops**")
                    elif loop_var < 25:
                        reasons.append("**fairly consistent loops**")
                    
                    # Spiral shape analysis
                    if spiral_r2 > 0.85:
                        reasons.append("**perfectly formed spiral**")
                    elif spiral_r2 > 0.75:
                        reasons.append("**well-formed spiral shape**")
                    elif spiral_r2 > 0.65:
                        reasons.append("**good spiral shape**")
                    
                    # Curve consistency analysis
                    if curvature_std < 0.0002:
                        reasons.append("**highly uniform curves**")
                    elif curvature_std < 0.0004:
                        reasons.append("**consistent curves**")
                    elif curvature_std < 0.0006:
                        reasons.append("**fairly consistent curves**")
                    
                    # Line quality analysis
                    if stroke_width_std < 0.8:
                        reasons.append("**uniform line thickness**")
                    elif stroke_width_std < 1.2:
                        reasons.append("**stable line quality**")
                    elif stroke_width_std < 1.8:
                        reasons.append("**decent line control**")
                    
                    # Circularity analysis
                    if circularity > 0.8:
                        reasons.append("**excellent circular form**")
                    elif circularity > 0.7:
                        reasons.append("**good circular form**")
                    
                    # If no specific reasons found
                    if not reasons:
                        reasons.append("**good control**")
                        reasons.append("**normal drawing patterns**")
                    
                    # Build natural explanation
                    explanation_text = "This image shows a **healthy** drawing because it is "
                    if len(reasons) == 1:
                        explanation_text += reasons[0]
                    elif len(reasons) == 2:
                        explanation_text += f"{reasons[0]} and {reasons[1]}"
                    else:
                        explanation_text += ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
                    explanation_text += "."
                    
                    st.success(explanation_text)
                    st.info("✅ These characteristics indicate good motor control and coordination.")
                
                else:  # Noisy
                    # Analyze actual feature values for THIS specific image
                    entropy = handcrafted.get('Entropy', 0)
                    stroke_length = handcrafted.get('Stroke_Length', 0)
                    skel_length = handcrafted.get('Skel_Length', 0)
                    stroke_width_std = handcrafted.get('Std_Thickness', 0)
                    mean_thickness = handcrafted.get('Mean_Thickness', 0)
                    
                    # Build specific issues based on THIS image's features
                    issues = []
                    
                    if entropy < 2.5:
                        issues.append("**very poor image quality**")
                    elif entropy < 3.5:
                        issues.append("**poor image quality**")
                    elif entropy < 4.5:
                        issues.append("**low image quality**")
                    
                    if stroke_length < 500:
                        issues.append("**incomplete or faint drawing**")
                    elif stroke_length < 1000:
                        issues.append("**weak or unclear lines**")
                    
                    if skel_length < 300:
                        issues.append("**very sparse line detection**")
                    elif skel_length < 600:
                        issues.append("**unclear line structure**")
                    
                    if mean_thickness < 1.5:
                        issues.append("**extremely thin or faint lines**")
                    elif mean_thickness < 2.5:
                        issues.append("**very faint lines**")
                    
                    if stroke_width_std > 4.0:
                        issues.append("**extremely inconsistent image quality**")
                    elif stroke_width_std > 3.0:
                        issues.append("**very inconsistent quality**")
                    elif stroke_width_std > 2.0:
                        issues.append("**inconsistent quality**")
                    
                    if not issues:
                        issues.append("**quality problems**")
                    
                    # Build natural explanation
                    explanation_text = "This image has "
                    if len(issues) == 1:
                        explanation_text += issues[0]
                    elif len(issues) == 2:
                        explanation_text += f"{issues[0]} and {issues[1]}"
                    else:
                        explanation_text += ", ".join(issues[:-1]) + f", and {issues[-1]}"
                    explanation_text += ", making accurate analysis difficult."
                    
                    st.warning(explanation_text)
                    st.info("⚠️ Please upload a clear, hand-drawn spiral on white paper for accurate analysis.")
            
            # Display prediction
            with col2:
                st.subheader("Prediction Results")
                
                # Show prediction with confidence
                if result['prediction'] == "Parkinson":
                    st.error(f"⚠️ Prediction: {result['prediction']}")
                    st.error(f"Confidence: {result['confidence']:.1f}%")
                elif result['prediction'] == "Healthy":
                    st.success(f"✅ Prediction: {result['prediction']}")
                    st.success(f"Confidence: {result['confidence']:.1f}%")
                else:  # Noisy
                    st.warning(f"⚠️ Prediction: {result['prediction']}")
                    st.warning(f"Image quality issues detected")
                
                # Show all class probabilities
                st.write("\n**Class Probabilities:**")
                probs_df = pd.DataFrame({
                    'Class': label_encoder.classes_,
                    'Probability': result['probabilities']
                })
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(probs_df['Class'], probs_df['Probability'], color=['#2ecc71', '#e74c3c', '#f39c12'], width=0.6)
                ax.set_ylim(0, 1)  # Fixed scale from 0 to 1
                ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax.set_ylabel('Probability')
                ax.set_xlabel('Class')
                ax.set_title('Class Probabilities')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                plt.xticks(rotation=45)
                
                # Add percentage labels on top of bars
                for i, (bar, prob) in enumerate(zip(bars, probs_df['Probability'])):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{prob*100:.1f}%',
                           ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Explainability
            st.subheader("Understanding the Prediction")
            st.write("Here's what influenced the model's decision:")
            
            tab1, tab2 = st.tabs(["SHAP Explanation", "LIME Explanation"])
            
            with tab1:
                st.write("SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction:")
                
                # Generate SHAP values first
                with st.spinner("Generating SHAP explanation..."):
                    force_plot, shap_values = get_shap_explanation(model, result['X'], feature_names)
                    
                    # Get SHAP values for predicted class
                    prediction_idx = model.predict(result['X'])[0]
                    predicted_class_name = label_encoder.classes_[prediction_idx]
                    
                    if isinstance(shap_values, list):
                        shap_vec = shap_values[prediction_idx][0]
                    elif shap_values.ndim == 3:
                        shap_vec = shap_values[0, prediction_idx, :]
                    else:
                        shap_vec = shap_values[0]
                    
                    # Generate English explanation
                    st.write(f"\n**📝 SHAP Analysis: Why '{predicted_class_name}' Was Predicted**")
                    
                    # Separate positive and negative contributions
                    feature_shap_pairs = [(feature_names[i], shap_vec[i]) for i in range(len(shap_vec))]
                    positive_features = [(f, v) for f, v in feature_shap_pairs if v > 0]
                    negative_features = [(f, v) for f, v in feature_shap_pairs if v < 0]
                    
                    # Sort by absolute value
                    positive_features = sorted(positive_features, key=lambda x: abs(x[1]), reverse=True)[:5]
                    negative_features = sorted(negative_features, key=lambda x: abs(x[1]), reverse=True)[:3]
                    
                    if positive_features:
                        st.success("**✅ Features Pushing Toward This Prediction:**")
                        explanation_text = f"The model predicted **{predicted_class_name}** because these features strongly support it:\n\n"
                        for i, (feature, value) in enumerate(positive_features, 1):
                            explanation_text += f"{i}. **{feature}**: SHAP value = {value:+.5f}\n"
                        st.info(explanation_text)
                    
                    if negative_features:
                        st.warning("**⚠️ Features Pushing Away From This Prediction:**")
                        explanation_text = f"However, these features suggested it might NOT be **{predicted_class_name}**:\n\n"
                        for i, (feature, value) in enumerate(negative_features, 1):
                            explanation_text += f"{i}. **{feature}**: SHAP value = {value:+.5f}\n"
                        st.info(explanation_text)
                
                st.write("\n**📊 Visual Breakdown:**")
                # Create two columns for side-by-side charts
                shap_col1, shap_col2 = st.columns(2)
                
                with shap_col1:
                    st.write("**Main Feature Contributions:**")
                    st.pyplot(force_plot)
                
                with shap_col2:
                    st.write("**All Important Features:**")
                    
                    # Show all features that appear in the main SHAP chart
                    with st.spinner("Calculating importance..."):
                        prediction_idx = model.predict(result['X'])[0]
                        if isinstance(shap_values, list):
                            class_idx = prediction_idx
                            feature_importance = np.abs(shap_values[class_idx][0])
                        elif shap_values.ndim == 3:
                            # Multiclass: (n_samples, n_classes, n_features)
                            feature_importance = np.abs(shap_values[0, prediction_idx, :])
                        else:
                            feature_importance = np.abs(shap_values[0])
                        
                        # Get same features as shown in main chart
                        imp = feature_importance
                        top_idx = np.argsort(-imp)[:15]
                        threshold = np.max(imp) * 0.005
                        
                        # Filter and ensure minimum 5 features
                        important_features = []
                        important_values = []
                        for idx in top_idx:
                            if imp[idx] >= threshold or len(important_features) < 5:
                                important_features.append(feature_names[idx])
                                important_values.append(imp[idx])
                        
                        n_show = len(important_features)
                        
                        feature_imp_df = pd.DataFrame({
                            'Feature': important_features,
                            'Absolute Importance': important_values
                        })
                        
                        fig2, ax = plt.subplots(figsize=(8, 6))
                        sns.barplot(data=feature_imp_df, x='Absolute Importance', y='Feature', color='steelblue')
                        plt.title(f"Top {n_show} Feature (Importance)")
                        plt.xlabel('Absolute Importance (Higher = More Impact)')
                        plt.tight_layout()
                        st.pyplot(fig2)
            
            with tab2:
                st.write("LIME (Local Interpretable Model-agnostic Explanations) provides a local explanation for this specific prediction:")
                with st.spinner("Generating LIME explanation..."):
                    try:
                        lime_exp, pred_class = get_lime_explanation(model, result['X'], feature_names, label_encoder.classes_)
                        
                        # Get explanation for the predicted class
                        lime_list = lime_exp.as_list(label=pred_class)
                        
                        if lime_list:
                            # Generate English explanation
                            predicted_class_name = label_encoder.classes_[pred_class]
                            st.write(f"\n**📝 What Made the Model Predict '{predicted_class_name}'?**")
                            
                            # Separate positive and negative contributions
                            positive_features = [(f, w) for f, w in lime_list if w > 0]
                            negative_features = [(f, w) for f, w in lime_list if w < 0]
                            
                            if positive_features:
                                st.success("**✅ Features Supporting this Prediction:**")
                                explanation_text = f"The model predicted **{predicted_class_name}** mainly because:\n\n"
                                for i, (feature, weight) in enumerate(positive_features[:5], 1):
                                    # Clean feature name for display
                                    feature_clean = feature.split('<=')[0].strip() if '<=' in feature else feature.split('>')[0].strip()
                                    explanation_text += f"{i}. **{feature_clean}**: {feature} (impact: {weight:+.3f})\n"
                                st.info(explanation_text)
                            
                            if negative_features:
                                st.warning("**⚠️ Features Working Against this Prediction:**")
                                explanation_text = f"However, some features suggested it might NOT be **{predicted_class_name}**:\n\n"
                                for i, (feature, weight) in enumerate(negative_features[:3], 1):
                                    feature_clean = feature.split('<=')[0].strip() if '<=' in feature else feature.split('>')[0].strip()
                                    explanation_text += f"{i}. **{feature_clean}**: {feature} (impact: {weight:+.3f})\n"
                                st.info(explanation_text)
                            
                            st.write("\n**📊 Visual Representation:**")
                            # Create custom visualization
                            features = [item[0] for item in lime_list]
                            weights = [item[1] for item in lime_list]
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            colors = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights]  # Green for positive, Red for negative
                            y_pos = np.arange(len(features))
                            ax.barh(y_pos, weights, color=colors)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(features)
                            ax.set_xlabel('Feature Weight (Impact on Prediction)')
                            ax.set_title(f'LIME Explanation for {predicted_class_name} Prediction')
                            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                            
                            # Add legend
                            from matplotlib.patches import Patch
                            legend_elements = [Patch(facecolor='#2ecc71', label='Positive (Supports)'),
                                              Patch(facecolor='#e74c3c', label='Negative (Against)')]
                            ax.legend(handles=legend_elements, loc='lower right')
                            
                            plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                            
                            # Also show as a table
                            st.write("\n**📋 Detailed Feature Contributions:**")
                            lime_df = pd.DataFrame(lime_list, columns=['Feature Condition', 'Weight'])
                            st.dataframe(lime_df, use_container_width=True)
                        else:
                            st.warning("LIME could not generate explanations for this prediction.")
                    except Exception as e:
                        st.error(f"Error generating LIME explanation: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Feature details
            with st.expander("View Extracted Features"):
                st.write("**Handcrafted Features:**")
                st.write("These are human-interpretable measurements from the spiral image:")
                handcrafted_df = pd.DataFrame([result['features']['handcrafted_features']])
                st.dataframe(handcrafted_df, use_container_width=True)
                
                st.write("\n**CNN Features (Deep Learning Features):**")
                st.write("First 10 of 512 features automatically extracted by ResNet18 neural network:")
                cnn_features = result['features']['cnn_features'][:10]
                
                # Create a proper dataframe with feature names
                cnn_df = pd.DataFrame({
                    'cnn_feat_0': [cnn_features[0]],
                    'cnn_feat_1': [cnn_features[1]],
                    'cnn_feat_2': [cnn_features[2]],
                    'cnn_feat_3': [cnn_features[3]],
                    'cnn_feat_4': [cnn_features[4]],
                    'cnn_feat_5': [cnn_features[5]],
                    'cnn_feat_6': [cnn_features[6]],
                    'cnn_feat_7': [cnn_features[7]],
                    'cnn_feat_8': [cnn_features[8]],
                    'cnn_feat_9': [cnn_features[9]]
                })
                st.dataframe(cnn_df, use_container_width=True)
                
                # Show visual chart with proper labels
                st.write("\n**CNN Features Visualization:**")
                fig, ax = plt.subplots(figsize=(10, 4))
                feature_names = [f'cnn_feat_{i}' for i in range(10)]
                ax.bar(feature_names, cnn_features, color='steelblue')
                ax.set_xlabel('Feature Name')
                ax.set_ylabel('Feature Value')
                ax.set_title('First 10 CNN Features')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
    
    else:  # Batch Upload Mode
        st.subheader("📁 Batch Upload")
        
        # Initialize session state
        if 'batch_files' not in st.session_state:
            st.session_state.batch_files = []
        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = 0
        
        uploaded_files = st.file_uploader(
            "Upload multiple spiral drawing images...", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"batch_uploader_{st.session_state.uploader_key}"
        )
        
        # Update session state with new uploads
        if uploaded_files:
            st.session_state.batch_files = uploaded_files
        
        if st.session_state.batch_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📊 **{len(st.session_state.batch_files)} images** uploaded. Click 'Analyze All' to start processing.")
            with col2:
                if st.button("🗑️ Clear All", type="secondary"):
                    st.session_state.batch_files = []
                    st.session_state.batch_results = None
                    st.session_state.failed_images = []
                    st.session_state.uploader_key += 1  # Change key to reset uploader
                    st.rerun()
            
            # Initialize batch results in session state
            if 'batch_results' not in st.session_state:
                st.session_state.batch_results = None
            if 'failed_images' not in st.session_state:
                st.session_state.failed_images = []
            
            if st.button("🚀 Analyze All Images", type="primary"):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                failed_images = []
                
                # Process each image
                for idx, uploaded_file in enumerate(st.session_state.batch_files):
                    status_text.text(f"Processing {idx + 1}/{len(st.session_state.batch_files)}: {uploaded_file.name}")
                    progress_bar.progress((idx + 1) / len(st.session_state.batch_files))
                    
                    result = process_single_image(uploaded_file, model, scaler, imputer, 
                                                 label_encoder, feature_names, feature_extractor)
                    
                    if result:
                        # Store image bytes for later display
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        results.append({
                            'Image': uploaded_file.name,
                            'Prediction': result['prediction'],
                            'Confidence': f"{result['confidence']:.1f}%",
                            'Healthy_Prob': f"{result['probabilities'][0]*100:.1f}%",
                            'Noisy_Prob': f"{result['probabilities'][1]*100:.1f}%",
                            'Parkinson_Prob': f"{result['probabilities'][2]*100:.1f}%",
                            'result_obj': result,
                            'image_bytes': image_bytes
                        })
                    else:
                        failed_images.append(uploaded_file.name)
                
                status_text.text("✅ Processing complete!")
                progress_bar.progress(1.0)
                
                # Store results in session state
                st.session_state.batch_results = results
                st.session_state.failed_images = failed_images
            
            # Display results if they exist in session state
            if st.session_state.batch_results:
                results = st.session_state.batch_results
                failed_images = st.session_state.failed_images
                
                # Display results summary
                st.success(f"**Analysis Complete!** Processed {len(results)} images successfully.")
                
                if failed_images:
                    st.warning(f"⚠️ Failed to process {len(failed_images)} images: {', '.join(failed_images)}")
                
                # Summary statistics
                st.subheader("📊 Batch Analysis Summary")
                
                col1, col2, col3 = st.columns(3)
                
                predictions = [r['Prediction'] for r in results]
                with col1:
                    healthy_count = predictions.count('Healthy')
                    st.metric("Healthy", healthy_count, delta=f"{healthy_count/len(results)*100:.1f}%")
                
                with col2:
                    parkinson_count = predictions.count('Parkinson')
                    st.metric("Parkinson", parkinson_count, delta=f"{parkinson_count/len(results)*100:.1f}%")
                
                with col3:
                    noisy_count = predictions.count('Noisy')
                    st.metric("Noisy/Unclear", noisy_count, delta=f"{noisy_count/len(results)*100:.1f}%")
                
                # Results table
                st.subheader("📋 Detailed Results")
                results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['result_obj', 'image_bytes']} for r in results])
                st.dataframe(results_df, use_container_width=True)
                
                # Download results as CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="parkinson_detection_results.csv",
                    mime="text/csv"
                )
                
                # Visualization
                st.subheader("📈 Results Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction distribution pie chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    prediction_counts = pd.Series(predictions).value_counts()
                    colors = ['#28a745' if x == 'Healthy' else '#dc3545' if x == 'Parkinson' else '#ffc107' 
                             for x in prediction_counts.index]
                    ax.pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%',
                          colors=colors, startangle=90)
                    ax.set_title('Prediction Distribution')
                    st.pyplot(fig)
                
                with col2:
                    # Confidence distribution
                    confidences = [float(r['Confidence'].strip('%')) for r in results]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(confidences, bins=20, color='steelblue', edgecolor='black')
                    ax.set_xlabel('Confidence (%)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Confidence Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Individual image viewer
                st.subheader("🔍 View Individual Results")
                
                # Use index-based selection for reliability
                selected_idx = st.selectbox(
                    "Select an image to view details:",
                    options=list(range(len(results))),
                    format_func=lambda i: f"{results[i]['Image']} - {results[i]['Prediction']}"
                )
                
                # Always display if we have a selection
                selected_result = results[selected_idx]
                
                # Display with unique keys for each element
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display the image from stored bytes
                    from io import BytesIO
                    img = Image.open(BytesIO(selected_result['image_bytes']))
                    st.image(img, caption=selected_result['Image'], width=400)
                    
                    # Add "Why this prediction?" section under the image
                    st.markdown("---")
                    st.subheader("🔍 Why this prediction?")
                    
                    result_obj = selected_result['result_obj']
                    handcrafted = result_obj['features']['handcrafted_features']
                    
                    if selected_result['Prediction'] == "Parkinson":
                        # Analyze actual feature values for THIS specific image
                        jerkiness = handcrafted.get('Jerkiness', 0)
                        wobble = handcrafted.get('Wobble', 0)
                        curvature_std = handcrafted.get('Curv_Std', 0)
                        spiral_r2 = handcrafted.get('Spiral_R2', 1.0)
                        stroke_width_std = handcrafted.get('Std_Thickness', 0)
                        loop_var = handcrafted.get('Loop_Variation', 0)
                        path_rmse = handcrafted.get('Path_RMSE', 0)
                        
                        # Build specific reasons based on THIS image's features
                        reasons = []
                        
                        # Shakiness analysis
                        if jerkiness > 0.015:
                            reasons.append("**extremely shaky and jerky** lines")
                        elif jerkiness > 0.010:
                            reasons.append("**very shaky** drawing")
                        elif jerkiness > 0.006:
                            reasons.append("**noticeable shakiness**")
                        elif jerkiness > 0.003:
                            reasons.append("**slight shakiness**")
                        
                        # Wobble analysis
                        if wobble > 0.20:
                            reasons.append("**severe wobbling** throughout")
                        elif wobble > 0.12:
                            reasons.append("**excessive wobbling**")
                        elif wobble > 0.08:
                            reasons.append("**noticeable wobbling**")
                        elif wobble > 0.04:
                            reasons.append("**some wobbling**")
                        
                        # Loop variation analysis
                        if loop_var > 50:
                            reasons.append("**very inconsistent loop sizes**")
                        elif loop_var > 30:
                            reasons.append("**irregular loop patterns**")
                        elif loop_var > 20:
                            reasons.append("**uneven loops**")
                        
                        # Curvature analysis
                        if curvature_std > 0.0008:
                            reasons.append("**highly irregular curves**")
                        elif curvature_std > 0.0005:
                            reasons.append("**irregular curves**")
                        elif curvature_std > 0.0003:
                            reasons.append("**somewhat irregular curves**")
                        
                        # Spiral shape analysis
                        if spiral_r2 < 0.3:
                            reasons.append("**very poorly formed spiral**")
                        elif spiral_r2 < 0.5:
                            reasons.append("**poorly formed spiral**")
                        elif spiral_r2 < 0.7:
                            reasons.append("**uneven spiral shape**")
                        
                        # Line width analysis
                        if stroke_width_std > 3.0:
                            reasons.append("**extremely unsteady line thickness**")
                        elif stroke_width_std > 2.0:
                            reasons.append("**very unsteady lines**")
                        elif stroke_width_std > 1.2:
                            reasons.append("**unsteady line thickness**")
                        
                        # Path error analysis
                        if path_rmse > 30:
                            reasons.append("**very erratic path**")
                        elif path_rmse > 20:
                            reasons.append("**erratic drawing path**")
                        
                        # If no specific reasons found, use general ones
                        if not reasons:
                            reasons.append("**subtle tremor patterns**")
                            reasons.append("**irregular movement**")
                        
                        # Build natural explanation
                        explanation_text = "This image shows **Parkinson's disease** because the drawing has "
                        if len(reasons) == 1:
                            explanation_text += reasons[0]
                        elif len(reasons) == 2:
                            explanation_text += f"{reasons[0]} and {reasons[1]}"
                        else:
                            explanation_text += ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
                        explanation_text += "."
                        
                        st.warning(explanation_text)
                        st.info("💡 These signs indicate tremors and difficulty with fine motor control, typical of Parkinson's patients.")
                    
                    elif selected_result['Prediction'] == "Healthy":
                        # Analyze actual feature values for THIS specific image
                        jerkiness = handcrafted.get('Jerkiness', 1)
                        wobble = handcrafted.get('Wobble', 1)
                        spiral_r2 = handcrafted.get('Spiral_R2', 0)
                        curvature_std = handcrafted.get('Curv_Std', 1)
                        stroke_width_std = handcrafted.get('Std_Thickness', 10)
                        loop_var = handcrafted.get('Loop_Variation', 100)
                        circularity = handcrafted.get('Circularity', 0)
                        
                        # Build specific reasons based on THIS image's features
                        reasons = []
                        
                        # Smoothness analysis
                        if jerkiness < 0.002:
                            reasons.append("**exceptionally smooth and steady**")
                        elif jerkiness < 0.004:
                            reasons.append("**very smooth with minimal shakiness**")
                        elif jerkiness < 0.006:
                            reasons.append("**smooth drawing**")
                        elif jerkiness < 0.008:
                            reasons.append("**reasonably smooth**")
                        
                        # Steadiness analysis
                        if wobble < 0.03:
                            reasons.append("**extremely steady with no wobbling**")
                        elif wobble < 0.06:
                            reasons.append("**very steady**")
                        elif wobble < 0.10:
                            reasons.append("**steady hand movement**")
                        elif wobble < 0.15:
                            reasons.append("**mostly steady**")
                        
                        # Loop consistency analysis
                        if loop_var < 12:
                            reasons.append("**highly consistent loop sizes**")
                        elif loop_var < 18:
                            reasons.append("**consistent loops**")
                        elif loop_var < 25:
                            reasons.append("**fairly consistent loops**")
                        
                        # Spiral shape analysis
                        if spiral_r2 > 0.85:
                            reasons.append("**perfectly formed spiral**")
                        elif spiral_r2 > 0.75:
                            reasons.append("**well-formed spiral shape**")
                        elif spiral_r2 > 0.65:
                            reasons.append("**good spiral shape**")
                        
                        # Curve consistency analysis
                        if curvature_std < 0.0002:
                            reasons.append("**highly uniform curves**")
                        elif curvature_std < 0.0004:
                            reasons.append("**consistent curves**")
                        elif curvature_std < 0.0006:
                            reasons.append("**fairly consistent curves**")
                        
                        # Line quality analysis
                        if stroke_width_std < 0.8:
                            reasons.append("**uniform line thickness**")
                        elif stroke_width_std < 1.2:
                            reasons.append("**stable line quality**")
                        elif stroke_width_std < 1.8:
                            reasons.append("**decent line control**")
                        
                        # Circularity analysis
                        if circularity > 0.8:
                            reasons.append("**excellent circular form**")
                        elif circularity > 0.7:
                            reasons.append("**good circular form**")
                        
                        # If no specific reasons found
                        if not reasons:
                            reasons.append("**good control**")
                            reasons.append("**normal drawing patterns**")
                        
                        # Build natural explanation
                        explanation_text = "This image shows a **healthy** drawing because it is "
                        if len(reasons) == 1:
                            explanation_text += reasons[0]
                        elif len(reasons) == 2:
                            explanation_text += f"{reasons[0]} and {reasons[1]}"
                        else:
                            explanation_text += ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
                        explanation_text += "."
                        
                        st.success(explanation_text)
                        st.info("✅ These characteristics indicate good motor control and coordination.")
                    
                    else:  # Noisy
                        # Analyze actual feature values for THIS specific image
                        entropy = handcrafted.get('Entropy', 0)
                        stroke_length = handcrafted.get('Stroke_Length', 0)
                        skel_length = handcrafted.get('Skel_Length', 0)
                        stroke_width_std = handcrafted.get('Std_Thickness', 0)
                        mean_thickness = handcrafted.get('Mean_Thickness', 0)
                        
                        # Build specific issues based on THIS image's features
                        issues = []
                        
                        if entropy < 2.5:
                            issues.append("**very poor image quality**")
                        elif entropy < 3.5:
                            issues.append("**poor image quality**")
                        elif entropy < 4.5:
                            issues.append("**low image quality**")
                        
                        if stroke_length < 500:
                            issues.append("**incomplete or faint drawing**")
                        elif stroke_length < 1000:
                            issues.append("**weak or unclear lines**")
                        
                        if skel_length < 300:
                            issues.append("**very sparse line detection**")
                        elif skel_length < 600:
                            issues.append("**unclear line structure**")
                        
                        if mean_thickness < 1.5:
                            issues.append("**extremely thin or faint lines**")
                        elif mean_thickness < 2.5:
                            issues.append("**very faint lines**")
                        
                        if stroke_width_std > 4.0:
                            issues.append("**extremely inconsistent image quality**")
                        elif stroke_width_std > 3.0:
                            issues.append("**very inconsistent quality**")
                        elif stroke_width_std > 2.0:
                            issues.append("**inconsistent quality**")
                        
                        if not issues:
                            issues.append("**quality problems**")
                        
                        # Build natural explanation
                        explanation_text = "This image has "
                        if len(issues) == 1:
                            explanation_text += issues[0]
                        elif len(issues) == 2:
                            explanation_text += f"{issues[0]} and {issues[1]}"
                        else:
                            explanation_text += ", ".join(issues[:-1]) + f", and {issues[-1]}"
                        explanation_text += ", making accurate analysis difficult."
                        
                        st.warning(explanation_text)
                        st.info("⚠️ Please upload a clear, hand-drawn spiral on white paper for accurate analysis.")
                
                with col2:
                    st.write("")  # Add spacing
                    st.metric("Prediction", selected_result['Prediction'])
                    st.metric("Confidence", selected_result['Confidence'])
                    st.write("---")
                    st.write("**All Probabilities:**")
                    
                    # Display probabilities with progress bars
                    st.write(f"🟢 Healthy: {selected_result['Healthy_Prob']}")
                    st.progress(float(selected_result['Healthy_Prob'].strip('%'))/100)
                    
                    st.write(f"🟡 Noisy: {selected_result['Noisy_Prob']}")
                    st.progress(float(selected_result['Noisy_Prob'].strip('%'))/100)
                    
                    st.write(f"🔴 Parkinson: {selected_result['Parkinson_Prob']}")
                    st.progress(float(selected_result['Parkinson_Prob'].strip('%'))/100)
    
    # Disclaimer at the bottom of the page
    st.markdown("---")
    st.markdown("#### ⚠️ Disclaimer")
    st.warning("""
    
    This spiral-drawing–based Parkinson’s screening system is not a medical or diagnostic tool and should not be considered 100% accurate. The output represents only a preliminary indication based on subtle motor-pattern analysis and may produce false positives or false negatives. Results can vary due to factors such as drawing conditions, device quality, user fatigue, or data variability. This system is intended solely for research, educational, and clinical-support purposes. It does not replace professional medical judgment. Any indication of Parkinson’s generated by this system must be followed by a full clinical evaluation by a qualified neurologist and appropriate diagnostic tests.
    
    """)

if __name__ == "__main__":
    main()