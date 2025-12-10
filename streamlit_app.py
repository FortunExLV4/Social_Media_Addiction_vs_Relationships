import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODELS & ARTIFACTS
# ============================================================
@st.cache_resource
def load_models():
    dt_model = joblib.load('models/decision_tree_model.pkl')
    nb_model = joblib.load('models/naive_bayes_model.pkl')
    nn_model = joblib.load('models/neural_network_model.pkl')
    lr_model = joblib.load('models/linear_regression_model.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    
    scaler_class = joblib.load('models/scaler_class.pkl')
    scaler_reg = joblib.load('models/scaler_reg.pkl')
    scaler_kmeans = joblib.load('models/scaler_kmeans.pkl')
    
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_info = joblib.load('models/feature_info.pkl')
    
    return {
        'models': {
            'Decision Tree': dt_model,
            'Naive Bayes': nb_model,
            'Neural Network': nn_model,
            'Linear Regression': lr_model,
            'K-Means': kmeans_model
        },
        'scalers': {
            'class': scaler_class,
            'reg': scaler_reg,
            'kmeans': scaler_kmeans
        },
        'label_encoders': label_encoders,
        'feature_info': feature_info
    }

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def create_engineered_features(input_data):
    """Create the engineered features from raw input."""
    # Sleep Deficit
    input_data['Sleep_Deficit'] = max(0, 8 - input_data['Sleep_Hours_Per_Night'])
    
    # Usage-Sleep Ratio
    input_data['Usage_Sleep_Ratio'] = input_data['Avg_Daily_Usage_Hours'] / (input_data['Sleep_Hours_Per_Night'] + 0.1)
    
    # Relationship Strain
    relationship_weights = {'Single': 1.0, 'In Relationship': 1.5, 'Complicated': 2.0}
    input_data['Relationship_Strain'] = (
        input_data['Conflicts_Over_Social_Media'] * 
        relationship_weights.get(input_data['Relationship_Status'], 1.0)
    )
    
    # Addiction Risk Score (simplified version for interface)
    # Normalize each component to 0-1 range using expected ranges
    usage_norm = min(1, input_data['Avg_Daily_Usage_Hours'] / 10)
    sleep_deficit_norm = min(1, input_data['Sleep_Deficit'] / 4)
    conflict_norm = min(1, input_data['Conflicts_Over_Social_Media'] / 10)
    mental_health_norm = 1 - (input_data['Mental_Health_Score'] / 10)
    
    input_data['Addiction_Risk_Score'] = (
        0.35 * usage_norm +
        0.25 * sleep_deficit_norm +
        0.20 * conflict_norm +
        0.20 * mental_health_norm
    )
    
    return input_data

def prepare_classification_input(input_data, feature_info, scaler):
    """Prepare input for classification models (one-hot encoded)."""
    # Create DataFrame with all expected columns
    expected_cols = feature_info['X_class_columns']
    df_input = pd.DataFrame(0, index=[0], columns=expected_cols)
    
    # Fill numeric columns
    numeric_cols = feature_info['numeric_cols']
    for col in numeric_cols:
        if col in input_data:
            df_input[col] = input_data[col]
    
    # Fill one-hot encoded categorical columns
    categorical_mappings = {
        'Gender': input_data.get('Gender', 'Male'),
        'Academic_Level': input_data.get('Academic_Level', 'Undergraduate'),
        'Country': input_data.get('Country', 'USA'),
        'Most_Used_Platform': input_data.get('Most_Used_Platform', 'Instagram'),
        'Relationship_Status': input_data.get('Relationship_Status', 'Single'),
        'Region': input_data.get('Region', 'Northern America')
    }
    
    for cat_col, value in categorical_mappings.items():
        col_name = f"{cat_col}_{value}"
        if col_name in df_input.columns:
            df_input[col_name] = 1
    
    # Scale numeric features
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    
    return df_input

def prepare_regression_input(input_data, feature_info, scaler):
    """Prepare input for regression model."""
    expected_cols = feature_info['X_reg_columns']
    df_input = pd.DataFrame(0, index=[0], columns=expected_cols)
    
    numeric_cols = feature_info['numeric_cols']
    for col in numeric_cols:
        if col in input_data:
            df_input[col] = input_data[col]
    
    categorical_mappings = {
        'Gender': input_data.get('Gender', 'Male'),
        'Academic_Level': input_data.get('Academic_Level', 'Undergraduate'),
        'Country': input_data.get('Country', 'USA'),
        'Most_Used_Platform': input_data.get('Most_Used_Platform', 'Instagram'),
        'Relationship_Status': input_data.get('Relationship_Status', 'Single'),
        'Region': input_data.get('Region', 'Northern America'),
        'Affects_Academic_Performance': input_data.get('Affects_Academic_Performance', 'No')
    }
    
    for cat_col, value in categorical_mappings.items():
        col_name = f"{cat_col}_{value}"
        if col_name in df_input.columns:
            df_input[col_name] = 1
    
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    
    return df_input

def prepare_kmeans_input(input_data, feature_info, scaler):
    """Prepare input for K-Means clustering."""
    kmeans_cols = feature_info['kmeans_columns']
    df_input = pd.DataFrame([input_data])[kmeans_cols]
    df_input_scaled = pd.DataFrame(
        scaler.transform(df_input),
        columns=kmeans_cols
    )
    return df_input_scaled

def get_cluster_profile(cluster_id):
    """Return a description of each cluster."""
    profiles = {
        0: ("🟢 Low Risk", "Healthy balance between social media use and life. Good sleep, low addiction indicators."),
        1: ("🟡 Moderate Risk", "Some signs of overuse. May need to monitor habits and set boundaries."),
        2: ("🔴 High Risk", "High usage, poor sleep, elevated addiction markers. Consider intervention."),
        3: ("🟠 At-Risk", "Borderline patterns. Early intervention recommended.")
    }
    return profiles.get(cluster_id, ("❓ Unknown", "Cluster profile not defined."))

def create_gauge_chart(value, title, max_val=10):
    """Create a gauge chart for visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val*0.33], 'color': "lightgreen"},
                {'range': [max_val*0.33, max_val*0.66], 'color': "yellow"},
                {'range': [max_val*0.66, max_val], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_risk_radar(input_data):
    """Create a radar chart showing risk factors."""
    categories = ['Usage', 'Sleep Deficit', 'Conflicts', 'Mental Health (inv)', 'Relationship Strain']
    
    # Normalize values to 0-10 scale
    values = [
        min(10, input_data['Avg_Daily_Usage_Hours']),
        min(10, input_data['Sleep_Deficit'] * 2.5),
        min(10, input_data['Conflicts_Over_Social_Media']),
        10 - input_data['Mental_Health_Score'],
        min(10, input_data['Relationship_Strain'])
    ]
    values.append(values[0])  # Close the radar
    categories.append(categories[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    return fig

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Load models
    try:
        artifacts = load_models()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.info("Please run the training script first to generate model files.")
        return
    
    models = artifacts['models']
    scalers = artifacts['scalers']
    feature_info = artifacts['feature_info']
    
    # ========== HEADER ==========
    st.title("📱 Social Media Addiction Predictor")
    st.markdown("""
    This application uses **Machine Learning** to predict:
    - Whether social media affects your **academic performance**
    - Your estimated **addiction score**
    - Your **risk profile** based on usage patterns
    
    ---
    """)
    
    # ========== SIDEBAR: INPUT FORM ==========
    st.sidebar.header("📝 Enter Your Information")
    
    # Personal Info
    st.sidebar.subheader("Personal Info")
    age = st.sidebar.slider("Age", 14, 35, 20)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])
    academic_level = st.sidebar.selectbox(
        "Academic Level",
        ["High School", "Undergraduate", "Postgraduate"]
    )
    
    country = st.sidebar.selectbox(
        "Country",
        ["USA", "UK", "India", "Bangladesh", "Germany", "France", 
         "Australia", "Canada", "Pakistan", "Brazil", "Japan", "Other"]
    )
    
    # Map country to region
    country_to_region = {
        "USA": "Northern America",
        "UK": "Northern Europe",
        "India": "Southern Asia",
        "Bangladesh": "Southern Asia",
        "Germany": "Western Europe",
        "France": "Western Europe",
        "Australia": "Australia and New Zealand",
        "Canada": "Northern America",
        "Pakistan": "Southern Asia",
        "Brazil": "Latin America and the Caribbean",
        "Japan": "Eastern Asia",
        "Other": "Other"
    }
    region = country_to_region.get(country, "Other")
    
    # Social Media Usage
    st.sidebar.subheader("📲 Social Media Usage")
    avg_usage = st.sidebar.slider(
        "Average Daily Usage (hours)",
        0.0, 12.0, 3.0, 0.5
    )
    platform = st.sidebar.selectbox(
        "Most Used Platform",
        ["Instagram", "TikTok", "YouTube", "Facebook", "Twitter", 
         "Snapchat", "WhatsApp", "LinkedIn", "Reddit", "Other"]
    )
    
    # Health & Wellbeing
    st.sidebar.subheader("😴 Health & Wellbeing")
    sleep_hours = st.sidebar.slider(
        "Sleep Hours Per Night",
        3.0, 10.0, 7.0, 0.5
    )
    mental_health = st.sidebar.slider(
        "Mental Health Score (1-10)",
        1, 10, 7,
        help="1 = Very Poor, 10 = Excellent"
    )
    
    # Relationships
    st.sidebar.subheader("💑 Relationships")
    relationship_status = st.sidebar.selectbox(
        "Relationship Status",
        ["Single", "In Relationship", "Complicated"]
    )
    conflicts = st.sidebar.slider(
        "Conflicts Over Social Media (1-10)",
        1, 10, 3,
        help="How often does social media cause conflicts?"
    )
    
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)
    
    # ========== MAIN CONTENT ==========
    if predict_button:
        # Collect input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Academic_Level': academic_level,
            'Country': country,
            'Region': region,
            'Avg_Daily_Usage_Hours': avg_usage,
            'Most_Used_Platform': platform,
            'Sleep_Hours_Per_Night': sleep_hours,
            'Mental_Health_Score': mental_health,
            'Relationship_Status': relationship_status,
            'Conflicts_Over_Social_Media': conflicts,
            'Addicted_Score': 5  # Placeholder for K-Means
        }
        
        # Create engineered features
        input_data = create_engineered_features(input_data)
        
        # ========== PREDICTIONS ==========
        st.header("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        # --- Classification Predictions ---
        with col1:
            st.subheader("📚 Academic Impact")
            
            # Prepare input
            X_class = prepare_classification_input(
                input_data, feature_info, scalers['class']
            )
            
            # Get predictions from all classifiers
            predictions = {}
            probabilities = {}
            
            for name in ['Decision Tree', 'Naive Bayes', 'Neural Network']:
                model = models[name]
                pred = model.predict(X_class)[0]
                prob = model.predict_proba(X_class)[0][1]
                predictions[name] = pred
                probabilities[name] = prob
            
            # Ensemble: majority vote
            avg_prob = np.mean(list(probabilities.values()))
            ensemble_pred = 1 if avg_prob >= 0.5 else 0
            
            if ensemble_pred == 1:
                st.error("⚠️ **YES** - Social media likely affects your academics")
            else:
                st.success("✅ **NO** - Social media unlikely to affect your academics")
            
            st.metric("Risk Probability", f"{avg_prob*100:.1f}%")
            
            # Show individual model predictions
            with st.expander("Model Breakdown"):
                for name, prob in probabilities.items():
                    st.write(f"**{name}:** {prob*100:.1f}% risk")
        
        # --- Regression Prediction ---
        with col2:
            st.subheader("📊 Addiction Score")
            
            # For regression, we need to include Affects_Academic_Performance
            input_data['Affects_Academic_Performance'] = 'Yes' if ensemble_pred == 1 else 'No'
            
            X_reg = prepare_regression_input(
                input_data, feature_info, scalers['reg']
            )
            
            predicted_addiction = models['Linear Regression'].predict(X_reg)[0]
            predicted_addiction = np.clip(predicted_addiction, 1, 10)
            
            # Gauge chart
            fig_gauge = create_gauge_chart(
                predicted_addiction,
                "Predicted Addiction Score"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if predicted_addiction >= 7:
                st.error("High addiction indicators detected")
            elif predicted_addiction >= 4:
                st.warning("Moderate addiction indicators")
            else:
                st.success("Low addiction indicators")
        
        # --- Clustering Prediction ---
        with col3:
            st.subheader("👥 Risk Profile")
            
            # Update Addicted_Score with prediction for clustering
            input_data['Addicted_Score'] = predicted_addiction
            
            X_kmeans = prepare_kmeans_input(
                input_data, feature_info, scalers['kmeans']
            )
            
            cluster = models['K-Means'].predict(X_kmeans)[0]
            profile_name, profile_desc = get_cluster_profile(cluster)
            
            st.markdown(f"### {profile_name}")
            st.write(profile_desc)
            st.metric("Cluster", f"Group {cluster}")
        
        # ========== RISK ANALYSIS ==========
        st.markdown("---")
        st.header("📈 Risk Factor Analysis")
        
        col_radar, col_details = st.columns([1, 1])
        
        with col_radar:
            st.subheader("Risk Radar")
            fig_radar = create_risk_radar(input_data)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col_details:
            st.subheader("Key Indicators")
            
            # Display engineered features
            st.write("**Engineered Risk Metrics:**")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric(
                    "Sleep Deficit",
                    f"{input_data['Sleep_Deficit']:.1f} hrs",
                    delta=f"from 8hr ideal",
                    delta_color="inverse"
                )
                st.metric(
                    "Usage/Sleep Ratio",
                    f"{input_data['Usage_Sleep_Ratio']:.2f}",
                    help="Higher = more usage relative to sleep"
                )
            
            with metrics_col2:
                st.metric(
                    "Relationship Strain",
                    f"{input_data['Relationship_Strain']:.1f}",
                    help="Conflicts weighted by relationship status"
                )
                st.metric(
                    "Addiction Risk Score",
                    f"{input_data['Addiction_Risk_Score']:.2f}",
                    help="Composite score (0-1)"
                )
        
        # ========== RECOMMENDATIONS ==========
        st.markdown("---")
        st.header("💡 Recommendations")
        
        recommendations = []
        
        if avg_usage > 4:
            recommendations.append("📵 **Reduce screen time**: Consider setting daily limits on social media apps.")
        
        if sleep_hours < 7:
            recommendations.append("😴 **Improve sleep**: Aim for 7-9 hours. Avoid screens 1 hour before bed.")
        
        if conflicts > 5:
            recommendations.append("💬 **Address conflicts**: Social media is causing relationship stress. Consider open conversations about boundaries.")
        
        if mental_health < 5:
            recommendations.append("🧠 **Mental health support**: Consider speaking with a counselor or therapist.")
        
        if input_data['Usage_Sleep_Ratio'] > 0.7:
            recommendations.append("⚖️ **Rebalance priorities**: You're spending significant time scrolling relative to sleeping.")
        
        if predicted_addiction > 6:
            recommendations.append("🚨 **Digital detox**: Consider taking periodic breaks from social media (e.g., weekends off).")
        
        if not recommendations:
            recommendations.append("✨ **Keep it up!** Your social media habits appear healthy. Maintain your current balance.")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # ========== INPUT SUMMARY ==========
        st.markdown("---")
        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame({
                'Attribute': [
                    'Age', 'Gender', 'Academic Level', 'Country', 'Region',
                    'Daily Usage (hrs)', 'Platform', 'Sleep (hrs)',
                    'Mental Health', 'Relationship Status', 'Conflicts'
                ],
                'Value': [
                    age, gender, academic_level, country, region,
                    avg_usage, platform, sleep_hours,
                    mental_health, relationship_status, conflicts
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    else:
        # Default state before prediction
        st.info("👈 Fill in your information in the sidebar and click **Predict** to see results.")
        
        # Show sample visualization
        st.subheader("📊 About This Tool")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🌳 Decision Tree**
            
            Learns patterns by asking yes/no questions about your habits.
            """)
        
        with col2:
            st.markdown("""
            **📈 Naive Bayes**
            
            Uses probability theory to estimate your risk level.
            """)
        
        with col3:
            st.markdown("""
            **🧠 Neural Network**
            
            Mimics brain neurons to find complex patterns.
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### How It Works
        
        1. **Input your data** in the sidebar (age, usage habits, sleep, etc.)
        2. **Click Predict** to run multiple ML models
        3. **View results** including:
           - Whether social media affects your academics
           - Your predicted addiction score
           - Your risk profile cluster
           - Personalized recommendations
        
        ### Models Used
        
        | Model | Type | Purpose |
        |-------|------|---------|
        | Decision Tree | Classification | Predict academic impact |
        | Naive Bayes | Classification | Predict academic impact |
        | Neural Network | Classification | Predict academic impact |
        | Linear Regression | Regression | Predict addiction score |
        | K-Means | Clustering | Identify risk profile |
        """)

# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()