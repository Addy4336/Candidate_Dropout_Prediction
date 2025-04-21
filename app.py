import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("logistic_regression_dropout_model.pkl")

# Feature means and stds used during training (for scaling)
scaling_info = {
    'experience': (1.93594, 1.893535),
    'test_score': (0.71364, 0.145451),
    'form_time': (10.15864, 7.140687),
    'email_response_delay': (4.82507, 4.934872),
    'final_ctc_expectation': (14.65482, 9.177816)
}

scale_cols = list(scaling_info.keys())
all_features = scale_cols + [
    'follow_ups', 'is_referred', 'timezone_diff', 'past_startups'
]

# Helper to scale values
def scale(value, mean, std):
    return (value - mean) / std

def scale_df(df):
    for col in scale_cols:
        df[col] = (df[col] - scaling_info[col][0]) / scaling_info[col][1]
    return df

# Page config
st.set_page_config(page_title="Dropout Predictor", layout="wide")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch CSV Upload", "📊 Dashboard"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.title("🎯 Candidate Dropout Predictor")
    st.markdown("Predict whether a candidate will drop out before the final interview.")

    st.sidebar.header("📝 Candidate Info")

    # Real-world input
    experience = st.sidebar.slider("Experience (years)", 0.0, 10.0, 2.0)
    test_score = st.sidebar.slider("Test Score", 0.2, 1.0, 0.7)
    form_time = st.sidebar.slider("Form Time (minutes)", 1, 60, 10)
    email_delay = st.sidebar.slider("Email Response Delay (hours)", 0, 48, 5)
    ctc_expect = st.sidebar.slider("CTC Expectation (LPA)", 3, 50, 12)
    follow_ups = st.sidebar.selectbox("Follow-ups", [0, 1, 2, 3])
    is_referred = st.sidebar.radio("Referred?", ["No", "Yes"])
    timezone_diff = st.sidebar.slider("Timezone Difference (hours)", 0, 12, 0)
    past_startups = st.sidebar.slider("Past Startups", 0, 10, 1)

    if st.button("🔍 Predict"):
        # Scale inputs
        data = {
            'experience': scale(experience, *scaling_info['experience']),
            'test_score': scale(test_score, *scaling_info['test_score']),
            'form_time': scale(form_time, *scaling_info['form_time']),
            'email_response_delay': scale(email_delay, *scaling_info['email_response_delay']),
            'final_ctc_expectation': scale(ctc_expect, *scaling_info['final_ctc_expectation']),
            'follow_ups': follow_ups,
            'is_referred': 1 if is_referred == "Yes" else 0,
            'timezone_diff': timezone_diff,
            'past_startups': past_startups
        }
        input_df = pd.DataFrame([data])
        prob = model.predict_proba(input_df.to_numpy())[0][1]

        st.subheader("📊 Prediction")
        st.metric("Dropout Probability", f"{prob:.2%}")
        if prob >= 0.5:
            st.error("🚨 High Risk — Recommend Immediate Action")
        elif prob >= 0.2:
            st.warning("⚠️ Moderate Risk — Monitor Closely")
        else:
            st.success("✅ Low Risk — Likely to Show Up")

# --- Tab 2: Batch Prediction ---
with tab2:
    st.title("📁 Batch Prediction via CSV")
    st.markdown("Upload a CSV with multiple candidates to predict dropout risk.")

    st.markdown("### Required Columns:")
    st.code(", ".join(all_features))

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        try:
            df_scaled = scale_df(df_input.copy())
            probs = model.predict_proba(df_scaled.to_numpy())[:, 1]
            df_input["Dropout_Probability"] = probs
            df_input["Risk_Level"] = pd.cut(probs, bins=[0, 0.2, 0.5, 1],
                                            labels=["Low", "Moderate", "High"])
            st.success("✅ Prediction Complete")
            st.dataframe(df_input)

            csv = df_input.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", data=csv, file_name="dropout_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Tab 3: Dashboard ---
with tab3:
    st.title("📊 Model Dashboard")

    # ROC and Coefficients
    st.subheader("📈 Feature Importance (Odds Ratio)")
    coef = model.coef_[0]
    features = all_features
    odds = np.exp(coef)
    importance_df = pd.DataFrame({
        "Feature": features,
        "Odds Ratio": odds
    }).sort_values(by="Odds Ratio", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df["Feature"], importance_df["Odds Ratio"], color="#007acc")
    ax.set_title("Feature Impact (Odds Ratios)")
    ax.axvline(1.0, color="gray", linestyle="--")
    st.pyplot(fig)

    st.markdown("🔹 Odds Ratio > 1: Increases chance of dropout  \n🔹 Odds Ratio < 1: Decreases dropout likelihood")
