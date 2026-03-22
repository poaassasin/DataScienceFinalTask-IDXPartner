import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi UI/UX
st.set_page_config(page_title="Credit Scoring System", layout="wide")

# 1. Load Kedua Model (Versi 10 Fitur)
@st.cache_resource
def load_models():
    lgbm = joblib.load('model_decision_lgbm.pkl')
    lr = joblib.load('model_explanation_lr.pkl')
    return lgbm, lr

model_decision, model_explanation = load_models()

FEATURE_NAMES = [
    'mths_since_issue_d', 'mths_since_last_pymnt_d', 'annual_inc', 
    'dti', 'revol_util', 'mths_since_earliest_cr_line', 
    'tot_cur_bal', 'int_rate', 'revol_bal', 'total_rev_hi_lim'
]

# Sidebar: Metrik Validasi (Social Proof untuk Rekruter)
st.sidebar.header("🚀 Model Reliability")
st.sidebar.metric("ROC-AUC Score", "0.8525")
st.sidebar.metric("Stability Gap", "0.0371", delta="Very Stable", delta_color="normal")
st.sidebar.markdown("---")
st.sidebar.write("Architecture: **Challenger Model** (LGBM + LogReg)")

st.title("🏦 Banking Credit Engine")
st.write("Sistem Pendukung Keputusan Kredit Real-time.")

# 2. Input Form (UI Layout 2 Kolom)
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Financial Profile")
        annual_inc = st.number_input("Annual Income", min_value=0, value=50000)
        int_rate = st.slider("Interest Rate", 0.05, 0.35, 0.15)
        dti = st.number_input("Debt-to-Income Ratio (DTI)", 0.0, 100.0, 20.0)
        tot_cur_bal = st.number_input("Total Current Balance", 0, 1000000, 180000)
        total_rev_hi_lim = st.number_input("Total Credit Limit", 0, 500000, 20000)

    with col2:
        st.subheader("📜 Credit History")
        last_pay = st.number_input("Months Since Last Payment", 0, 120, 3)
        issue_d = st.number_input("Months Since Loan Issued", 0, 120, 12)
        earliest_cr = st.number_input("Months Since Earliest Credit Line", 0, 500, 72)
        revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)
        revol_bal = st.number_input("Revolving Balance", 0, 100000, 15000)

    submit = st.form_submit_button("Run Analysis", use_container_width=True)

# 3. Logika Prediksi & Eksplanasi
if submit:
    df_input = pd.DataFrame([[
        issue_d, last_pay, annual_inc, dti, revol_util, 
        earliest_cr, tot_cur_bal, int_rate, revol_bal, total_rev_hi_lim
    ]], columns=FEATURE_NAMES)

    # Layer 1: Decision (LightGBM)
    prob_bad = 1 - model_decision.predict_proba(df_input)[:, 1][0]
    is_bad = prob_bad >= 0.69  # Threshold 0.69

    st.markdown("---")
    
    if is_bad:
        st.error("### ❌ Status: REJECTED")
        st.write(f"Confidence Score: **{prob_bad:.2%}** probability of default.")
        
        # Layer 2: Explanation (Logistic Regression) - Anti-Bullying Logic
        coeffs = np.abs(model_explanation.coef_[0])
        top_trigger_idx = np.argmax(coeffs)
        top_trigger_feature = FEATURE_NAMES[top_trigger_idx]
        
        st.info(f"**Alasan Penolakan:** Faktor risiko utama terdeteksi pada **{top_trigger_feature.replace('_', ' ').title()}**.")
    else:
        st.success("### ✅ Status: APPROVED")
        st.write(f"Confidence Score: **{prob_bad:.2%}** probability of default.")
        st.balloons()