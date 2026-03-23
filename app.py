import streamlit as st
import pandas as pd
import numpy as np
import joblib

model_lgbm = joblib.load('model_decision_lgbm.pkl')
model_logreg = joblib.load('model_explanation_lr.pkl')

# 1. Konfigurasi Halaman & Sidebar
st.set_page_config(page_title="Banking Credit Engine", layout="wide")

with st.sidebar:
    st.markdown("### 🚀 Model Reliability")
    # Angka sesuai laporan terbaru Anda
    st.metric("ROC-AUC Score", "0.8523")
    st.write("**Stability Gap LGBM:** 0.0340")
    st.write("**Stability Gap LogReg:** 0.0002")
    st.markdown("---")
    st.info("Decision: **LightGBM**\n\nExplanation: **LogReg**")

# 2. Definisi 10 Fitur LightGBM (Sesuai Urutan Rank Produksi)
# Fitur inilah yang akan muncul di UI dan digunakan untuk prediksi
FEATURES_LGBM = [
    'mths_since_issue_d', 'mths_since_last_pymnt_d', 'mths_since_earliest_cr_line',
    'dti', 'annual_inc', 'revol_util', 'tot_cur_bal', 'revol_bal', 'installment', 'int_rate'
]

st.title("🏦 Banking Credit Engine")
st.caption("Sistem Keputusan Kredit Berbasis 10 Fitur Utama LightGBM.")

# 3. Form Input Pengguna (Hanya 10 Fitur Utama)
with st.form("credit_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Financial Profile")
        annual_inc = st.number_input("Annual Income", value=20000)
        int_rate = st.slider("Interest Rate (%)", 0.05, 0.30, 0.08)
        dti = st.number_input("Debt-to-Income Ratio (DTI)", value=5.0)
        tot_cur_bal = st.number_input("Total Current Balance", value=5000)
        installment = st.number_input("Monthly Installment", value=30)

    with col2:
        st.markdown("### 📜 Credit History")
        mths_issue = st.number_input("Months Since Loan Issued", value=120)
        mths_last_pay = st.number_input("Months Since Last Payment", value=1)
        mths_earliest_cr = st.number_input("Months Since Earliest Credit Line", value=72)
        revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 48.0)
        revol_bal = st.number_input("Revolving Balance", value=15000)

    submit = st.form_submit_button("Run Analysis")

# 4. Prediksi & Penjelasan (Layered Logic)
if submit:
    # Siapkan data dalam DataFrame
    input_values = [
        mths_issue, mths_last_pay, mths_earliest_cr, dti, 
        annual_inc, revol_util, tot_cur_bal, revol_bal, installment, int_rate
    ]

    df_input = pd.DataFrame([input_values], columns=FEATURES_LGBM)

    prob_bad = model_lgbm.predict_proba(df_input)[:, 1][0]

    is_rejected = prob_bad >= 0.66

    st.markdown("---")
    if is_rejected:
        st.error("### ❌ Status: REJECTED")
        st.write(f"Confidence Score: **{prob_bad:.2%}** probability of default.")

        # --- LAYER 2: PENJELASAN (LOGISTIC REGRESSION) ---
        # Penjelasan dari LogReg tetap pakai logika 'Local Impact'
        coeffs = model_logreg.coef_[0]
    
        # Hitung dampak lokal: Nilai Input * Koefisien
        # Fitur dengan hasil kali paling negatif (terkecil) adalah biang kerok risiko
        local_impact = np.array(input_values) * coeffs
        risk_idx = np.argmin(local_impact)
        risk_feature = FEATURES_LGBM[risk_idx]

        st.info(f"**Alasan Penolakan:** Faktor risiko utama pada **{FEATURES_LGBM[risk_idx].replace('_', ' ').title()}**.")
    else:
        st.success("### ✅ Status: APPROVED")
        st.write(f"Confidence Score: **{prob_bad:.2%}** probability of default.")
        st.balloons()