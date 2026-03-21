from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Credit Scoring System - Final Stable")

# 1. Load Model Produksi (10 Fitur)
# Pastikan nama file sesuai dengan yang Anda simpan di folder
model_decision = joblib.load('model_decision_lgbm.pkl')
model_explanation = joblib.load('model_explanation_lr.pkl')

FEATURE_NAMES = [
    'mths_since_issue_d', 'mths_since_last_pymnt_d', 'annual_inc', 
    'dti', 'revol_util', 'mths_since_earliest_cr_line', 
    'tot_cur_bal', 'int_rate', 'revol_bal', 'total_rev_hi_lim'
]

class LoanApplication(BaseModel):
    mths_since_issue_d: float
    mths_since_last_pymnt_d: float
    annual_inc: float
    dti: float
    revol_util: float
    mths_since_earliest_cr_line: float
    tot_cur_bal: float
    int_rate: float
    revol_bal: float
    total_rev_hi_lim: float

@app.post("/predict")
def predict_credit_risk(data: LoanApplication):
    # Konversi input ke DataFrame dengan urutan kolom yang benar
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])[FEATURE_NAMES]

    # --- PENANGANAN OUTLIER (CLIPPING) ---
    # Batasi nilai Annual Inc secara logis agar tidak mendominasi model linear
    df_proc = df_input.copy()
    df_proc['annual_inc'] = df_proc['annual_inc'].clip(upper=500000000) # Batas 500 Juta

    # --- LAYER 1: DECISION (LightGBM) ---
    # Menggunakan Threshold 0.68 yang Anda temukan sangat stabil
    prob_bad_lgbm = 1 - model_decision.predict_proba(df_proc)[:, 1][0]
    is_bad = 1 if prob_bad_lgbm >= 0.68 else 0
    
    explanation_msg = "Nasabah memiliki profil risiko rendah (Approved)."
    
    if is_bad == 1:
        # Kita tidak lagi mengalikan dengan nilai input nasabah yang ekstrem
        # Tapi kita lihat fitur mana yang memiliki koefisien (bobot) risiko tertinggi di model
        coeffs = np.abs(model_explanation.coef_[0])
        
        # Cari fitur yang secara statistik paling 'riskan' di model Anda
        top_trigger_idx = np.argmax(coeffs)
        top_trigger_feature = FEATURE_NAMES[top_trigger_idx]
        
        explanation_msg = (
            f"Rejected. Risiko tinggi terdeteksi berdasarkan profil kredit Anda. "
            f"Faktor risiko utama pada model: {top_trigger_feature.replace('_', ' ').title()}."
        )

    return {
        "decision": "Rejected" if is_bad == 1 else "Approved",
        "probability_bad": f"{prob_bad_lgbm:.2%}",
        "message": explanation_msg,
        "model_performance": {
            "test_roc_auc": 0.8524,
            "test_recall_bad": 0.8374,
            "stability_gap": "0.0218 (Very Stable)"
        }
    }