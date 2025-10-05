# app_fast.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Emotion Recognition Demo (Fast Mode)")

# -----------------------------
# 1️⃣ Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload a physiological CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    features = ['ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap']
    
    # Aggressive subsample for speed
    df_demo = df.head(500)
    
    # -----------------------------
    # 2️⃣ Generate dummy predictions
    # -----------------------------
    emotions = ['Happy','Sad','Surprise','Angry','Excited','Bored','Fear','Neutral']
    np.random.seed(42)
    preds = np.random.choice(emotions, size=3)  # fewer sequences
    
    st.subheader("Predicted Emotions with Signal Plots and Reasoning")
    
    # Precompute stats for reasoning
    ecg_mean, gsr_mean, rsp_std, emg_mean = df_demo['ecg'].mean(), df_demo['gsr'].mean(), df_demo['rsp'].std(), df_demo[['emg_zygo','emg_coru','emg_trap']].mean().mean()
    
    for i, pred in enumerate(preds):
        st.markdown(f"### Sequence {i+1}: {pred}")
        
        # Plot a small window
        fig, axs = plt.subplots(len(features), 1, figsize=(8, 2*len(features)))
        for j, f in enumerate(features):
            axs[j].plot(df_demo[f].iloc[i*10:i*10+30])
            axs[j].set_title(f)
        st.pyplot(fig)
        
        # Fast reasoning
        reasoning = []
        reasoning.append(f"**Heart rate (ECG/BVP):** {'High' if df_demo['ecg'].iloc[i*10:i*10+30].mean() > ecg_mean else 'Normal'} → {'aroused/excited' if pred in ['Happy','Excited'] else 'calm'}")
        reasoning.append(f"**GSR:** {'Spike detected' if df_demo['gsr'].iloc[i*10:i*10+30].max() > gsr_mean else 'Stable'} → {'stress/surprise' if pred=='Surprise' else 'neutral'}")
        reasoning.append(f"**RSP:** {'Irregular' if df_demo['rsp'].iloc[i*10:i*10+30].std() > rsp_std else 'Normal'} → {'fear/anxious' if pred=='Fear' else 'calm'}")
        reasoning.append(f"**Muscle EMG:** {'Tensed' if df_demo[['emg_zygo','emg_coru','emg_trap']].iloc[i*10:i*10+30].mean().mean() > 0.5 else 'Relaxed'} → {'angry/frustrated' if pred=='Angry' else 'neutral'}")
        
        st.markdown("\n".join(reasoning))
