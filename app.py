# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Emotion Recognition Demo (CASE Dataset)")

# -----------------------------
# 1️⃣ Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload a physiological CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    features = ['ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap']
    
    # Subsample for demo speed
    df_demo = df.head(2000)
    
    # -----------------------------
    # 2️⃣ Generate dummy predictions
    # -----------------------------
    emotions = ['Happy','Sad','Surprise','Angry','Excited','Bored','Fear','Neutral']
    np.random.seed(42)
    preds = np.random.choice(emotions, size=5)
    
    st.subheader("Predicted Emotions with Signal Plots and Reasoning")
    
    for i, pred in enumerate(preds):
        st.markdown(f"### Sequence {i+1}: {pred}")
        
        # Plot all signals
        fig, axs = plt.subplots(len(features), 1, figsize=(10, 2*len(features)))
        for j, f in enumerate(features):
            axs[j].plot(df_demo[f].iloc[i*10:i*10+50])  # first 50 samples of this sequence
            axs[j].set_title(f)
        st.pyplot(fig)
        
        # Detailed reasoning
        reasoning = []
        reasoning.append(f"**Heart rate (ECG/BVP):** {'High' if df_demo['ecg'].iloc[i*10:i*10+50].mean() > df_demo['ecg'].mean() else 'Normal'} → {'aroused/excited' if pred in ['Happy','Excited'] else 'calm'}")
        reasoning.append(f"**GSR:** {'Spike detected' if df_demo['gsr'].iloc[i*10:i*10+50].max() > df_demo['gsr'].mean() else 'Stable'} → {'stress/surprise' if pred=='Surprise' else 'neutral'}")
        reasoning.append(f"**RSP:** {'Irregular' if df_demo['rsp'].iloc[i*10:i*10+50].std() > df_demo['rsp'].std() else 'Normal'} → {'fear/anxious' if pred=='Fear' else 'calm'}")
        reasoning.append(f"**Muscle EMG (Zygo/Coru/Trap):** {'Tensed' if df_demo[['emg_zygo','emg_coru','emg_trap']].iloc[i*10:i*10+50].mean().mean() > 0.5 else 'Relaxed'} → {'angry/frustrated' if pred=='Angry' else 'neutral'}")
        
        st.markdown("\n".join(reasoning))

