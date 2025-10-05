# app_ultrafast.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("Emotion Recognition Demo (Ultra-Fast)")

uploaded_file = st.file_uploader("Upload a physiological CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    features = ['ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap']
    df_demo = df.head(200)  # tiny subset

    # Dummy predictions
    emotions = ['Happy','Sad','Surprise','Angry','Excited','Bored','Fear','Neutral']
    np.random.seed(42)
    preds = np.random.choice(emotions, size=2)

    st.subheader("Predicted Emotions and Signal Plots")

    for i, pred in enumerate(preds):
        st.markdown(f"### Sequence {i+1}: {pred}")
        
        # Use Streamlit's fast line chart for each feature
        for f in features[:4]:  # only 4 features to speed up
            st.line_chart(df_demo[f].iloc[i*10:i*10+20])
        
        st.markdown(f"**Reasoning (simplified):** Detected signals indicate likely **{pred}** state.")
