import streamlit as st
import pandas as pd
import os
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save

st.set_page_config(page_title="AutoML Lab - coding.biz", layout="wide")
st.title("⚗️ AutoML Lab")
st.markdown("Upload your dataset, select the target variable, and let AI handle the rest.")

# Step 1: File Upload
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Step 2: Select Task Type
    task = st.radio("📌 Select ML Task Type", ["Classification", "Regression"], horizontal=True)

    # Step 3: Select Target Column
    target = st.selectbox("🎯 Select your target column", df.columns)

    # Step 4: Run AutoML
    if st.button("🚀 Run AutoML"):
        with st.spinner("Setting up the modeling environment and training..."):

            if task == "Classification":
                s = clf_setup(data=df, target=target, session_id=123, log_experiment=False, html=False)
                best_model = clf_compare()
                results_df = clf_pull()
                clf_save(best_model, "best_model")
            else:
                s = reg_setup(data=df, target=target, session_id=123, log_experiment=False, html=False)
                best_model = reg_compare()
                results_df = reg_pull()
                reg_save(best_model, "best_model")

        st.success("✅ Model Training Complete!")

        # Display Results
        st.subheader("🏆 Model Comparison Results")
        st.dataframe(results_df)

        # Step 5: Download Best Model
        with open("best_model.pkl", "rb") as f:
            st.download_button("📥 Download Best Model (.pkl)", f, file_name="best_model.pkl")
