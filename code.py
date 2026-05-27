
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

#  PAGE SETTINGS 
st.set_page_config(page_title="Customer Lifetime Value Engine", layout="wide")

st.title("📊 Customer Lifetime Value Engine")

#  SHOW FILES 
st.subheader("📁 Files Available in Current Folder")
st.write(os.listdir())

#  LOAD DATA-
try:
    df = pd.read_csv("train_BRCpofr.csv")

    st.success("✅ Dataset Loaded Successfully")

    #  PREVIEW 
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    #  SHAPE 
    st.subheader("📌 Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    #  COLUMNS
    st.subheader("📌 Columns")
    st.write(df.columns)

    #  MISSING VALUES 
    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    #  NUMERIC COLUMNS 
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    st.subheader("📌 Numeric Columns")
    st.write(numeric_cols)

    #  HISTOGRAM 
    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select Numeric Column for Histogram",
            numeric_cols
        )

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, ax=ax1)

        plt.xticks(rotation=45)

        st.pyplot(fig1)

    #  CORRELATION HEATMAP
    if len(numeric_cols) > 1:

        st.subheader("📈 Correlation Heatmap")

        fig2, ax2 = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap='coolwarm',
            ax=ax2
        )

        st.pyplot(fig2)

    #  COUNT PLOT 
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:

        selected_cat = st.selectbox(
            "Select Categorical Column",
            categorical_cols
        )

        st.subheader(f"📊 Count Plot of {selected_cat}")

        fig3, ax3 = plt.subplots(figsize=(10, 5))

        sns.countplot(
            x=df[selected_cat],
            ax=ax3
        )

        plt.xticks(rotation=45)

        st.pyplot(fig3)

except Exception as e:
    st.error(f"❌ Error: {e}")
