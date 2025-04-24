import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Set up page configuration
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'columns' not in st.session_state:
    st.session_state.columns = []
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []

# Main page header
st.title("📊 Data Analysis and Visualization App")
st.markdown("""
This application allows you to upload, analyze, visualize, and export data.
Use the sidebar to navigate through different sections.
""")

# Check if data is loaded
if st.session_state.data is not None:
    st.success(f"Data loaded: {st.session_state.filename}")
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(st.session_state.data.head(10))
    
    # Display data information
    st.subheader("Data Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", st.session_state.data.shape[0])
    with col2:
        st.metric("Columns", st.session_state.data.shape[1])
    with col3:
        st.metric("Missing Values", st.session_state.data.isna().sum().sum())
    
    # Provide links to other sections if data is loaded
    st.subheader("What would you like to do?")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("[![Analysis](https://img.shields.io/badge/Go%20to-Analysis-blue?style=for-the-badge)](Data_Analysis)")
    with cols[1]:
        st.markdown("[![Visualization](https://img.shields.io/badge/Go%20to-Visualization-green?style=for-the-badge)](Data_Visualization)")
    with cols[2]:
        st.markdown("[![Export](https://img.shields.io/badge/Go%20to-Export-orange?style=for-the-badge)](Data_Export)")
else:
    # Instructions when no data is loaded
    st.info("Please upload your data file using the Upload section in the sidebar.")
    
    # Create folder structure if it doesn't exist
    Path("pages").mkdir(exist_ok=True)
    
    # Quick demo with sample data
    if st.button("Load Sample Data for Demo"):
        sample_data = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'C', 'B', 'C'],
            'Value1': [10, 20, 15, 25, 30, 35],
            'Value2': [100, 200, 150, 250, 300, 350],
            'Date': pd.date_range(start='2023-01-01', periods=6)
        })
        
        st.session_state.data = sample_data
        st.session_state.filename = "sample_data.csv"
        st.session_state.columns = sample_data.columns.tolist()
        st.session_state.numeric_columns = sample_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.session_state.categorical_columns = sample_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.success("Sample data loaded successfully!")
        st.rerun()

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
st.sidebar.info("Upload your data and explore different sections to analyze and visualize it.")
