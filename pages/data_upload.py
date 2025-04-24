import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, get_column_types

st.title("📤 Data Upload")
st.markdown("Upload your data file (CSV or Excel) to get started with analysis.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

# Setup container for options
options_container = st.container()

with options_container:
    if uploaded_file is not None:
        # Load the data
        data, filename = load_data(uploaded_file)
        
        if data is not None:
            # Preview the data
            st.subheader("Data Preview")
            st.dataframe(data.head(5))
            
            # Data info
            st.subheader("Data Information")
            
            # Display basic info in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Missing Values", data.isna().sum().sum())
            
            # Column data types
            st.subheader("Column Data Types")
            dtypes_df = pd.DataFrame({
                'Column': data.columns,
                'Data Type': [str(data[col].dtype) for col in data.columns],
                'Non-Null Count': [data[col].count() for col in data.columns],
                'Missing Values': [data[col].isna().sum() for col in data.columns]
            })
            st.dataframe(dtypes_df)
            
            # Data preprocessing options
            st.subheader("Data Preprocessing Options")
            
            preprocess_options = st.multiselect(
                "Select preprocessing options:",
                ["Handle missing values", "Remove duplicates", "Convert data types"]
            )
            
            modified_data = data.copy()
            
            # Handle missing values
            if "Handle missing values" in preprocess_options:
                st.write("Handle Missing Values")
                
                missing_cols = [col for col in data.columns if data[col].isna().any()]
                if missing_cols:
                    for col in missing_cols:
                        st.write(f"Column: {col} - {data[col].isna().sum()} missing values")
                        
                        strategy = st.selectbox(
                            f"How to handle missing values in {col}?",
                            ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with value"],
                            key=f"missing_{col}"
                        )
                        
                        if strategy == "Drop rows":
                            modified_data = modified_data.dropna(subset=[col])
                        elif strategy == "Fill with mean" and pd.api.types.is_numeric_dtype(data[col]):
                            modified_data[col] = modified_data[col].fillna(modified_data[col].mean())
                        elif strategy == "Fill with median" and pd.api.types.is_numeric_dtype(data[col]):
                            modified_data[col] = modified_data[col].fillna(modified_data[col].median())
                        elif strategy == "Fill with mode":
                            modified_data[col] = modified_data[col].fillna(modified_data[col].mode()[0])
                        elif strategy == "Fill with value":
                            fill_value = st.text_input(f"Enter value to fill in {col}", key=f"fill_{col}")
                            if fill_value:
                                # Try to convert the fill_value to the column's data type
                                try:
                                    if pd.api.types.is_numeric_dtype(data[col]):
                                        fill_value = float(fill_value)
                                    modified_data[col] = modified_data[col].fillna(fill_value)
                                except ValueError:
                                    st.error(f"The value could not be converted to the column's data type.")
                else:
                    st.write("No missing values found in the dataset.")
            
            # Remove duplicates
            if "Remove duplicates" in preprocess_options:
                st.write("Remove Duplicates")
                
                duplicate_rows = modified_data.duplicated().sum()
                if duplicate_rows > 0:
                    st.write(f"Found {duplicate_rows} duplicate rows")
                    
                    subset_cols = st.multiselect(
                        "Select columns to consider for identifying duplicates (leave empty to use all columns):",
                        modified_data.columns.tolist(),
                        key="duplicate_cols"
                    )
                    
                    keep_option = st.radio(
                        "Which duplicate to keep?",
                        ["first", "last", "none"],
                        key="duplicate_keep"
                    )
                    
                    if st.button("Remove Duplicates"):
                        if subset_cols:
                            modified_data = modified_data.drop_duplicates(subset=subset_cols, keep=keep_option)
                        else:
                            modified_data = modified_data.drop_duplicates(keep=keep_option)
                        st.success(f"Removed {duplicate_rows} duplicate rows")
                else:
                    st.write("No duplicate rows found in the dataset.")
            
            # Convert data types
            if "Convert data types" in preprocess_options:
                st.write("Convert Data Types")
                
                for col in modified_data.columns:
                    current_type = modified_data[col].dtype
                    st.write(f"Column: {col} - Current type: {current_type}")
                    
                    target_type = st.selectbox(
                        f"Convert {col} to:",
                        ["Keep current", "numeric", "text", "category", "datetime"],
                        key=f"convert_{col}"
                    )
                    
                    if target_type != "Keep current":
                        try:
                            if target_type == "numeric":
                                modified_data[col] = pd.to_numeric(modified_data[col], errors='coerce')
                            elif target_type == "text":
                                modified_data[col] = modified_data[col].astype(str)
                            elif target_type == "category":
                                modified_data[col] = modified_data[col].astype('category')
                            elif target_type == "datetime":
                                date_format = st.text_input(
                                    f"Enter date format for {col} (e.g., '%Y-%m-%d', leave empty for auto-detection):",
                                    key=f"date_format_{col}"
                                )
                                if date_format:
                                    modified_data[col] = pd.to_datetime(modified_data[col], format=date_format, errors='coerce')
                                else:
                                    modified_data[col] = pd.to_datetime(modified_data[col], errors='coerce')
                        except Exception as e:
                            st.error(f"Error converting {col}: {str(e)}")
            
            # Save changes button
            if st.button("Apply Changes and Continue"):
                # Calculate column types
                columns, numeric_columns, categorical_columns = get_column_types(modified_data)
                
                # Update session state
                st.session_state.data = modified_data
                st.session_state.filename = filename
                st.session_state.columns = columns
                st.session_state.numeric_columns = numeric_columns
                st.session_state.categorical_columns = categorical_columns
                
                st.success("Data loaded successfully. You can now proceed to analysis and visualization.")
                
                # Show sample of processed data
                st.subheader("Processed Data Preview")
                st.dataframe(modified_data.head(5))
                
                # Provide navigation links
                st.markdown("### Navigate to:")
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("[![Analysis](https://img.shields.io/badge/Go%20to-Analysis-blue?style=for-the-badge)](Data_Analysis)")
                with cols[1]:
                    st.markdown("[![Visualization](https://img.shields.io/badge/Go%20to-Visualization-green?style=for-the-badge)](Data_Visualization)")
                with cols[2]:
                    st.markdown("[![Export](https://img.shields.io/badge/Go%20to-Export-orange?style=for-the-badge)](Data_Export)")
    else:
        st.info("Please upload a CSV or Excel file to get started.")
        
        # Instructions for using the app
        st.markdown("""
        ### Instructions:
        1. Upload your data file in CSV or Excel format
        2. Preview the data and check for any issues
        3. Apply preprocessing options if needed
        4. Once your data is ready, click 'Apply Changes and Continue'
        5. Navigate to other sections to analyze, visualize, and export your data
        
        ### Supported File Formats:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        """)
