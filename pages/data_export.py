import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64
import json
from datetime import datetime
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_download_link

st.title("📁 Data Export")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload your data first using the Data Upload section.")
    st.stop()

# Get data from session state
data = st.session_state.data

# Main export options
st.header("Export Options")

export_what = st.radio(
    "What would you like to export?",
    ["Entire Dataset", "Filtered Dataset", "Summary Statistics", "Visualizations"]
)

# Export Entire Dataset
if export_what == "Entire Dataset":
    st.subheader("Export Entire Dataset")
    
    # Display data info
    st.write(f"Dataset: {st.session_state.filename}")
    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    
    # Select export format
    export_format = st.selectbox(
        "Select Export Format",
        ["CSV", "Excel", "JSON"]
    )
    
    # Create download link
    if export_format == "CSV":
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{st.session_state.filename}_export.csv">Download CSV File</a>'
        
    elif export_format == "Excel":
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name='Data')
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{st.session_state.filename}_export.xlsx">Download Excel File</a>'
        
    elif export_format == "JSON":
        # Convert to JSON format
        json_str = data.to_json(orient="records", date_format="iso")
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{st.session_state.filename}_export.json">Download JSON File</a>'
    
    # Display download link
    st.markdown(href, unsafe_allow_html=True)
    
    # Display preview
    with st.expander("Preview Data"):
        st.dataframe(data.head(10))

# Export Filtered Dataset
elif export_what == "Filtered Dataset":
    st.subheader("Export Filtered Dataset")
    
    # Create filter options
    st.write("Apply filters to the dataset before exporting:")
    
    # Column selector for filtering
    filter_columns = st.multiselect(
        "Select columns to filter by:",
        data.columns.tolist()
    )
    
    filtered_data = data.copy()
    
    # Create filters for each selected column
    for column in filter_columns:
        st.write(f"Filter for: {column}")
        
        # Different filter types based on data type
        if pd.api.types.is_numeric_dtype(data[column]):
            # Numeric column - filter by range
            min_val = float(data[column].min())
            max_val = float(data[column].max())
            
            filter_range = st.slider(
                f"Range for {column}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            
            filtered_data = filtered_data[(filtered_data[column] >= filter_range[0]) & 
                                         (filtered_data[column] <= filter_range[1])]
            
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            # Datetime column - filter by date range
            min_date = data[column].min()
            max_date = data[column].max()
            
            filter_date_range = st.date_input(
                f"Date range for {column}:",
                value=(min_date.date(), max_date.date())
            )
            
            if len(filter_date_range) == 2:
                start_date, end_date = filter_date_range
                filtered_data = filtered_data[(filtered_data[column].dt.date >= start_date) & 
                                             (filtered_data[column].dt.date <= end_date)]
            
        else:
            # Categorical or object column - filter by selection
            unique_values = data[column].dropna().unique().tolist()
            
            if len(unique_values) <= 10:
                selected_values = st.multiselect(
                    f"Select values for {column}:",
                    unique_values,
                    default=unique_values
                )
                
                if selected_values:
                    filtered_data = filtered_data[filtered_data[column].isin(selected_values)]
            else:
                # Too many unique values, use text input for searching
                search_term = st.text_input(f"Search in {column} (leave empty for all):")
                
                if search_term:
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(search_term, case=False, na=False)]
    
    # Show filter results
    st.write(f"Filtered data has {filtered_data.shape[0]} rows (from original {data.shape[0]} rows)")
    
    # Column selection for export
    st.write("Select columns to include in the export:")
    export_columns = st.multiselect(
        "Select columns to export (leave empty to export all):",
        data.columns.tolist()
    )
    
    if export_columns:
        export_data = filtered_data[export_columns].copy()
    else:
        export_data = filtered_data.copy()
    
    # Preview filtered data
    with st.expander("Preview Filtered Data"):
        st.dataframe(export_data.head(10))
    
    # Select export format
    export_format = st.selectbox(
        "Select Export Format",
        ["CSV", "Excel", "JSON"]
    )
    
    # Generate download link
    if export_format.lower() == "csv":
        file_format = "csv"
    elif export_format.lower() == "excel":
        file_format = "excel"
    else:
        file_format = "json"
    
    download_filename = f"{st.session_state.filename.split('.')[0]}_filtered"
    
    # Create and display download link
    if not export_data.empty:
        download_link = get_download_link(export_data, download_filename, file_format)
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.warning("No data to export after applying filters.")

# Export Summary Statistics
elif export_what == "Summary Statistics":
    st.subheader("Export Summary Statistics")
    
    # Select columns for statistics
    stat_columns = st.multiselect(
        "Select columns for statistics:",
        data.columns.tolist(),
        default=st.session_state.numeric_columns[:min(5, len(st.session_state.numeric_columns))]
    )
    
    if stat_columns:
        # Calculate statistics for selected columns
        stats_df = pd.DataFrame()
        
        for column in stat_columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(data[column]):
                # Calculate statistics
                stats = {
                    'Column': column,
                    'Mean': data[column].mean(),
                    'Median': data[column].median(),
                    'Std Dev': data[column].std(),
                    'Min': data[column].min(),
                    'Max': data[column].max(),
                    'Count': data[column].count(),
                    'Missing': data[column].isna().sum()
                }
                
                # Add to stats dataframe
                stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)
            else:
                # For non-numeric columns, calculate frequency statistics
                value_counts = data[column].value_counts()
                top_value = value_counts.index[0] if not value_counts.empty else None
                
                stats = {
                    'Column': column,
                    'Type': str(data[column].dtype),
                    'Unique Values': data[column].nunique(),
                    'Most Common': top_value,
                    'Most Common Count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'Count': data[column].count(),
                    'Missing': data[column].isna().sum()
                }
                
                # Add to stats dataframe
                stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)
        
        # Display statistics
        st.write("Summary Statistics:")
        st.dataframe(stats_df)
        
        # Generate report to export
        st.subheader("Export Statistics Report")
        
        # Choose export format
        export_format = st.selectbox(
            "Select Export Format",
            ["CSV", "Excel", "HTML", "JSON"]
        )
        
        # Generate report based on format
        if export_format == "CSV":
            csv = stats_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="statistics_report.csv">Download Statistics Report (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "Excel":
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Add statistics
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Add data summary sheet
                summary = pd.DataFrame({
                    'Information': [
                        'Dataset Name',
                        'Number of Rows',
                        'Number of Columns',
                        'Generated Date',
                        'Total Missing Values'
                    ],
                    'Value': [
                        st.session_state.filename,
                        data.shape[0],
                        data.shape[1],
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        data.isna().sum().sum()
                    ]
                })
                
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add correlation matrix if we have numeric columns
                numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_columns) >= 2:
                    corr_matrix = data[numeric_columns].corr()
                    corr_matrix.to_excel(writer, sheet_name='Correlation Matrix')
                
                # Format the workbook
                workbook = writer.book
                
                # Add formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'bg_color': '#D9E1F2',
                    'border': 1
                })
                
                # Apply formatting to statistics sheet
                worksheet = writer.sheets['Statistics']
                for col_num, value in enumerate(stats_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 15)
            
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="statistics_report.xlsx">Download Statistics Report (Excel)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "HTML":
            # Create a formatted HTML report
            html = """
            <html>
            <head>
                <title>Statistics Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2C3E50; }
                    h2 { color: #3498DB; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th { background-color: #3498DB; color: white; text-align: left; padding: 8px; }
                    td { border: 1px solid #ddd; padding: 8px; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    .summary { background-color: #EBF5FB; padding: 10px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>Statistics Report</h1>
                <div class="summary">
                    <h2>Dataset Summary</h2>
                    <p><strong>Dataset Name:</strong> {filename}</p>
                    <p><strong>Number of Rows:</strong> {rows}</p>
                    <p><strong>Number of Columns:</strong> {cols}</p>
                    <p><strong>Generated Date:</strong> {date}</p>
                </div>
                
                <h2>Column Statistics</h2>
                {stats_table}
            </body>
            </html>
            """.format(
                filename=st.session_state.filename,
                rows=data.shape[0],
                cols=data.shape[1],
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                stats_table=stats_df.to_html(index=False)
            )
            
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="statistics_report.html">Download Statistics Report (HTML)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "JSON":
            # Convert stats to JSON
            stats_json = stats_df.to_json(orient="records")
            
            # Add metadata
            report = {
                "metadata": {
                    "filename": st.session_state.filename,
                    "rows": data.shape[0],
                    "columns": data.shape[1],
                    "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "statistics": json.loads(stats_json)
            }
            
            # Add correlation data if available
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_columns) >= 2:
                corr_matrix = data[numeric_columns].corr()
                report["correlation"] = json.loads(corr_matrix.to_json())
            
            # Convert to string and encode
            json_str = json.dumps(report, indent=4)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="statistics_report.json">Download Statistics Report (JSON)</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Please select at least one column for statistics.")

# Export Visualizations
elif export_what == "Visualizations":
    st.subheader("Export Visualizations")
    
    # Choose visualization type
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
    )
    
    # Configure visualization based on type
    if viz_type == "Bar Chart":
        # Bar chart configuration
        x_col = st.selectbox("Select X-axis (Categories):", st.session_state.categorical_columns)
        y_col = st.selectbox("Select Y-axis (Values):", st.session_state.numeric_columns)
        
        if x_col and y_col:
            # Create bar chart
            fig = px.bar(
                data,
                x=x_col,
                y=y_col,
                title=f"Bar Chart: {y_col} by {x_col}"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        # Line chart configuration
        x_col = st.selectbox("Select X-axis:", data.columns.tolist())
        y_cols = st.multiselect(
            "Select Y-axis (Values, multiple allowed):",
            st.session_state.numeric_columns,
            default=[st.session_state.numeric_columns[0]] if st.session_state.numeric_columns else []
        )
        
        if x_col and y_cols:
            # Create line chart
            fig = px.line(
                data,
                x=x_col,
                y=y_cols,
                title=f"Line Chart: {', '.join(y_cols)} by {x_col}"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        # Scatter plot configuration
        x_col = st.selectbox("Select X-axis:", st.session_state.numeric_columns)
        y_col = st.selectbox(
            "Select Y-axis:",
            st.session_state.numeric_columns,
            index=min(1, len(st.session_state.numeric_columns)-1) if len(st.session_state.numeric_columns) > 1 else 0
        )
        color_col = st.selectbox("Select Color Column (optional):", ["None"] + st.session_state.categorical_columns)
        
        color = None if color_col == "None" else color_col
        
        if x_col and y_col:
            # Create scatter plot
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=color,
                title=f"Scatter Plot: {y_col} vs {x_col}"
            )
            
            # Add trend line
            fig.update_layout(showlegend=True)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        # Histogram configuration
        value_col = st.selectbox("Select Column for Histogram:", st.session_state.numeric_columns)
        bins = st.slider("Number of Bins:", min_value=5, max_value=100, value=20)
        
        if value_col:
            # Create histogram
            fig = px.histogram(
                data,
                x=value_col,
                nbins=bins,
                title=f"Histogram of {value_col}"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        # Box plot configuration
        y_col = st.selectbox("Select Value Column:", st.session_state.numeric_columns)
        x_col = st.selectbox("Select Grouping Column (optional):", ["None"] + st.session_state.categorical_columns)
        
        x_col = None if x_col == "None" else x_col
        
        if y_col:
            # Create box plot
            if x_col:
                fig = px.box(
                    data,
                    x=x_col,
                    y=y_col,
                    title=f"Box Plot of {y_col} by {x_col}"
                )
            else:
                fig = px.box(
                    data,
                    y=y_col,
                    title=f"Box Plot of {y_col}"
                )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        # Correlation heatmap configuration
        corr_columns = st.multiselect(
            "Select Columns for Correlation Heatmap:",
            st.session_state.numeric_columns,
            default=st.session_state.numeric_columns[:min(5, len(st.session_state.numeric_columns))]
        )
        
        if corr_columns and len(corr_columns) >= 2:
            # Calculate correlation matrix
            corr_matrix = data[corr_columns].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Correlation Matrix"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
    
    # Export options for the visualization
    st.subheader("Export Options")
    
    if 'fig' in locals():
        export_format = st.selectbox(
            "Select Export Format",
            ["PNG", "JPEG", "SVG", "PDF", "HTML"]
        )
        
        # Set figure size
        width = st.slider("Figure Width (pixels):", min_value=600, max_value=2000, value=1200)
        height = st.slider("Figure Height (pixels):", min_value=400, max_value=1600, value=800)
        
        # Update figure size
        fig.update_layout(
            width=width,
            height=height
        )
        
        # Generate download options based on format
        if export_format == "HTML":
            # Export as interactive HTML
            html_bytes = fig.to_html(full_html=True, include_plotlyjs='cdn')
            b64 = base64.b64encode(html_bytes.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="visualization.html">Download Interactive Visualization (HTML)</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            # Export as image
            if export_format == "PNG":
                mime_type = "image/png"
                file_ext = "png"
            elif export_format == "JPEG":
                mime_type = "image/jpeg"
                file_ext = "jpg"
            elif export_format == "SVG":
                mime_type = "image/svg+xml"
                file_ext = "svg"
            elif export_format == "PDF":
                mime_type = "application/pdf"
                file_ext = "pdf"
            
            # Create download button using Plotly's to_image function
            img_bytes = fig.to_image(format=file_ext.lower(), width=width, height=height)
            b64 = base64.b64encode(img_bytes).decode()
            href = f'<a href="data:{mime_type};base64,{b64}" download="visualization.{file_ext}">Download Visualization ({export_format})</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Please configure a visualization first.")
