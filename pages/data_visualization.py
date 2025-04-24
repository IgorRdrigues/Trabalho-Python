import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_matplotlib_figure, create_plotly_figure

st.title("📈 Data Visualization")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload your data first using the Data Upload section.")
    st.stop()

# Get data from session state
data = st.session_state.data
columns = st.session_state.columns
numeric_columns = st.session_state.numeric_columns
categorical_columns = st.session_state.categorical_columns

# Sidebar for visualization options
st.sidebar.header("Visualization Options")
visualization_type = st.sidebar.selectbox(
    "Select Visualization Type",
    ["Basic Charts", "Statistical Plots", "Interactive Plots", "Custom Visualization"]
)

# Visualization engine
viz_engine = st.sidebar.radio(
    "Select Visualization Engine",
    ["Plotly (Interactive)", "Matplotlib (Static)"],
    index=0
)

# Basic Charts
if visualization_type == "Basic Charts":
    st.header("Basic Charts")
    
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"]
    )
    
    # Configuration options based on chart type
    if chart_type == "Bar Chart":
        x_col = st.selectbox("Select X-axis (Categories):", categorical_columns)
        y_col = st.selectbox("Select Y-axis (Values):", numeric_columns)
        color_col = st.selectbox("Select Color Column (optional):", ["None"] + categorical_columns)
        color = None if color_col == "None" else color_col
        
        # Horizontal bar option
        horizontal = st.checkbox("Horizontal Bar Chart")
        
        # Create chart
        if x_col and y_col:
            st.subheader(f"Bar Chart: {y_col} by {x_col}")
            
            # Prepare data
            if color:
                # For color grouping, use plotly directly
                fig = px.bar(
                    data, 
                    x=x_col if not horizontal else y_col,
                    y=y_col if not horizontal else x_col,
                    color=color,
                    title=f"{y_col} by {x_col}",
                    orientation='v' if not horizontal else 'h'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Aggregate data for simple bar chart
                agg_data = data.groupby(x_col)[y_col].mean().reset_index()
                
                if viz_engine == "Plotly (Interactive)":
                    fig = px.bar(
                        agg_data, 
                        x=x_col if not horizontal else y_col,
                        y=y_col if not horizontal else x_col,
                        title=f"{y_col} by {x_col}",
                        orientation='v' if not horizontal else 'h'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if horizontal:
                        ax.barh(agg_data[x_col], agg_data[y_col])
                        ax.set_xlabel(y_col)
                        ax.set_ylabel(x_col)
                    else:
                        ax.bar(agg_data[x_col], agg_data[y_col])
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                    ax.set_title(f"{y_col} by {x_col}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    elif chart_type == "Line Chart":
        x_col = st.selectbox("Select X-axis:", columns)
        y_cols = st.multiselect("Select Y-axis (Values, multiple allowed):", numeric_columns)
        
        # Check if x-column is datetime
        if x_col in data.columns and pd.api.types.is_datetime64_any_dtype(data[x_col]):
            st.info(f"Column '{x_col}' is detected as datetime. Data will be automatically sorted.")
            sort_by_x = True
        else:
            sort_by_x = st.checkbox("Sort by X-axis values", value=True)
        
        # Create chart
        if x_col and y_cols:
            st.subheader(f"Line Chart: {', '.join(y_cols)} by {x_col}")
            
            # Prepare data - sort if needed
            plot_data = data.copy()
            if sort_by_x:
                plot_data = plot_data.sort_values(by=x_col)
            
            if viz_engine == "Plotly (Interactive)":
                fig = go.Figure()
                
                for y_col in y_cols:
                    fig.add_trace(go.Scatter(
                        x=plot_data[x_col],
                        y=plot_data[y_col],
                        mode='lines+markers',
                        name=y_col
                    ))
                
                fig.update_layout(
                    title=f"Line Chart: {', '.join(y_cols)} by {x_col}",
                    xaxis_title=x_col,
                    yaxis_title="Value",
                    legend_title="Variables"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for y_col in y_cols:
                    ax.plot(plot_data[x_col], plot_data[y_col], marker='o', label=y_col)
                
                ax.set_xlabel(x_col)
                ax.set_ylabel("Value")
                ax.set_title(f"Line Chart: {', '.join(y_cols)} by {x_col}")
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis:", numeric_columns)
        y_col = st.selectbox("Select Y-axis:", numeric_columns, index=min(1, len(numeric_columns)-1))
        color_col = st.selectbox("Select Color Column (optional):", ["None"] + categorical_columns)
        size_col = st.selectbox("Select Size Column (optional):", ["None"] + numeric_columns)
        
        # Add trendline option
        add_trendline = st.checkbox("Add Trendline", value=True)
        
        # Create chart
        if x_col and y_col:
            st.subheader(f"Scatter Plot: {y_col} vs {x_col}")
            
            color = None if color_col == "None" else color_col
            size = None if size_col == "None" else size_col
            
            if viz_engine == "Plotly (Interactive)":
                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color,
                    size=size,
                    title=f"Scatter Plot: {y_col} vs {x_col}",
                    trendline="ols" if add_trendline else None
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if color:
                    # Get unique categories and create scatter plot for each
                    for cat in data[color].unique():
                        subset = data[data[color] == cat]
                        ax.scatter(subset[x_col], subset[y_col], label=cat, alpha=0.7)
                    ax.legend()
                else:
                    ax.scatter(data[x_col], data[y_col], alpha=0.7)
                
                # Add trendline
                if add_trendline:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        data[x_col].dropna(), 
                        data[y_col].dropna()
                    )
                    
                    x = np.array([data[x_col].min(), data[x_col].max()])
                    y = intercept + slope * x
                    ax.plot(x, y, 'r', label=f'y = {slope:.2f}x + {intercept:.2f} (R²={r_value**2:.2f})')
                    ax.legend()
                
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
                
                plt.tight_layout()
                st.pyplot(fig)
    
    elif chart_type == "Pie Chart":
        value_col = st.selectbox("Select Value Column:", numeric_columns)
        name_col = st.selectbox("Select Category Column:", categorical_columns)
        
        # Create chart
        if value_col and name_col:
            st.subheader(f"Pie Chart: {value_col} by {name_col}")
            
            # Prepare data - aggregate by category
            pie_data = data.groupby(name_col)[value_col].sum().reset_index()
            
            if viz_engine == "Plotly (Interactive)":
                fig = px.pie(
                    pie_data,
                    values=value_col,
                    names=name_col,
                    title=f"Distribution of {value_col} by {name_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.pie(
                    pie_data[value_col],
                    labels=pie_data[name_col],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                ax.set_title(f"Distribution of {value_col} by {name_col}")
                st.pyplot(fig)
    
    elif chart_type == "Histogram":
        value_col = st.selectbox("Select Column for Histogram:", numeric_columns)
        bins = st.slider("Number of Bins:", min_value=5, max_value=100, value=20)
        color_col = st.selectbox("Group By (optional):", ["None"] + categorical_columns)
        
        # Create chart
        if value_col:
            st.subheader(f"Histogram: {value_col}")
            
            color = None if color_col == "None" else color_col
            
            if viz_engine == "Plotly (Interactive)":
                fig = px.histogram(
                    data,
                    x=value_col,
                    color=color,
                    nbins=bins,
                    title=f"Histogram of {value_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if color:
                    # For each category, create a histogram
                    for cat in data[color].unique():
                        subset = data[data[color] == cat]
                        ax.hist(subset[value_col], bins=bins, alpha=0.5, label=cat)
                    ax.legend()
                else:
                    ax.hist(data[value_col], bins=bins)
                
                ax.set_xlabel(value_col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {value_col}")
                
                plt.tight_layout()
                st.pyplot(fig)

# Statistical Plots
elif visualization_type == "Statistical Plots":
    st.header("Statistical Plots")
    
    stat_plot_type = st.selectbox(
        "Select Statistical Plot Type",
        ["Box Plot", "Violin Plot", "Density Plot", "Q-Q Plot", "Heatmap"]
    )
    
    # Box Plot
    if stat_plot_type == "Box Plot":
        y_col = st.selectbox("Select Value Column:", numeric_columns)
        x_col = st.selectbox("Select Grouping Column (optional):", ["None"] + categorical_columns)
        
        x_col = None if x_col == "None" else x_col
        
        if y_col:
            st.subheader(f"Box Plot: {y_col}" + (f" by {x_col}" if x_col else ""))
            
            if viz_engine == "Plotly (Interactive)":
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
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if x_col:
                    data.boxplot(column=y_col, by=x_col, ax=ax)
                    plt.suptitle("")  # Remove default title
                    ax.set_title(f"Box Plot of {y_col} by {x_col}")
                else:
                    data.boxplot(column=y_col, ax=ax)
                    ax.set_title(f"Box Plot of {y_col}")
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Violin Plot
    elif stat_plot_type == "Violin Plot":
        y_col = st.selectbox("Select Value Column:", numeric_columns)
        x_col = st.selectbox("Select Grouping Column:", categorical_columns)
        
        if y_col and x_col:
            st.subheader(f"Violin Plot: {y_col} by {x_col}")
            
            if viz_engine == "Plotly (Interactive)":
                fig = px.violin(
                    data,
                    x=x_col,
                    y=y_col,
                    box=True,  # include box plot inside violin
                    points="all",  # show all points
                    title=f"Violin Plot of {y_col} by {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                import seaborn as sns
                sns.violinplot(x=x_col, y=y_col, data=data, ax=ax)
                
                ax.set_title(f"Violin Plot of {y_col} by {x_col}")
                plt.tight_layout()
                st.pyplot(fig)
    
    # Density Plot
    elif stat_plot_type == "Density Plot":
        value_cols = st.multiselect("Select Value Columns:", numeric_columns)
        
        if value_cols:
            st.subheader(f"Density Plot: {', '.join(value_cols)}")
            
            if viz_engine == "Plotly (Interactive)":
                fig = go.Figure()
                
                for col in value_cols:
                    # Create histogram with kde
                    fig.add_trace(go.Histogram(
                        x=data[col],
                        name=col,
                        histnorm='probability density',
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    title=f"Density Plot of {', '.join(value_cols)}",
                    xaxis_title="Value",
                    yaxis_title="Density",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                import seaborn as sns
                for col in value_cols:
                    sns.kdeplot(data[col], ax=ax, label=col)
                
                ax.set_title(f"Density Plot of {', '.join(value_cols)}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Q-Q Plot (Quantile-Quantile plot for normality check)
    elif stat_plot_type == "Q-Q Plot":
        value_col = st.selectbox("Select Column for Q-Q Plot:", numeric_columns)
        
        if value_col:
            st.subheader(f"Q-Q Plot: {value_col}")
            
            if viz_engine == "Plotly (Interactive)":
                from scipy import stats
                
                # Calculate theoretical quantiles
                sorted_data = sorted(data[value_col].dropna())
                n = len(sorted_data)
                
                # Calculate empirical and theoretical quantiles
                empirical_quants = np.arange(1, n + 1) / (n + 1)  # Plotting positions
                theoretical_quants = stats.norm.ppf(empirical_quants)
                
                # Create QQ plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=theoretical_quants,
                    y=sorted_data,
                    mode='markers',
                    name='Data'
                ))
                
                # Add reference line
                min_val = min(theoretical_quants)
                max_val = max(theoretical_quants)
                
                # Calculate slope and intercept for reference line
                slope, intercept, r, p, stderr = stats.linregress(theoretical_quants, sorted_data)
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[intercept + slope * min_val, intercept + slope * max_val],
                    mode='lines',
                    name='Reference Line'
                ))
                
                fig.update_layout(
                    title=f"Q-Q Plot for {value_col}",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                from scipy import stats
                
                # Create QQ plot
                stats.probplot(data[value_col].dropna(), plot=ax)
                
                ax.set_title(f"Q-Q Plot for {value_col}")
                plt.tight_layout()
                st.pyplot(fig)
    
    # Heatmap (Correlation Matrix)
    elif stat_plot_type == "Heatmap":
        corr_columns = st.multiselect(
            "Select Columns for Correlation Heatmap:",
            numeric_columns,
            default=numeric_columns[:min(5, len(numeric_columns))]
        )
        
        if corr_columns and len(corr_columns) >= 2:
            st.subheader("Correlation Heatmap")
            
            # Calculate correlation matrix
            corr_matrix = data[corr_columns].corr()
            
            if viz_engine == "Plotly (Interactive)":
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                import seaborn as sns
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                
                ax.set_title("Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig)

# Interactive Plots
elif visualization_type == "Interactive Plots":
    st.header("Interactive Plots")
    
    interactive_plot_type = st.selectbox(
        "Select Interactive Plot Type",
        ["Animated Bar Chart", "Bubble Chart", "Interactive 3D Scatter", "Map Visualization"]
    )
    
    # Animated Bar Chart
    if interactive_plot_type == "Animated Bar Chart":
        category_col = st.selectbox("Select Category Column:", categorical_columns)
        value_col = st.selectbox("Select Value Column:", numeric_columns)
        time_col = st.selectbox("Select Time Column (for animation):", columns)
        
        if category_col and value_col and time_col:
            st.subheader(f"Animated Bar Chart: {value_col} by {category_col} over {time_col}")
            
            # Check if the time column is datetime
            if pd.api.types.is_datetime64_any_dtype(data[time_col]):
                # Format datetime to string for better display
                data[f'{time_col}_str'] = data[time_col].dt.strftime('%Y-%m-%d')
                time_col_use = f'{time_col}_str'
            else:
                time_col_use = time_col
            
            # Create animated bar chart
            fig = px.bar(
                data,
                x=category_col,
                y=value_col,
                animation_frame=time_col_use,
                title=f"{value_col} by {category_col} over {time_col}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Bubble Chart
    elif interactive_plot_type == "Bubble Chart":
        x_col = st.selectbox("Select X-axis:", numeric_columns)
        y_col = st.selectbox("Select Y-axis:", numeric_columns, index=min(1, len(numeric_columns)-1))
        size_col = st.selectbox("Select Size Column:", numeric_columns, index=min(2, len(numeric_columns)-1))
        color_col = st.selectbox("Select Color Column:", ["None"] + categorical_columns)
        
        color = None if color_col == "None" else color_col
        
        if x_col and y_col and size_col:
            st.subheader(f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})")
            
            # Create bubble chart
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                size=size_col,
                color=color,
                hover_name=color if color else None,
                size_max=60,
                title=f"Bubble Chart: {y_col} vs {x_col} (size: {size_col})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Interactive 3D Scatter
    elif interactive_plot_type == "Interactive 3D Scatter":
        x_col = st.selectbox("Select X-axis:", numeric_columns)
        y_col = st.selectbox("Select Y-axis:", numeric_columns, index=min(1, len(numeric_columns)-1))
        z_col = st.selectbox("Select Z-axis:", numeric_columns, index=min(2, len(numeric_columns)-1))
        color_col = st.selectbox("Select Color Column:", ["None"] + categorical_columns + numeric_columns)
        
        color = None if color_col == "None" else color_col
        
        if x_col and y_col and z_col:
            st.subheader(f"3D Scatter Plot: {x_col}, {y_col}, {z_col}")
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                data,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color,
                opacity=0.7,
                title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}"
            )
            
            # Update layout for better viewing
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Map Visualization
    elif interactive_plot_type == "Map Visualization":
        st.write("For map visualization, you need columns with latitude and longitude or geographic region names.")
        
        # Try to identify potential lat/lon columns
        lat_cols = [col for col in columns if any(s in col.lower() for s in ['lat', 'latitude'])]
        lon_cols = [col for col in columns if any(s in col.lower() for s in ['lon', 'long', 'longitude'])]
        
        # Select columns for map
        lat_col = st.selectbox("Select Latitude Column:", ["None"] + lat_cols + numeric_columns)
        lon_col = st.selectbox("Select Longitude Column:", ["None"] + lon_cols + numeric_columns)
        
        # For region-based maps
        region_col = st.selectbox("OR Select Region Column (country, state, etc.):", ["None"] + categorical_columns)
        
        # Value column for coloring
        value_col = st.selectbox("Select Value Column for Color:", ["None"] + numeric_columns)
        
        # Check if we have coordinates
        have_coords = lat_col != "None" and lon_col != "None"
        have_regions = region_col != "None"
        have_values = value_col != "None"
        
        if have_coords or have_regions:
            if have_coords:
                st.subheader("Map Visualization (Coordinate-based)")
                
                # Create scatter map
                fig = px.scatter_mapbox(
                    data,
                    lat=lat_col,
                    lon=lon_col,
                    color=value_col if have_values else None,
                    hover_name=region_col if have_regions else None,
                    zoom=1
                )
                
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":0,"l":0,"b":0}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif have_regions:
                st.subheader("Map Visualization (Region-based)")
                
                if have_values:
                    # Choropleth map
                    # Note: This is a simplified version. Actual implementation may require
                    # matching region names to standard codes (ISO for countries, etc.)
                    st.warning("Region-based maps require matching region names to standard codes. This is a simplified example.")
                    
                    # Try to determine if we're dealing with countries
                    sample_values = data[region_col].dropna().unique()[:5]
                    st.write(f"Sample region values: {', '.join(sample_values)}")
                    
                    # Create a simple choropleth map
                    fig = px.choropleth(
                        data,
                        locations=region_col,
                        locationmode="country names",  # Assumes country names
                        color=value_col,
                        title=f"Choropleth Map: {value_col} by {region_col}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Please select a value column for region-based maps.")
        else:
            st.warning("Please select valid columns for map visualization.")

# Custom Visualization
elif visualization_type == "Custom Visualization":
    st.header("Custom Visualization")
    
    st.write("Create a custom visualization by selecting the chart type and variables:")
    
    # Select chart type
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["scatter", "line", "bar", "histogram", "box", "pie", "heatmap"]
    )
    
    # Configure variables based on chart type
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("X-axis:", columns)
        
        if chart_type in ["scatter", "line", "bar", "box"]:
            y_col = st.selectbox("Y-axis:", numeric_columns)
        else:
            y_col = st.selectbox("Y-axis (optional):", ["None"] + numeric_columns)
            y_col = None if y_col == "None" else y_col
    
    with col2:
        color_col = st.selectbox("Color By (optional):", ["None"] + categorical_columns)
        color_col = None if color_col == "None" else color_col
        
        title = st.text_input("Chart Title:", f"Custom {chart_type.capitalize()} Chart")
    
    # Create visualization
    if x_col:
        if viz_engine == "Plotly (Interactive)":
            fig = create_plotly_figure(data, chart_type, x_col, y_col, color_col, title)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not create the selected visualization. Please check your selections.")
        else:
            fig = create_matplotlib_figure(data, chart_type, x_col, y_col, color_col, title)
            
            if fig:
                st.pyplot(fig)
            else:
                st.error("Could not create the selected visualization. Please check your selections.")
    
    # Custom plot options
    st.subheader("Advanced Customization")
    
    with st.expander("Color Scheme and Styling"):
        if viz_engine == "Plotly (Interactive)":
            color_scheme = st.selectbox(
                "Color Scheme:",
                ["default", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Greens", "Reds"]
            )
            
            show_grid = st.checkbox("Show Grid Lines", value=True)
            
            # Apply customization if a figure exists
            if 'fig' in locals() and fig:
                if color_scheme != "default":
                    fig.update_traces(marker=dict(colorscale=color_scheme.lower()))
                
                fig.update_layout(
                    xaxis=dict(showgrid=show_grid),
                    yaxis=dict(showgrid=show_grid)
                )
                
                st.plotly_chart(fig, use_container_width=True)
