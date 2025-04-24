import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import calculate_basic_stats, calculate_group_stats, filter_dataframe, normalize_data

st.title("📊 Data Analysis")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload your data first using the Data Upload section.")
    st.stop()

# Get data from session state
data = st.session_state.data
columns = st.session_state.columns
numeric_columns = st.session_state.numeric_columns
categorical_columns = st.session_state.categorical_columns

# Sidebar for analysis options
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Basic Statistics", "Group Analysis", "Data Filtering", "Advanced Analysis"]
)

# Basic Statistics Analysis
if analysis_type == "Basic Statistics":
    st.header("Basic Statistical Analysis")
    
    # Select column for analysis
    selected_column = st.selectbox(
        "Select a column for analysis:",
        numeric_columns
    )
    
    if selected_column:
        # Calculate statistics
        stats = calculate_basic_stats(data, selected_column)
        
        if stats:
            # Display statistics in a nice format
            st.subheader(f"Statistics for {selected_column}")
            
            # Create multiple columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean", round(stats['mean'], 2))
                st.metric("Minimum", round(stats['min'], 2))
                st.metric("25th Percentile", round(stats['25%'], 2))
            
            with col2:
                st.metric("Median", round(stats['median'], 2))
                st.metric("Maximum", round(stats['max'], 2))
                st.metric("75th Percentile", round(stats['75%'], 2))
            
            with col3:
                st.metric("Std Deviation", round(stats['std'], 2))
                st.metric("Count", stats['count'])
                st.metric("IQR", round(stats['IQR'], 2))
            
            # Display histogram
            st.subheader(f"Distribution of {selected_column}")
            hist_values = data[selected_column].dropna()
            
            # Calculate number of bins using Sturges' rule
            num_bins = int(np.ceil(np.log2(len(hist_values)) + 1))
            
            # Plot histogram
            hist_chart = pd.DataFrame(hist_values).hist(bins=num_bins)
            st.pyplot(hist_chart[0][0].figure)
            
            # Display boxplot
            st.subheader(f"Boxplot of {selected_column}")
            fig, ax = plt.subplots()
            data.boxplot(column=[selected_column], ax=ax)
            st.pyplot(fig)
            
            # Identify potential outliers
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = stats['IQR']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data[(data[selected_column] < lower_bound) | (data[selected_column] > upper_bound)]
            
            if not outliers.empty:
                st.subheader("Potential Outliers")
                st.write(f"Found {len(outliers)} potential outliers (values outside the range [{round(lower_bound, 2)}, {round(upper_bound, 2)}])")
                st.dataframe(outliers)
        else:
            st.error("Could not calculate statistics for the selected column.")

# Group Analysis
elif analysis_type == "Group Analysis":
    st.header("Group Analysis")
    
    # Select columns for grouping and analysis
    group_col = st.selectbox(
        "Select a column to group by:",
        categorical_columns
    )
    
    analysis_col = st.selectbox(
        "Select a numeric column to analyze:",
        numeric_columns
    )
    
    if group_col and analysis_col:
        # Calculate group statistics
        group_stats = calculate_group_stats(data, analysis_col, group_col)
        
        if group_stats is not None:
            st.subheader(f"Group Statistics of {analysis_col} by {group_col}")
            st.dataframe(group_stats)
            
            # Plot group statistics
            st.subheader("Group Comparison")
            
            chart_type = st.radio(
                "Select chart type:",
                ["Bar Chart", "Box Plot"]
            )
            
            if chart_type == "Bar Chart":
                # Bar chart for mean values by group
                st.bar_chart(data.groupby(group_col)[analysis_col].mean())
            elif chart_type == "Box Plot":
                # Box plot for distribution by group
                fig, ax = plt.subplots(figsize=(10, 6))
                data.boxplot(column=analysis_col, by=group_col, ax=ax)
                plt.title(f"{analysis_col} by {group_col}")
                plt.suptitle("")  # Remove default title
                st.pyplot(fig)
            
            # ANOVA test if there are more than 2 groups (simple implementation)
            groups = data.groupby(group_col)[analysis_col].apply(list).values
            
            if len(groups) > 1:
                try:
                    from scipy import stats
                    f_val, p_val = stats.f_oneway(*groups)
                    
                    st.subheader("One-way ANOVA Test")
                    st.write(f"F-statistic: {f_val:.4f}")
                    st.write(f"p-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.success("There is a statistically significant difference between groups (p < 0.05).")
                    else:
                        st.info("There is no statistically significant difference between groups (p >= 0.05).")
                except:
                    st.warning("Could not perform ANOVA test on the selected groups.")
        else:
            st.error("Could not perform group analysis on the selected columns.")

# Data Filtering
elif analysis_type == "Data Filtering":
    st.header("Data Filtering and Exploration")
    
    # Create filter conditions
    st.subheader("Define Filter Conditions")
    
    filters = []
    
    # Add filters dynamically
    num_filters = st.number_input("Number of filter conditions:", min_value=0, max_value=10, value=1)
    
    for i in range(int(num_filters)):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_col = st.selectbox(f"Column {i+1}:", columns, key=f"filter_col_{i}")
        
        with col2:
            # Determine suitable operators based on column type
            if filter_col in numeric_columns:
                operators = ["equals", "not equals", "greater than", "less than"]
            else:
                operators = ["equals", "not equals", "contains"]
            
            filter_op = st.selectbox(f"Operator {i+1}:", operators, key=f"filter_op_{i}")
        
        with col3:
            # Determine input type based on column type
            if filter_col in numeric_columns:
                filter_val = st.number_input(f"Value {i+1}:", key=f"filter_val_{i}")
            elif filter_col in categorical_columns:
                filter_val = st.selectbox(f"Value {i+1}:", data[filter_col].dropna().unique(), key=f"filter_val_{i}")
            else:
                filter_val = st.text_input(f"Value {i+1}:", key=f"filter_val_{i}")
        
        filters.append((filter_col, filter_op, filter_val))
    
    # Apply filters
    if st.button("Apply Filters"):
        filtered_data = filter_dataframe(data, filters)
        
        st.subheader("Filtered Data")
        st.write(f"Showing {len(filtered_data)} of {len(data)} rows ({round(len(filtered_data)/len(data)*100, 2)}%)")
        st.dataframe(filtered_data)
        
        # Download filtered data
        if not filtered_data.empty:
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                "Download Filtered Data as CSV",
                csv,
                "filtered_data.csv",
                "text/csv",
                key='download-csv'
            )
    
    # Data exploration options
    st.subheader("Data Exploration")
    
    # Show unique values for categorical columns
    if categorical_columns:
        explore_col = st.selectbox(
            "Explore unique values and counts for categorical column:",
            categorical_columns
        )
        
        if explore_col:
            value_counts = data[explore_col].value_counts().reset_index()
            value_counts.columns = [explore_col, 'Count']
            
            st.write(f"Unique values in {explore_col}:")
            st.dataframe(value_counts)
            
            # Visualize the distribution
            st.bar_chart(value_counts.set_index(explore_col))

# Advanced Analysis
elif analysis_type == "Advanced Analysis":
    st.header("Advanced Analysis")
    
    advanced_option = st.selectbox(
        "Select Advanced Analysis Method:",
        ["Correlation Analysis", "Normalization/Scaling", "Principal Component Analysis (PCA)", "Clustering"]
    )
    
    # Correlation Analysis
    if advanced_option == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
        else:
            # Select columns for correlation
            corr_columns = st.multiselect(
                "Select columns for correlation analysis:",
                numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if corr_columns and len(corr_columns) >= 2:
                # Calculate correlation matrix
                corr_matrix = data[corr_columns].corr()
                
                # Display correlation matrix
                st.write("Correlation Matrix:")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
                
                # Visualize correlation matrix as heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
                
                # Find and display highly correlated pairs
                high_corr_threshold = st.slider(
                    "High correlation threshold:",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.7,
                    step=0.05
                )
                
                # Get pairs with high correlation
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                            corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if corr_pairs:
                    st.write(f"Highly correlated pairs (|correlation| >= {high_corr_threshold}):")
                    st.table(pd.DataFrame(corr_pairs))
                    
                    # Scatter plot for selected pair
                    if len(corr_pairs) > 0:
                        st.subheader("Scatter Plot for Correlated Variables")
                        
                        selected_pair = st.selectbox(
                            "Select a pair to visualize:",
                            [f"{pair['Variable 1']} vs {pair['Variable 2']} (corr: {pair['Correlation']:.2f})" 
                             for pair in corr_pairs]
                        )
                        
                        if selected_pair:
                            var1 = selected_pair.split(" vs ")[0]
                            var2 = selected_pair.split(" vs ")[1].split(" (corr:")[0]
                            
                            # Create scatter plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(data[var1], data[var2], alpha=0.5)
                            ax.set_xlabel(var1)
                            ax.set_ylabel(var2)
                            ax.set_title(f"Scatter Plot: {var1} vs {var2}")
                            
                            # Add trend line
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                data[var1].dropna(), 
                                data[var2].dropna()
                            )
                            
                            x = np.array([data[var1].min(), data[var1].max()])
                            y = intercept + slope * x
                            ax.plot(x, y, 'r')
                            
                            st.pyplot(fig)
                else:
                    st.info(f"No variable pairs with correlation >= {high_corr_threshold} found.")
    
    # Normalization/Scaling
    elif advanced_option == "Normalization/Scaling":
        st.subheader("Data Normalization/Scaling")
        
        # Select columns to normalize
        norm_columns = st.multiselect(
            "Select numeric columns to normalize:",
            numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))]
        )
        
        if norm_columns:
            # Select normalization method
            norm_method = st.radio(
                "Select normalization method:",
                ["Min-Max Scaling (0-1)", "Z-Score Standardization", "Robust Scaling"]
            )
            
            # Map selection to method name
            method_map = {
                "Min-Max Scaling (0-1)": "minmax",
                "Z-Score Standardization": "zscore",
                "Robust Scaling": "robust"
            }
            
            # Normalize data
            if st.button("Apply Normalization"):
                normalized_data = normalize_data(data, norm_columns, method_map[norm_method])
                
                # Show original vs normalized data
                st.write("Original vs. Normalized Data (first 10 rows):")
                
                # Extract only the relevant columns
                display_cols = norm_columns + [f"{col}_normalized" for col in norm_columns]
                st.dataframe(normalized_data[display_cols].head(10))
                
                # Visualization of normalization effect
                col1, col2 = st.columns(2)
                
                if len(norm_columns) > 0:
                    selected_col = norm_columns[0]
                    
                    with col1:
                        st.write(f"Original Distribution of {selected_col}")
                        fig, ax = plt.subplots()
                        ax.hist(data[selected_col].dropna(), bins=20)
                        st.pyplot(fig)
                    
                    with col2:
                        st.write(f"Normalized Distribution of {selected_col}")
                        fig, ax = plt.subplots()
                        ax.hist(normalized_data[f"{selected_col}_normalized"].dropna(), bins=20)
                        st.pyplot(fig)
    
    # Principal Component Analysis (PCA)
    elif advanced_option == "Principal Component Analysis (PCA)":
        st.subheader("Principal Component Analysis (PCA)")
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for PCA.")
        else:
            # Select columns for PCA
            pca_columns = st.multiselect(
                "Select numeric columns for PCA:",
                numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if pca_columns and len(pca_columns) >= 2:
                # Number of components
                n_components = st.slider(
                    "Number of principal components:",
                    min_value=2,
                    max_value=min(len(pca_columns), 10),
                    value=min(2, len(pca_columns))
                )
                
                # Apply PCA
                if st.button("Run PCA"):
                    # Prepare data
                    pca_data = data[pca_columns].dropna()
                    
                    if len(pca_data) < 2:
                        st.error("Not enough data points after removing missing values.")
                    else:
                        # Standardize the data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(pca_data)
                        
                        # Apply PCA
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        # Create DataFrame with PCA results
                        pca_df = pd.DataFrame(
                            data=pca_result,
                            columns=[f'PC{i+1}' for i in range(n_components)]
                        )
                        
                        # Display PCA results
                        st.write("PCA Results (first 10 rows):")
                        st.dataframe(pca_df.head(10))
                        
                        # Display explained variance
                        st.subheader("Explained Variance")
                        explained_variance = pca.explained_variance_ratio_
                        
                        # Create DataFrame for variance
                        variance_df = pd.DataFrame({
                            'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                            'Explained Variance (%)': [v * 100 for v in explained_variance],
                            'Cumulative Variance (%)': [sum(explained_variance[:i+1]) * 100 for i in range(n_components)]
                        })
                        
                        st.table(variance_df)
                        
                        # Plot explained variance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(
                            [f'PC{i+1}' for i in range(n_components)],
                            explained_variance * 100
                        )
                        ax.set_ylabel('Explained Variance (%)')
                        ax.set_title('Explained Variance by Principal Component')
                        st.pyplot(fig)
                        
                        # Plot cumulative explained variance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(
                            [f'PC{i+1}' for i in range(n_components)],
                            [sum(explained_variance[:i+1]) * 100 for i in range(n_components)],
                            marker='o'
                        )
                        ax.set_ylabel('Cumulative Explained Variance (%)')
                        ax.set_title('Cumulative Explained Variance')
                        ax.axhline(y=80, color='r', linestyle='-', alpha=0.3)
                        ax.text(0, 81, '80% Threshold', color='r')
                        st.pyplot(fig)
                        
                        # If we have at least 2 components, show a scatter plot
                        if n_components >= 2:
                            st.subheader("PCA Visualization (First 2 Components)")
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
                            ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                            ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                            ax.set_title('PCA: First Two Principal Components')
                            
                            # Add color by category if selected
                            color_by = st.selectbox(
                                "Color points by category (optional):",
                                ["None"] + categorical_columns
                            )
                            
                            if color_by != "None":
                                # Get the categorical column data matching the PCA data
                                cat_data = data.loc[pca_data.index, color_by]
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                scatter = ax.scatter(
                                    pca_df['PC1'], 
                                    pca_df['PC2'],
                                    c=pd.factorize(cat_data)[0],
                                    cmap='viridis',
                                    alpha=0.7
                                )
                                
                                # Add legend
                                legend_elements = [
                                    plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                              label=cat, markersize=10)
                                    for i, cat in enumerate(pd.factorize(cat_data)[1])
                                ]
                                
                                ax.legend(handles=legend_elements, title=color_by)
                                ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                                ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                                ax.set_title(f'PCA: First Two Principal Components (Colored by {color_by})')
                            
                            st.pyplot(fig)
                            
                            # Loading plot (feature contributions)
                            st.subheader("Feature Contributions")
                            
                            loadings = pca.components_
                            loadings_df = pd.DataFrame(
                                loadings.T,
                                columns=[f'PC{i+1}' for i in range(n_components)],
                                index=pca_columns
                            )
                            
                            st.write("PCA Loadings (Feature Contributions):")
                            st.dataframe(loadings_df)
                            
                            # Visualize loadings for first two components
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Plot arrows for each feature
                            for i, feature in enumerate(pca_columns):
                                ax.arrow(
                                    0, 0,
                                    loadings[0, i],
                                    loadings[1, i],
                                    head_width=0.05,
                                    head_length=0.05,
                                    fc='blue',
                                    ec='blue'
                                )
                                ax.text(
                                    loadings[0, i] * 1.15,
                                    loadings[1, i] * 1.15,
                                    feature,
                                    color='black',
                                    ha='center',
                                    va='center'
                                )
                            
                            # Add circle
                            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
                            ax.add_patch(circle)
                            
                            # Set plot limits and labels
                            plt.xlim(-1.1, 1.1)
                            plt.ylim(-1.1, 1.1)
                            plt.grid(True)
                            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                            plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
                            plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
                            plt.title('PCA Loading Plot (Feature Contributions)')
                            
                            st.pyplot(fig)
    
    # Clustering
    elif advanced_option == "Clustering":
        st.subheader("Clustering Analysis")
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for clustering.")
        else:
            # Select columns for clustering
            cluster_columns = st.multiselect(
                "Select numeric columns for clustering:",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )
            
            if cluster_columns and len(cluster_columns) >= 2:
                # Number of clusters
                n_clusters = st.slider(
                    "Number of clusters:",
                    min_value=2,
                    max_value=10,
                    value=3
                )
                
                # Apply clustering
                if st.button("Run Clustering"):
                    # Prepare data
                    cluster_data = data[cluster_columns].dropna()
                    
                    if len(cluster_data) < n_clusters:
                        st.error("Not enough data points after removing missing values.")
                    else:
                        # Standardize the data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(cluster_data)
                        
                        # Apply KMeans clustering
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(scaled_data)
                        
                        # Add cluster labels to the original data
                        cluster_result = cluster_data.copy()
                        cluster_result['Cluster'] = clusters
                        
                        # Display clustering results
                        st.write("Clustering Results (first 10 rows):")
                        st.dataframe(cluster_result.head(10))
                        
                        # Cluster statistics
                        st.subheader("Cluster Statistics")
                        
                        # Count samples in each cluster
                        cluster_counts = pd.DataFrame(
                            cluster_result['Cluster'].value_counts()
                        ).reset_index()
                        cluster_counts.columns = ['Cluster', 'Count']
                        
                        # Sort by cluster number
                        cluster_counts = cluster_counts.sort_values('Cluster')
                        
                        # Display counts
                        st.write("Number of samples in each cluster:")
                        st.dataframe(cluster_counts)
                        
                        # Bar chart of cluster counts
                        st.bar_chart(cluster_counts.set_index('Cluster'))
                        
                        # Calculate cluster centers in original feature space
                        cluster_centers = pd.DataFrame(
                            scaler.inverse_transform(kmeans.cluster_centers_),
                            columns=cluster_columns
                        )
                        cluster_centers.index.name = 'Cluster'
                        cluster_centers.index = [f'Cluster {i}' for i in range(n_clusters)]
                        
                        st.write("Cluster Centers:")
                        st.dataframe(cluster_centers)
                        
                        # Visualization
                        st.subheader("Cluster Visualization")
                        
                        # If we have many dimensions, apply PCA for visualization
                        if len(cluster_columns) > 2:
                            st.write("Using PCA to visualize clusters in 2D space")
                            
                            # Apply PCA
                            pca = PCA(n_components=2)
                            pca_result = pca.fit_transform(scaled_data)
                            
                            # Create scatter plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            scatter = ax.scatter(
                                pca_result[:, 0],
                                pca_result[:, 1],
                                c=clusters,
                                cmap='viridis',
                                alpha=0.7
                            )
                            
                            # Add cluster centers
                            centers_pca = pca.transform(kmeans.cluster_centers_)
                            ax.scatter(
                                centers_pca[:, 0],
                                centers_pca[:, 1],
                                marker='X',
                                s=200,
                                c='red',
                                label='Cluster Centers'
                            )
                            
                            # Add legend
                            legend1 = ax.legend(*scatter.legend_elements(),
                                               title="Clusters")
                            ax.add_artist(legend1)
                            ax.legend()
                            
                            ax.set_xlabel('Principal Component 1')
                            ax.set_ylabel('Principal Component 2')
                            ax.set_title('Cluster Visualization (PCA)')
                            
                            st.pyplot(fig)
                        else:
                            # Direct visualization if we have only 2 features
                            fig, ax = plt.subplots(figsize=(10, 8))
                            scatter = ax.scatter(
                                cluster_data[cluster_columns[0]],
                                cluster_data[cluster_columns[1]],
                                c=clusters,
                                cmap='viridis',
                                alpha=0.7
                            )
                            
                            # Add cluster centers
                            centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
                            ax.scatter(
                                centers_original[:, 0],
                                centers_original[:, 1],
                                marker='X',
                                s=200,
                                c='red',
                                label='Cluster Centers'
                            )
                            
                            # Add legend
                            legend1 = ax.legend(*scatter.legend_elements(),
                                               title="Clusters")
                            ax.add_artist(legend1)
                            ax.legend()
                            
                            ax.set_xlabel(cluster_columns[0])
                            ax.set_ylabel(cluster_columns[1])
                            ax.set_title('Cluster Visualization')
                            
                            st.pyplot(fig)

# Add imports for matplotlib and seaborn at the top
import matplotlib.pyplot as plt
import seaborn as sns
