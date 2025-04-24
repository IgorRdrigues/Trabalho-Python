import pandas as pd
import numpy as np
from sklearn import preprocessing
import base64
import io
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

def load_data(file):
    """
    Carrega dados de um arquivo enviado
    
    Args:
        file: O objeto de arquivo enviado
    
    Returns:
        DataFrame pandas contendo os dados
    """
    filename = file.name
    
    try:
        if filename.endswith('.csv'):
            data = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        elif filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            st.error("Formato de arquivo não suportado. Por favor, envie um arquivo CSV ou Excel.")
            return None, None
        
        return data, filename
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {str(e)}")
        st.info("Tentando métodos alternativos de carregamento...")
        
        try:
            # Tentar codificações alternativas para CSV
            if filename.endswith('.csv'):
                for encoding in ['latin1', 'ISO-8859-1', 'cp1252']:
                    try:
                        data = pd.read_csv(file, encoding=encoding)
                        st.success(f"Arquivo carregado com sucesso usando codificação {encoding}")
                        return data, filename
                    except:
                        pass
            
            return None, None
        except Exception as e:
            st.error(f"Falha no carregamento: {str(e)}")
            return None, None

def get_column_types(df):
    """
    Determine column types from DataFrame
    
    Args:
        df: pandas DataFrame
    
    Returns:
        lists of column names by type
    """
    if df is None:
        return [], [], []
    
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also include boolean and datetime columns as categorical
    bool_columns = df.select_dtypes(include=['bool']).columns.tolist()
    date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    categorical_columns = categorical_columns + bool_columns + date_columns
    
    return columns, numeric_columns, categorical_columns

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a numeric column
    
    Args:
        df: pandas DataFrame
        column: name of the column to analyze
    
    Returns:
        dictionary of statistics
    """
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return None
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isna().sum()
    }
    
    # Add mode if it exists (might be multiple values)
    mode_values = df[column].mode().tolist()
    if mode_values:
        stats['mode'] = mode_values[0] if len(mode_values) == 1 else mode_values
    
    # Calculate percentiles
    stats['25%'] = df[column].quantile(0.25)
    stats['75%'] = df[column].quantile(0.75)
    stats['IQR'] = stats['75%'] - stats['25%']
    
    return stats

def calculate_group_stats(df, numeric_column, groupby_column):
    """
    Calculate group statistics
    
    Args:
        df: pandas DataFrame
        numeric_column: name of the numeric column to analyze
        groupby_column: name of the column to group by
    
    Returns:
        DataFrame with group statistics
    """
    if not all(col in df.columns for col in [numeric_column, groupby_column]):
        return None
    
    grouped = df.groupby(groupby_column)[numeric_column].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    
    return grouped

def filter_dataframe(df, filters):
    """
    Filtra o dataframe com base em condições
    
    Args:
        df: pandas DataFrame
        filters: lista de condições de filtro
    
    Returns:
        DataFrame filtrado
    """
    if not filters:
        return df
    
    filtered_df = df.copy()
    
    for column, operator, value in filters:
        if operator == "equals" or operator == "igual a":
            filtered_df = filtered_df[filtered_df[column] == value]
        elif operator == "not equals" or operator == "diferente de":
            filtered_df = filtered_df[filtered_df[column] != value]
        elif operator == "greater than" or operator == "maior que":
            filtered_df = filtered_df[filtered_df[column] > value]
        elif operator == "less than" or operator == "menor que":
            filtered_df = filtered_df[filtered_df[column] < value]
        elif operator == "contains" or operator == "contém":
            filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
    
    return filtered_df

def get_download_link(df, filename, file_format="csv"):
    """
    Gera um link de download para o dataframe
    
    Args:
        df: pandas DataFrame
        filename: nome do arquivo
        file_format: formato de exportação (csv, excel, json)
    
    Returns:
        Link HTML para baixar o arquivo
    """
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        file_ext = "csv"
        mime_type = "text/csv"
    elif file_format == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Dados')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        file_ext = "xlsx"
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_format == "json":
        json_str = df.to_json(orient="records")
        b64 = base64.b64encode(json_str.encode()).decode()
        file_ext = "json"
        mime_type = "application/json"
    else:
        return None
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{file_ext}">Baixar Arquivo {file_ext.upper()}</a>'
    return href

def create_matplotlib_figure(df, plot_type, x_col, y_col, hue=None, title="Gráfico"):
    """
    Cria uma figura matplotlib com base no tipo de gráfico
    
    Args:
        df: pandas DataFrame
        plot_type: tipo de gráfico a ser criado
        x_col: coluna para o eixo x
        y_col: coluna para o eixo y
        hue: coluna para agrupamento por cor
        title: título do gráfico
    
    Returns:
        figura matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == "bar":
        if hue:
            groups = df.groupby([x_col, hue])[y_col].mean().unstack()
            groups.plot(kind='bar', ax=ax)
        else:
            df.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax)
    
    elif plot_type == "line":
        if hue:
            for name, group in df.groupby(hue):
                group.plot(x=x_col, y=y_col, kind='line', ax=ax, label=name)
        else:
            df.plot(x=x_col, y=y_col, kind='line', ax=ax)
    
    elif plot_type == "scatter":
        if hue:
            for name, group in df.groupby(hue):
                ax.scatter(group[x_col], group[y_col], label=name, alpha=0.7)
            ax.legend()
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.7)
    
    elif plot_type == "histogram":
        ax.hist(df[x_col], bins=20, alpha=0.7)
    
    elif plot_type == "box":
        if hue:
            df.boxplot(column=y_col, by=x_col, ax=ax)
        else:
            df.boxplot(column=y_col, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(x_col)
    if y_col and plot_type != "histogram":
        ax.set_ylabel(y_col)
    
    plt.tight_layout()
    return fig

def create_plotly_figure(df, plot_type, x_col, y_col, color=None, title="Gráfico"):
    """
    Cria uma figura plotly com base no tipo de gráfico
    
    Args:
        df: pandas DataFrame
        plot_type: tipo de gráfico a ser criado
        x_col: coluna para o eixo x
        y_col: coluna para o eixo y
        color: coluna para agrupamento por cor
        title: título do gráfico
    
    Returns:
        figura plotly
    """
    if plot_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color, title=title)
    
    elif plot_type == "line":
        fig = px.line(df, x=x_col, y=y_col, color=color, title=title)
    
    elif plot_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color, title=title)
    
    elif plot_type == "histogram":
        fig = px.histogram(df, x=x_col, color=color, title=title)
    
    elif plot_type == "box":
        fig = px.box(df, x=x_col, y=y_col, color=color, title=title)
    
    elif plot_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, title=title)
    
    elif plot_type == "heatmap":
        pivot_df = df.pivot_table(values=y_col, index=x_col, columns=color, aggfunc='mean')
        fig = px.imshow(pivot_df, title=title)
    
    else:
        return None
    
    fig.update_layout(title=title)
    return fig

def normalize_data(df, columns, method='minmax'):
    """
    Normaliza colunas selecionadas no dataframe
    
    Args:
        df: pandas DataFrame
        columns: lista de colunas para normalizar
        method: método de normalização ('minmax', 'zscore', 'robust')
    
    Returns:
        DataFrame com colunas normalizadas
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            if method == 'minmax':
                scaler = preprocessing.MinMaxScaler()
            elif method == 'zscore':
                scaler = preprocessing.StandardScaler()
            elif method == 'robust':
                scaler = preprocessing.RobustScaler()
            else:
                continue
                
            # Reshape and fit transform
            result_df[f"{col}_normalized"] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    
    return result_df
