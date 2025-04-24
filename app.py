import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Aplicativo de Análise de Dados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização das variáveis de estado da sessão
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

# Cabeçalho da página principal
st.title("📊 Aplicativo de Análise e Visualização de Dados")
st.markdown("""
Este aplicativo permite carregar, analisar, visualizar e exportar dados.
Use a barra lateral para navegar pelas diferentes seções.
""")

# Verificar se os dados foram carregados
if st.session_state.data is not None:
    st.success(f"Dados carregados: {st.session_state.filename}")
    
    # Exibir amostra de dados
    st.subheader("Amostra de Dados")
    st.dataframe(st.session_state.data.head(10))
    
    # Exibir informações sobre os dados
    st.subheader("Informações dos Dados")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Linhas", st.session_state.data.shape[0])
    with col2:
        st.metric("Colunas", st.session_state.data.shape[1])
    with col3:
        st.metric("Valores Ausentes", st.session_state.data.isna().sum().sum())
    
    # Fornecer links para outras seções se os dados estiverem carregados
    st.subheader("O que você gostaria de fazer?")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("[![Análise](https://img.shields.io/badge/Ir%20para-Análise-blue?style=for-the-badge)](Data_Analysis)")
    with cols[1]:
        st.markdown("[![Visualização](https://img.shields.io/badge/Ir%20para-Visualização-green?style=for-the-badge)](Data_Visualization)")
    with cols[2]:
        st.markdown("[![Exportar](https://img.shields.io/badge/Ir%20para-Exportar-orange?style=for-the-badge)](Data_Export)")
else:
    # Instruções quando não há dados carregados
    st.info("Por favor, carregue seu arquivo de dados usando a seção Upload na barra lateral.")
    
    # Criar estrutura de pastas se não existir
    Path("pages").mkdir(exist_ok=True)
    
    # Demonstração rápida com dados de exemplo
    if st.button("Carregar Dados de Exemplo para Demonstração"):
        sample_data = pd.DataFrame({
            'Categoria': ['A', 'B', 'A', 'C', 'B', 'C'],
            'Valor1': [10, 20, 15, 25, 30, 35],
            'Valor2': [100, 200, 150, 250, 300, 350],
            'Data': pd.date_range(start='2023-01-01', periods=6)
        })
        
        st.session_state.data = sample_data
        st.session_state.filename = "dados_exemplo.csv"
        st.session_state.columns = sample_data.columns.tolist()
        st.session_state.numeric_columns = sample_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.session_state.categorical_columns = sample_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.success("Dados de exemplo carregados com sucesso!")
        st.rerun()

# Navegação na barra lateral
st.sidebar.title("Navegação")
st.sidebar.markdown("---")
st.sidebar.info("Carregue seus dados e explore as diferentes seções para analisar e visualizar.")
