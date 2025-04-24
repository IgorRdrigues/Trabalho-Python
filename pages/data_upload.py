import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Adicionar diretório pai ao caminho para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, get_column_types

st.title("📤 Upload de Dados")
st.markdown("Carregue seu arquivo de dados (CSV ou Excel) para começar a análise.")

# Carregador de arquivos
uploaded_file = st.file_uploader("Escolha um arquivo", type=["csv", "xlsx", "xls"])

# Configurar container para opções
options_container = st.container()

with options_container:
    if uploaded_file is not None:
        # Carregar os dados
        data, filename = load_data(uploaded_file)
        
        if data is not None:
            # Pré-visualização dos dados
            st.subheader("Pré-visualização dos Dados")
            st.dataframe(data.head(5))
            
            # Informações dos dados
            st.subheader("Informações dos Dados")
            
            # Exibir informações básicas em colunas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Linhas", data.shape[0])
            with col2:
                st.metric("Colunas", data.shape[1])
            with col3:
                st.metric("Valores Ausentes", data.isna().sum().sum())
            
            # Tipos de dados das colunas
            st.subheader("Tipos de Dados das Colunas")
            dtypes_df = pd.DataFrame({
                'Coluna': data.columns,
                'Tipo de Dado': [str(data[col].dtype) for col in data.columns],
                'Contagem de Não-Nulos': [data[col].count() for col in data.columns],
                'Valores Ausentes': [data[col].isna().sum() for col in data.columns]
            })
            st.dataframe(dtypes_df)
            
            # Opções de pré-processamento de dados
            st.subheader("Opções de Pré-processamento")
            
            preprocess_options = st.multiselect(
                "Selecione as opções de pré-processamento:",
                ["Tratar valores ausentes", "Remover duplicatas", "Converter tipos de dados"]
            )
            
            modified_data = data.copy()
            
            # Tratar valores ausentes
            if "Tratar valores ausentes" in preprocess_options:
                st.write("Tratar Valores Ausentes")
                
                missing_cols = [col for col in data.columns if data[col].isna().any()]
                if missing_cols:
                    for col in missing_cols:
                        st.write(f"Coluna: {col} - {data[col].isna().sum()} valores ausentes")
                        
                        strategy = st.selectbox(
                            f"Como tratar valores ausentes em {col}?",
                            ["Remover linhas", "Preencher com média", "Preencher com mediana", "Preencher com moda", "Preencher com valor"],
                            key=f"missing_{col}"
                        )
                        
                        if strategy == "Remover linhas":
                            modified_data = modified_data.dropna(subset=[col])
                        elif strategy == "Preencher com média" and pd.api.types.is_numeric_dtype(data[col]):
                            modified_data[col] = modified_data[col].fillna(modified_data[col].mean())
                        elif strategy == "Preencher com mediana" and pd.api.types.is_numeric_dtype(data[col]):
                            modified_data[col] = modified_data[col].fillna(modified_data[col].median())
                        elif strategy == "Preencher com moda":
                            modified_data[col] = modified_data[col].fillna(modified_data[col].mode()[0])
                        elif strategy == "Preencher com valor":
                            fill_value = st.text_input(f"Digite o valor para preencher em {col}", key=f"fill_{col}")
                            if fill_value:
                                # Tentar converter o valor para o tipo de dados da coluna
                                try:
                                    if pd.api.types.is_numeric_dtype(data[col]):
                                        fill_value = float(fill_value)
                                    modified_data[col] = modified_data[col].fillna(fill_value)
                                except ValueError:
                                    st.error(f"O valor não pôde ser convertido para o tipo de dados da coluna.")
                else:
                    st.write("Não foram encontrados valores ausentes no conjunto de dados.")
            
            # Remover duplicatas
            if "Remover duplicatas" in preprocess_options:
                st.write("Remover Duplicatas")
                
                duplicate_rows = modified_data.duplicated().sum()
                if duplicate_rows > 0:
                    st.write(f"Encontradas {duplicate_rows} linhas duplicadas")
                    
                    subset_cols = st.multiselect(
                        "Selecione colunas para identificar duplicatas (deixe vazio para usar todas):",
                        modified_data.columns.tolist(),
                        key="duplicate_cols"
                    )
                    
                    keep_option = st.radio(
                        "Qual duplicata manter?",
                        ["primeira", "última", "nenhuma"],
                        key="duplicate_keep"
                    )
                    
                    # Mapear escolhas em português para as opções em inglês
                    keep_map = {"primeira": "first", "última": "last", "nenhuma": "none"}
                    
                    if st.button("Remover Duplicatas"):
                        if subset_cols:
                            modified_data = modified_data.drop_duplicates(subset=subset_cols, keep=keep_map[keep_option])
                        else:
                            modified_data = modified_data.drop_duplicates(keep=keep_map[keep_option])
                        st.success(f"Removidas {duplicate_rows} linhas duplicadas")
                else:
                    st.write("Não foram encontradas linhas duplicadas no conjunto de dados.")
            
            # Converter tipos de dados
            if "Converter tipos de dados" in preprocess_options:
                st.write("Converter Tipos de Dados")
                
                for col in modified_data.columns:
                    current_type = modified_data[col].dtype
                    st.write(f"Coluna: {col} - Tipo atual: {current_type}")
                    
                    target_type = st.selectbox(
                        f"Converter {col} para:",
                        ["Manter atual", "numérico", "texto", "categoria", "data/hora"],
                        key=f"convert_{col}"
                    )
                    
                    # Mapear escolhas em português para as opções em inglês
                    type_map = {
                        "Manter atual": "Keep current",
                        "numérico": "numeric", 
                        "texto": "text", 
                        "categoria": "category", 
                        "data/hora": "datetime"
                    }
                    
                    if target_type != "Manter atual":
                        try:
                            if type_map[target_type] == "numeric":
                                modified_data[col] = pd.to_numeric(modified_data[col], errors='coerce')
                            elif type_map[target_type] == "text":
                                modified_data[col] = modified_data[col].astype(str)
                            elif type_map[target_type] == "category":
                                modified_data[col] = modified_data[col].astype('category')
                            elif type_map[target_type] == "datetime":
                                date_format = st.text_input(
                                    f"Digite o formato da data para {col} (ex: '%Y-%m-%d', deixe vazio para detecção automática):",
                                    key=f"date_format_{col}"
                                )
                                if date_format:
                                    modified_data[col] = pd.to_datetime(modified_data[col], format=date_format, errors='coerce')
                                else:
                                    modified_data[col] = pd.to_datetime(modified_data[col], errors='coerce')
                        except Exception as e:
                            st.error(f"Erro ao converter {col}: {str(e)}")
            
            # Botão para salvar alterações
            if st.button("Aplicar Alterações e Continuar"):
                # Calcular tipos de colunas
                columns, numeric_columns, categorical_columns = get_column_types(modified_data)
                
                # Atualizar estado da sessão
                st.session_state.data = modified_data
                st.session_state.filename = filename
                st.session_state.columns = columns
                st.session_state.numeric_columns = numeric_columns
                st.session_state.categorical_columns = categorical_columns
                
                st.success("Dados carregados com sucesso. Agora você pode prosseguir para análise e visualização.")
                
                # Mostrar amostra dos dados processados
                st.subheader("Pré-visualização dos Dados Processados")
                st.dataframe(modified_data.head(5))
                
                # Fornecer links de navegação
                st.markdown("### Navegar para:")
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("[![Análise](https://img.shields.io/badge/Ir%20para-Análise-blue?style=for-the-badge)](Data_Analysis)")
                with cols[1]:
                    st.markdown("[![Visualização](https://img.shields.io/badge/Ir%20para-Visualização-green?style=for-the-badge)](Data_Visualization)")
                with cols[2]:
                    st.markdown("[![Exportar](https://img.shields.io/badge/Ir%20para-Exportar-orange?style=for-the-badge)](Data_Export)")
    else:
        st.info("Por favor, carregue um arquivo CSV ou Excel para começar.")
        
        # Instruções para usar o aplicativo
        st.markdown("""
        ### Instruções:
        1. Carregue seu arquivo de dados em formato CSV ou Excel
        2. Visualize os dados e verifique se há problemas
        3. Aplique opções de pré-processamento, se necessário
        4. Quando seus dados estiverem prontos, clique em 'Aplicar Alterações e Continuar'
        5. Navegue para outras seções para analisar, visualizar e exportar seus dados
        
        ### Formatos de Arquivo Suportados:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        """)
