import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
import os
from scipy import stats

# Adicionar diretório pai ao caminho para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import calculate_basic_stats, calculate_group_stats, filter_dataframe, normalize_data

st.title("📊 Análise de Dados")

# Verificar se os dados foram carregados
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Por favor, carregue seus dados primeiro usando a seção Upload de Dados.")
    st.stop()

# Obter dados do estado da sessão
data = st.session_state.data
columns = st.session_state.columns
numeric_columns = st.session_state.numeric_columns
categorical_columns = st.session_state.categorical_columns

# Barra lateral para opções de análise
st.sidebar.header("Opções de Análise")
analysis_type = st.sidebar.selectbox(
    "Selecione o Tipo de Análise",
    ["Estatísticas Básicas", "Análise de Grupos", "Filtragem de Dados", "Análise Avançada"]
)

# Análise de Estatísticas Básicas
if analysis_type == "Estatísticas Básicas":
    st.header("Análise Estatística Básica")
    
    # Selecionar coluna para análise
    selected_column = st.selectbox(
        "Selecione uma coluna para análise:",
        numeric_columns
    )
    
    if selected_column:
        # Calcular estatísticas
        stats = calculate_basic_stats(data, selected_column)
        
        if stats:
            # Exibir estatísticas em um formato agradável
            st.subheader(f"Estatísticas para {selected_column}")
            
            # Criar múltiplas colunas para melhor layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Média", round(stats['mean'], 2))
                st.metric("Mínimo", round(stats['min'], 2))
                st.metric("Percentil 25%", round(stats['25%'], 2))
            
            with col2:
                st.metric("Mediana", round(stats['median'], 2))
                st.metric("Máximo", round(stats['max'], 2))
                st.metric("Percentil 75%", round(stats['75%'], 2))
            
            with col3:
                st.metric("Desvio Padrão", round(stats['std'], 2))
                st.metric("Contagem", stats['count'])
                st.metric("IQR", round(stats['IQR'], 2))
            
            # Exibir histograma
            st.subheader(f"Distribuição de {selected_column}")
            hist_values = data[selected_column].dropna()
            
            # Calcular número de bins usando a regra de Sturges
            num_bins = int(np.ceil(np.log2(len(hist_values)) + 1))
            
            # Plotar histograma
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(hist_values, bins=num_bins)
            ax.set_title(f"Histograma de {selected_column}")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequência")
            st.pyplot(fig)
            
            # Exibir boxplot
            st.subheader(f"Boxplot de {selected_column}")
            fig, ax = plt.subplots(figsize=(10, 6))
            data.boxplot(column=[selected_column], ax=ax)
            st.pyplot(fig)
            
            # Identificar potenciais outliers
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = stats['IQR']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data[(data[selected_column] < lower_bound) | (data[selected_column] > upper_bound)]
            
            if not outliers.empty:
                st.subheader("Potenciais Outliers")
                st.write(f"Encontrados {len(outliers)} potenciais outliers (valores fora do intervalo [{round(lower_bound, 2)}, {round(upper_bound, 2)}])")
                st.dataframe(outliers)
        else:
            st.error("Não foi possível calcular estatísticas para a coluna selecionada.")

# Análise de Grupos
elif analysis_type == "Análise de Grupos":
    st.header("Análise de Grupos")
    
    # Selecionar colunas para agrupamento e análise
    group_col = st.selectbox(
        "Selecione uma coluna para agrupar por:",
        categorical_columns
    )
    
    analysis_col = st.selectbox(
        "Selecione uma coluna numérica para analisar:",
        numeric_columns
    )
    
    if group_col and analysis_col:
        # Calcular estatísticas de grupo
        group_stats = calculate_group_stats(data, analysis_col, group_col)
        
        if group_stats is not None:
            st.subheader(f"Estatísticas de Grupo de {analysis_col} por {group_col}")
            
            # Renomear colunas para português
            group_stats.columns = [group_col, 'Contagem', 'Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
            st.dataframe(group_stats)
            
            # Plotar estatísticas de grupo
            st.subheader("Comparação de Grupos")
            
            chart_type = st.radio(
                "Selecione o tipo de gráfico:",
                ["Gráfico de Barras", "Boxplot"]
            )
            
            if chart_type == "Gráfico de Barras":
                # Gráfico de barras para valores médios por grupo
                fig, ax = plt.subplots(figsize=(10, 6))
                data.groupby(group_col)[analysis_col].mean().plot(kind='bar', ax=ax)
                ax.set_title(f"Média de {analysis_col} por {group_col}")
                ax.set_ylabel(analysis_col)
                st.pyplot(fig)
                
            elif chart_type == "Boxplot":
                # Boxplot para distribuição por grupo
                fig, ax = plt.subplots(figsize=(10, 6))
                data.boxplot(column=analysis_col, by=group_col, ax=ax)
                plt.title(f"{analysis_col} por {group_col}")
                plt.suptitle("")  # Remover título padrão
                st.pyplot(fig)
            
            # Teste ANOVA se houver mais de 2 grupos (implementação simples)
            groups = data.groupby(group_col)[analysis_col].apply(list).values
            
            if len(groups) > 1:
                try:
                    f_val, p_val = stats.f_oneway(*groups)
                    
                    st.subheader("Teste ANOVA de uma via")
                    st.write(f"Estatística F: {f_val:.4f}")
                    st.write(f"Valor p: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.success("Há uma diferença estatisticamente significativa entre os grupos (p < 0,05).")
                    else:
                        st.info("Não há diferença estatisticamente significativa entre os grupos (p >= 0,05).")
                except:
                    st.warning("Não foi possível realizar o teste ANOVA nos grupos selecionados.")
        else:
            st.error("Não foi possível realizar análise de grupo nas colunas selecionadas.")

# Filtragem de Dados
elif analysis_type == "Filtragem de Dados":
    st.header("Filtragem e Exploração de Dados")
    
    # Criar condições de filtro
    st.subheader("Definir Condições de Filtro")
    
    filters = []
    
    # Adicionar filtros dinamicamente
    num_filters = st.number_input("Número de condições de filtro:", min_value=0, max_value=10, value=1)
    
    for i in range(int(num_filters)):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_col = st.selectbox(f"Coluna {i+1}:", columns, key=f"filter_col_{i}")
        
        with col2:
            # Determinar operadores adequados com base no tipo de coluna
            if filter_col in numeric_columns:
                operators = ["igual a", "diferente de", "maior que", "menor que"]
            else:
                operators = ["igual a", "diferente de", "contém"]
            
            filter_op = st.selectbox(f"Operador {i+1}:", operators, key=f"filter_op_{i}")
            
            # Mapear operadores em português para inglês
            op_map = {
                "igual a": "equals", 
                "diferente de": "not equals", 
                "maior que": "greater than", 
                "menor que": "less than",
                "contém": "contains"
            }
        
        with col3:
            # Determinar tipo de entrada com base no tipo de coluna
            if filter_col in numeric_columns:
                filter_val = st.number_input(f"Valor {i+1}:", key=f"filter_val_{i}")
            elif filter_col in categorical_columns:
                filter_val = st.selectbox(f"Valor {i+1}:", data[filter_col].dropna().unique(), key=f"filter_val_{i}")
            else:
                filter_val = st.text_input(f"Valor {i+1}:", key=f"filter_val_{i}")
        
        filters.append((filter_col, op_map[filter_op], filter_val))
    
    # Aplicar filtros
    if st.button("Aplicar Filtros"):
        filtered_data = filter_dataframe(data, filters)
        
        st.subheader("Dados Filtrados")
        st.write(f"Mostrando {len(filtered_data)} de {len(data)} linhas ({round(len(filtered_data)/len(data)*100, 2)}%)")
        st.dataframe(filtered_data)
        
        # Baixar dados filtrados
        if not filtered_data.empty:
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                "Baixar Dados Filtrados como CSV",
                csv,
                "dados_filtrados.csv",
                "text/csv",
                key='download-csv'
            )
    
    # Opções de exploração de dados
    st.subheader("Exploração de Dados")
    
    # Mostrar valores únicos para colunas categóricas
    if categorical_columns:
        explore_col = st.selectbox(
            "Explorar valores únicos e contagens para coluna categórica:",
            categorical_columns
        )
        
        if explore_col:
            value_counts = data[explore_col].value_counts().reset_index()
            value_counts.columns = [explore_col, 'Contagem']
            
            st.write(f"Valores únicos em {explore_col}:")
            st.dataframe(value_counts)
            
            # Visualizar a distribuição
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(value_counts[explore_col], value_counts['Contagem'])
            ax.set_title(f"Distribuição de {explore_col}")
            ax.set_xlabel(explore_col)
            ax.set_ylabel("Contagem")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# Análise Avançada
elif analysis_type == "Análise Avançada":
    st.header("Análise Avançada")
    
    advanced_option = st.selectbox(
        "Selecione o Método de Análise Avançada:",
        ["Análise de Correlação", "Normalização/Escala", "Análise de Componentes Principais (PCA)", "Clustering"]
    )
    
    # Análise de Correlação
    if advanced_option == "Análise de Correlação":
        st.subheader("Análise de Correlação")
        
        if len(numeric_columns) < 2:
            st.warning("Necessário pelo menos 2 colunas numéricas para análise de correlação.")
        else:
            # Selecionar colunas para correlação
            corr_columns = st.multiselect(
                "Selecione colunas para análise de correlação:",
                numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if corr_columns and len(corr_columns) >= 2:
                # Calcular matriz de correlação
                corr_matrix = data[corr_columns].corr()
                
                # Exibir matriz de correlação
                st.write("Matriz de Correlação:")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
                
                # Visualizar matriz de correlação como mapa de calor
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
                
                # Encontrar e exibir pares altamente correlacionados
                high_corr_threshold = st.slider(
                    "Limiar de alta correlação:",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.7,
                    step=0.05
                )
                
                # Obter pares com alta correlação
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                            corr_pairs.append({
                                'Variável 1': corr_matrix.columns[i],
                                'Variável 2': corr_matrix.columns[j],
                                'Correlação': corr_matrix.iloc[i, j]
                            })
                
                if corr_pairs:
                    st.write(f"Pares altamente correlacionados (|correlação| >= {high_corr_threshold}):")
                    st.table(pd.DataFrame(corr_pairs))
                    
                    # Gráfico de dispersão para par selecionado
                    if len(corr_pairs) > 0:
                        st.subheader("Gráfico de Dispersão para Variáveis Correlacionadas")
                        
                        selected_pair = st.selectbox(
                            "Selecione um par para visualizar:",
                            [f"{pair['Variável 1']} vs {pair['Variável 2']} (corr: {pair['Correlação']:.2f})" 
                             for pair in corr_pairs]
                        )
                        
                        if selected_pair:
                            var1 = selected_pair.split(" vs ")[0]
                            var2 = selected_pair.split(" vs ")[1].split(" (corr:")[0]
                            
                            # Criar gráfico de dispersão
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(data[var1], data[var2], alpha=0.5)
                            ax.set_xlabel(var1)
                            ax.set_ylabel(var2)
                            ax.set_title(f"Gráfico de Dispersão: {var1} vs {var2}")
                            
                            # Adicionar linha de tendência
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                data[var1].dropna(), 
                                data[var2].dropna()
                            )
                            
                            x = np.array([data[var1].min(), data[var1].max()])
                            y = intercept + slope * x
                            ax.plot(x, y, 'r')
                            
                            st.pyplot(fig)
                else:
                    st.info(f"Nenhum par de variáveis com correlação >= {high_corr_threshold} encontrado.")
    
    # Normalização/Escala
    elif advanced_option == "Normalização/Escala":
        st.subheader("Normalização/Escala de Dados")
        
        # Selecionar colunas para normalizar
        norm_columns = st.multiselect(
            "Selecione colunas numéricas para normalizar:",
            numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))]
        )
        
        if norm_columns:
            # Selecionar método de normalização
            norm_method = st.radio(
                "Selecione o método de normalização:",
                ["Escala Min-Max (0-1)", "Padronização Z-Score", "Escala Robusta"]
            )
            
            # Mapear seleção para nome do método
            method_map = {
                "Escala Min-Max (0-1)": "minmax",
                "Padronização Z-Score": "zscore",
                "Escala Robusta": "robust"
            }
            
            # Normalizar dados
            if st.button("Aplicar Normalização"):
                normalized_data = normalize_data(data, norm_columns, method_map[norm_method])
                
                # Mostrar dados originais vs normalizados
                st.write("Dados Originais vs. Normalizados (primeiras 10 linhas):")
                
                # Extrair apenas as colunas relevantes
                display_cols = norm_columns + [f"{col}_normalized" for col in norm_columns]
                st.dataframe(normalized_data[display_cols].head(10))
                
                # Visualização do efeito da normalização
                col1, col2 = st.columns(2)
                
                if len(norm_columns) > 0:
                    selected_col = norm_columns[0]
                    
                    with col1:
                        st.write(f"Distribuição Original de {selected_col}")
                        fig, ax = plt.subplots()
                        ax.hist(data[selected_col].dropna(), bins=20)
                        st.pyplot(fig)
                    
                    with col2:
                        st.write(f"Distribuição Normalizada de {selected_col}")
                        fig, ax = plt.subplots()
                        ax.hist(normalized_data[f"{selected_col}_normalized"].dropna(), bins=20)
                        st.pyplot(fig)
    
    # Análise de Componentes Principais (PCA)
    elif advanced_option == "Análise de Componentes Principais (PCA)":
        st.subheader("Análise de Componentes Principais (PCA)")
        
        # Verificar se há colunas numéricas suficientes
        if len(numeric_columns) < 3:
            st.warning("PCA requer pelo menos 3 colunas numéricas para ser útil. Adicione mais dados numéricos.")
        else:
            # Selecionar colunas para PCA
            pca_columns = st.multiselect(
                "Selecione colunas numéricas para PCA:",
                numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if pca_columns and len(pca_columns) >= 2:
                # Opções para PCA
                n_components = st.slider(
                    "Número de componentes principais:",
                    min_value=2,
                    max_value=min(len(pca_columns), 10),
                    value=2
                )
                
                # Preparar dados para PCA
                pca_data = data[pca_columns].dropna()
                
                if len(pca_data) < 2:
                    st.error("Dados insuficientes para PCA após remover valores ausentes.")
                else:
                    # Normalizar dados antes do PCA
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pca_data)
                    
                    # Aplicar PCA
                    pca = PCA(n_components=n_components)
                    principal_components = pca.fit_transform(scaled_data)
                    
                    # Criar DataFrame com componentes principais
                    pca_df = pd.DataFrame(
                        data=principal_components,
                        columns=[f'PC{i+1}' for i in range(n_components)]
                    )
                    
                    # Exibir resultado
                    st.subheader("Componentes Principais")
                    st.dataframe(pca_df.head(10))
                    
                    # Variância explicada
                    explained_variance = pca.explained_variance_ratio_ * 100
                    cumulative_variance = np.cumsum(explained_variance)
                    
                    st.subheader("Variância Explicada")
                    
                    # Criar DataFrame de variância
                    variance_df = pd.DataFrame({
                        'Componente': [f'PC{i+1}' for i in range(n_components)],
                        'Variância Explicada (%)': explained_variance,
                        'Variância Acumulada (%)': cumulative_variance
                    })
                    
                    st.dataframe(variance_df)
                    
                    # Plotar variância explicada
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(variance_df['Componente'], variance_df['Variância Explicada (%)'])
                    ax.set_ylabel('Variância Explicada (%)')
                    ax.set_title('Variância Explicada por Componente Principal')
                    
                    # Adicionar linha de variância acumulada
                    ax2 = ax.twinx()
                    ax2.plot(variance_df['Componente'], variance_df['Variância Acumulada (%)'], 'r-o')
                    ax2.set_ylabel('Variância Acumulada (%)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    st.pyplot(fig)
                    
                    # Visualizar projeção nos dois primeiros componentes principais
                    if n_components >= 2:
                        st.subheader("Visualização dos Dois Primeiros Componentes Principais")
                        
                        # Opção para colorir pontos
                        if categorical_columns:
                            color_column = st.selectbox(
                                "Colorir pontos por (opcional):",
                                ["Nenhum"] + categorical_columns
                            )
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            if color_column != "Nenhum":
                                # Adicionar coluna de categoria ao DataFrame PCA
                                color_data = data[color_column].iloc[pca_data.index]
                                scatter = ax.scatter(
                                    pca_df['PC1'], 
                                    pca_df['PC2'],
                                    c=pd.factorize(color_data)[0],
                                    alpha=0.7,
                                    cmap='viridis'
                                )
                                
                                # Adicionar legenda
                                legend_handles = []
                                for i, category in enumerate(color_data.unique()):
                                    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                       markerfacecolor=plt.cm.viridis(i / len(color_data.unique())), 
                                                       markersize=10, label=category))
                                
                                ax.legend(handles=legend_handles, title=color_column)
                            else:
                                ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
                            
                            ax.set_xlabel('Componente Principal 1')
                            ax.set_ylabel('Componente Principal 2')
                            ax.set_title('Projeção PCA em 2D')
                            ax.grid(True)
                            
                            st.pyplot(fig)
                        
                        # Mostrar loadings (contribuições das variáveis originais)
                        st.subheader("Contribuições das Variáveis Originais (Loadings)")
                        
                        loadings = pd.DataFrame(
                            pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=pca_columns
                        )
                        
                        st.dataframe(loadings)
                        
                        # Visualizar loadings para os dois primeiros componentes
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Criar mapa de calor para os loadings
                        sns.heatmap(
                            loadings.iloc[:, :2],
                            annot=True,
                            cmap='coolwarm',
                            center=0,
                            ax=ax
                        )
                        
                        ax.set_title('Contribuições das Variáveis para PC1 e PC2')
                        st.pyplot(fig)
    
    # Clustering
    elif advanced_option == "Clustering":
        st.subheader("Análise de Clustering")
        
        # Verificar se há colunas numéricas suficientes
        if len(numeric_columns) < 2:
            st.warning("Clustering requer pelo menos 2 colunas numéricas.")
        else:
            # Selecionar algoritmo de clustering
            cluster_algorithm = st.selectbox(
                "Selecione o algoritmo de clustering:",
                ["K-Means"]
            )
            
            # Selecionar colunas para clustering
            cluster_columns = st.multiselect(
                "Selecione colunas numéricas para clustering:",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )
            
            if cluster_columns and len(cluster_columns) >= 2:
                # Preparar dados para clustering
                cluster_data = data[cluster_columns].dropna()
                
                if len(cluster_data) < 2:
                    st.error("Dados insuficientes para clustering após remover valores ausentes.")
                else:
                    # K-Means Clustering
                    if cluster_algorithm == "K-Means":
                        # Configurar número de clusters
                        n_clusters = st.slider("Número de clusters (k):", 2, 10, 3)
                        
                        # Normalizar dados antes do clustering
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(cluster_data)
                        
                        # Aplicar K-Means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(scaled_data)
                        
                        # Adicionar labels de cluster aos dados
                        result_df = cluster_data.copy()
                        result_df['Cluster'] = cluster_labels
                        
                        # Exibir resultados
                        st.subheader("Resultados do Clustering")
                        st.write(f"Número de clusters: {n_clusters}")
                        
                        # Contagem de amostras por cluster
                        cluster_counts = pd.DataFrame(
                            result_df['Cluster'].value_counts().sort_index()
                        ).reset_index()
                        cluster_counts.columns = ['Cluster', 'Contagem']
                        
                        st.write("Contagem de amostras por cluster:")
                        st.dataframe(cluster_counts)
                        
                        # Visualizar clusters
                        if len(cluster_columns) >= 2:
                            st.subheader("Visualização de Clusters")
                            
                            if len(cluster_columns) == 2:
                                # Visualização direta para 2 dimensões
                                x_col, y_col = cluster_columns
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                scatter = ax.scatter(
                                    result_df[x_col],
                                    result_df[y_col],
                                    c=result_df['Cluster'],
                                    cmap='viridis',
                                    alpha=0.7
                                )
                                
                                # Adicionar centróides
                                centroids = kmeans.cluster_centers_
                                centroids_df = pd.DataFrame(
                                    scaler.inverse_transform(centroids),
                                    columns=cluster_columns
                                )
                                
                                ax.scatter(
                                    centroids_df[x_col],
                                    centroids_df[y_col],
                                    marker='X',
                                    s=200,
                                    color='red',
                                    label='Centróides'
                                )
                                
                                ax.set_xlabel(x_col)
                                ax.set_ylabel(y_col)
                                ax.set_title(f'Clusters K-Means: {x_col} vs {y_col}')
                                ax.legend()
                                
                                st.pyplot(fig)
                            else:
                                # Para mais de 2 dimensões, oferecer diferentes visualizações
                                st.write("Selecione 2 dimensões para visualizar:")
                                x_col = st.selectbox("Eixo X:", cluster_columns, index=0)
                                y_col = st.selectbox("Eixo Y:", cluster_columns, index=1)
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                scatter = ax.scatter(
                                    result_df[x_col],
                                    result_df[y_col],
                                    c=result_df['Cluster'],
                                    cmap='viridis',
                                    alpha=0.7
                                )
                                
                                # Adicionar centróides (projetos nas 2 dimensões selecionadas)
                                centroids = kmeans.cluster_centers_
                                centroids_df = pd.DataFrame(
                                    scaler.inverse_transform(centroids),
                                    columns=cluster_columns
                                )
                                
                                ax.scatter(
                                    centroids_df[x_col],
                                    centroids_df[y_col],
                                    marker='X',
                                    s=200,
                                    color='red',
                                    label='Centróides'
                                )
                                
                                ax.set_xlabel(x_col)
                                ax.set_ylabel(y_col)
                                ax.set_title(f'Clusters K-Means: {x_col} vs {y_col}')
                                ax.legend()
                                
                                st.pyplot(fig)
                            
                            # Visualizar estatísticas por cluster
                            st.subheader("Estatísticas por Cluster")
                            
                            # Calcular estatísticas para cada cluster
                            cluster_stats = result_df.groupby('Cluster')[cluster_columns].mean()
                            
                            # Plotar estatísticas por cluster
                            fig, ax = plt.subplots(figsize=(12, 6))
                            cluster_stats.T.plot(kind='bar', ax=ax)
                            ax.set_xlabel('Atributos')
                            ax.set_ylabel('Valor Médio')
                            ax.set_title('Valores Médios dos Atributos por Cluster')
                            ax.legend(title='Cluster')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Radar chart para cada cluster
                            st.subheader("Perfis de Cluster (Gráfico Radar)")
                            
                            # Normalizar estatísticas para radar chart
                            radar_stats = cluster_stats.copy()
                            for col in radar_stats.columns:
                                radar_stats[col] = (radar_stats[col] - radar_stats[col].min()) / (radar_stats[col].max() - radar_stats[col].min())
                            
                            # Plotar radar chart
                            fig = plt.figure(figsize=(12, 8))
                            
                            # Configurações para radar chart
                            categories = cluster_columns
                            N = len(categories)
                            
                            # Ângulos para radar chart
                            angles = [n / float(N) * 2 * np.pi for n in range(N)]
                            angles += angles[:1]  # Fechar o círculo
                            
                            # Criar subplot com projeção polar
                            ax = plt.subplot(111, polar=True)
                            
                            # Adicionar labels
                            plt.xticks(angles[:-1], categories, size=12)
                            
                            # Adicionar linhas para cada cluster
                            for cluster in range(n_clusters):
                                values = radar_stats.iloc[cluster].values.tolist()
                                values += values[:1]  # Fechar o círculo
                                
                                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
                                ax.fill(angles, values, alpha=0.1)
                            
                            plt.title('Perfis de Cluster', size=20)
                            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                            
                            st.pyplot(fig)