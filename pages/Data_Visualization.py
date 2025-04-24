import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Adicionar diretório pai ao caminho para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_matplotlib_figure, create_plotly_figure

st.title("📈 Visualização de Dados")

# Verificar se os dados foram carregados
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Por favor, carregue seus dados primeiro usando a seção Upload de Dados.")
    st.stop()

# Obter dados do estado da sessão
data = st.session_state.data
columns = st.session_state.columns
numeric_columns = st.session_state.numeric_columns
categorical_columns = st.session_state.categorical_columns

# Barra lateral para opções de visualização
st.sidebar.header("Opções de Visualização")
visualization_type = st.sidebar.selectbox(
    "Selecione o Tipo de Visualização",
    ["Gráficos Básicos", "Gráficos Estatísticos", "Gráficos Interativos", "Visualização Personalizada"]
)

# Motor de visualização
viz_engine = st.sidebar.radio(
    "Selecione o Motor de Visualização",
    ["Plotly (Interativo)", "Matplotlib (Estático)"],
    index=0
)

# Gráficos Básicos
if visualization_type == "Gráficos Básicos":
    st.header("Gráficos Básicos")
    
    chart_type = st.selectbox(
        "Selecione o Tipo de Gráfico",
        ["Gráfico de Barras", "Gráfico de Linhas", "Gráfico de Dispersão", "Gráfico de Pizza", "Histograma"]
    )
    
    # Opções de configuração com base no tipo de gráfico
    if chart_type == "Gráfico de Barras":
        x_col = st.selectbox("Selecione o eixo X (Categorias):", categorical_columns)
        y_col = st.selectbox("Selecione o eixo Y (Valores):", numeric_columns)
        color_col = st.selectbox("Selecione a Coluna de Cor (opcional):", ["Nenhuma"] + categorical_columns)
        color = None if color_col == "Nenhuma" else color_col
        
        # Opção de barra horizontal
        horizontal = st.checkbox("Gráfico de Barras Horizontal")
        
        # Criar gráfico
        if x_col and y_col:
            st.subheader(f"Gráfico de Barras: {y_col} por {x_col}")
            
            # Preparar dados
            if color:
                # Para agrupamento por cor, usar plotly diretamente
                fig = px.bar(
                    data, 
                    x=x_col if not horizontal else y_col,
                    y=y_col if not horizontal else x_col,
                    color=color,
                    title=f"{y_col} por {x_col}",
                    orientation='v' if not horizontal else 'h'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Agregar dados para gráfico de barras simples
                agg_data = data.groupby(x_col)[y_col].mean().reset_index()
                
                if viz_engine == "Plotly (Interativo)":
                    fig = px.bar(
                        agg_data, 
                        x=x_col if not horizontal else y_col,
                        y=y_col if not horizontal else x_col,
                        title=f"{y_col} por {x_col}",
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
                    ax.set_title(f"{y_col} por {x_col}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    elif chart_type == "Gráfico de Linhas":
        x_col = st.selectbox("Selecione o eixo X:", columns)
        y_cols = st.multiselect("Selecione o eixo Y (Valores, múltiplos permitidos):", numeric_columns)
        
        # Verificar se a coluna x é datetime
        if x_col in data.columns and pd.api.types.is_datetime64_any_dtype(data[x_col]):
            st.info(f"A coluna '{x_col}' é detectada como data/hora. Os dados serão automaticamente ordenados.")
            sort_by_x = True
        else:
            sort_by_x = st.checkbox("Ordenar por valores do eixo X", value=True)
        
        # Criar gráfico
        if x_col and y_cols:
            st.subheader(f"Gráfico de Linhas: {', '.join(y_cols)} por {x_col}")
            
            # Preparar dados - ordenar se necessário
            plot_data = data.copy()
            if sort_by_x:
                plot_data = plot_data.sort_values(by=x_col)
            
            if viz_engine == "Plotly (Interativo)":
                fig = go.Figure()
                
                for y_col in y_cols:
                    fig.add_trace(go.Scatter(
                        x=plot_data[x_col],
                        y=plot_data[y_col],
                        mode='lines+markers',
                        name=y_col
                    ))
                
                fig.update_layout(
                    title=f"Gráfico de Linhas: {', '.join(y_cols)} por {x_col}",
                    xaxis_title=x_col,
                    yaxis_title="Valor",
                    legend_title="Variáveis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for y_col in y_cols:
                    ax.plot(plot_data[x_col], plot_data[y_col], marker='o', label=y_col)
                
                ax.set_xlabel(x_col)
                ax.set_ylabel("Valor")
                ax.set_title(f"Gráfico de Linhas: {', '.join(y_cols)} por {x_col}")
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    elif chart_type == "Gráfico de Dispersão":
        x_col = st.selectbox("Selecione o eixo X:", numeric_columns)
        y_col = st.selectbox("Selecione o eixo Y:", numeric_columns, index=min(1, len(numeric_columns)-1))
        color_col = st.selectbox("Selecione a Coluna de Cor (opcional):", ["Nenhuma"] + categorical_columns)
        size_col = st.selectbox("Selecione a Coluna de Tamanho (opcional):", ["Nenhuma"] + numeric_columns)
        
        # Opção de linha de tendência
        add_trendline = st.checkbox("Adicionar Linha de Tendência", value=True)
        
        # Criar gráfico
        if x_col and y_col:
            st.subheader(f"Gráfico de Dispersão: {y_col} vs {x_col}")
            
            color = None if color_col == "Nenhuma" else color_col
            size = None if size_col == "Nenhuma" else size_col
            
            if viz_engine == "Plotly (Interativo)":
                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color,
                    size=size,
                    title=f"Gráfico de Dispersão: {y_col} vs {x_col}",
                    trendline="ols" if add_trendline else None
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if color:
                    # Obter categorias únicas e criar gráfico de dispersão para cada uma
                    for cat in data[color].unique():
                        subset = data[data[color] == cat]
                        ax.scatter(subset[x_col], subset[y_col], label=cat, alpha=0.7)
                    ax.legend()
                else:
                    ax.scatter(data[x_col], data[y_col], alpha=0.7)
                
                # Adicionar linha de tendência
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
                ax.set_title(f"Gráfico de Dispersão: {y_col} vs {x_col}")
                
                plt.tight_layout()
                st.pyplot(fig)
    
    elif chart_type == "Gráfico de Pizza":
        value_col = st.selectbox("Selecione a Coluna de Valor:", numeric_columns)
        name_col = st.selectbox("Selecione a Coluna de Categoria:", categorical_columns)
        
        # Criar gráfico
        if value_col and name_col:
            st.subheader(f"Gráfico de Pizza: {value_col} por {name_col}")
            
            # Preparar dados - agregar por categoria
            pie_data = data.groupby(name_col)[value_col].sum().reset_index()
            
            if viz_engine == "Plotly (Interativo)":
                fig = px.pie(
                    pie_data,
                    values=value_col,
                    names=name_col,
                    title=f"Distribuição de {value_col} por {name_col}"
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
                ax.axis('equal')  # Garante que o gráfico de pizza é desenhado como um círculo
                ax.set_title(f"Distribuição de {value_col} por {name_col}")
                st.pyplot(fig)
    
    elif chart_type == "Histograma":
        value_col = st.selectbox("Selecione a Coluna para Histograma:", numeric_columns)
        bins = st.slider("Número de Intervalos:", min_value=5, max_value=100, value=20)
        color_col = st.selectbox("Agrupar Por (opcional):", ["Nenhuma"] + categorical_columns)
        
        # Criar gráfico
        if value_col:
            st.subheader(f"Histograma: {value_col}")
            
            color = None if color_col == "Nenhuma" else color_col
            
            if viz_engine == "Plotly (Interativo)":
                fig = px.histogram(
                    data,
                    x=value_col,
                    color=color,
                    nbins=bins,
                    title=f"Histograma de {value_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if color:
                    # Para cada categoria, criar um histograma
                    for cat in data[color].unique():
                        subset = data[data[color] == cat]
                        ax.hist(subset[value_col], bins=bins, alpha=0.5, label=cat)
                    ax.legend()
                else:
                    ax.hist(data[value_col], bins=bins)
                
                ax.set_xlabel(value_col)
                ax.set_ylabel("Frequência")
                ax.set_title(f"Histograma de {value_col}")
                
                plt.tight_layout()
                st.pyplot(fig)

# Gráficos Estatísticos
elif visualization_type == "Gráficos Estatísticos":
    st.header("Gráficos Estatísticos")
    
    stat_plot_type = st.selectbox(
        "Selecione o Tipo de Gráfico Estatístico",
        ["Boxplot", "Gráfico de Violino", "Gráfico de Densidade", "Gráfico Q-Q", "Mapa de Calor"]
    )
    
    # Boxplot
    if stat_plot_type == "Boxplot":
        y_col = st.selectbox("Selecione a Coluna de Valor:", numeric_columns)
        x_col = st.selectbox("Selecione a Coluna de Agrupamento (opcional):", ["Nenhuma"] + categorical_columns)
        
        x_col = None if x_col == "Nenhuma" else x_col
        
        if y_col:
            st.subheader(f"Boxplot: {y_col}" + (f" por {x_col}" if x_col else ""))
            
            if viz_engine == "Plotly (Interativo)":
                if x_col:
                    fig = px.box(
                        data,
                        x=x_col,
                        y=y_col,
                        title=f"Boxplot de {y_col} por {x_col}"
                    )
                else:
                    fig = px.box(
                        data,
                        y=y_col,
                        title=f"Boxplot de {y_col}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if x_col:
                    data.boxplot(column=y_col, by=x_col, ax=ax)
                    plt.suptitle("")  # Remover título padrão
                    ax.set_title(f"Boxplot de {y_col} por {x_col}")
                else:
                    data.boxplot(column=y_col, ax=ax)
                    ax.set_title(f"Boxplot de {y_col}")
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Gráfico de Violino
    elif stat_plot_type == "Gráfico de Violino":
        y_col = st.selectbox("Selecione a Coluna de Valor:", numeric_columns)
        x_col = st.selectbox("Selecione a Coluna de Agrupamento:", categorical_columns)
        
        if y_col and x_col:
            st.subheader(f"Gráfico de Violino: {y_col} por {x_col}")
            
            if viz_engine == "Plotly (Interativo)":
                fig = px.violin(
                    data,
                    x=x_col,
                    y=y_col,
                    box=True,  # incluir boxplot dentro do violino
                    points="all",  # mostrar todos os pontos
                    title=f"Gráfico de Violino de {y_col} por {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.violinplot(x=x_col, y=y_col, data=data, ax=ax)
                
                ax.set_title(f"Gráfico de Violino de {y_col} por {x_col}")
                plt.tight_layout()
                st.pyplot(fig)
    
    # Gráfico de Densidade
    elif stat_plot_type == "Gráfico de Densidade":
        value_cols = st.multiselect("Selecione as Colunas de Valor:", numeric_columns)
        
        if value_cols:
            st.subheader(f"Gráfico de Densidade: {', '.join(value_cols)}")
            
            if viz_engine == "Plotly (Interativo)":
                fig = go.Figure()
                
                for col in value_cols:
                    # Criar histograma com kde
                    fig.add_trace(go.Histogram(
                        x=data[col],
                        name=col,
                        histnorm='probability density',
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    title=f"Gráfico de Densidade de {', '.join(value_cols)}",
                    xaxis_title="Valor",
                    yaxis_title="Densidade",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for col in value_cols:
                    sns.kdeplot(data[col], ax=ax, label=col)
                
                ax.set_title(f"Gráfico de Densidade de {', '.join(value_cols)}")
                ax.set_xlabel("Valor")
                ax.set_ylabel("Densidade")
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Gráfico Q-Q (Gráfico Quantil-Quantil para verificação de normalidade)
    elif stat_plot_type == "Gráfico Q-Q":
        value_col = st.selectbox("Selecione a Coluna para Gráfico Q-Q:", numeric_columns)
        
        if value_col:
            st.subheader(f"Gráfico Q-Q: {value_col}")
            
            # Preparar dados
            data_no_na = data[value_col].dropna()
            
            if len(data_no_na) < 2:
                st.error("Dados insuficientes para criar o gráfico Q-Q.")
            else:
                from scipy import stats
                
                # Calcular quantis teóricos e de amostra
                quantiles = np.linspace(0.01, 0.99, 100)
                sample_quantiles = np.quantile(data_no_na, quantiles)
                theoretical_quantiles = stats.norm.ppf(quantiles, loc=data_no_na.mean(), scale=data_no_na.std())
                
                if viz_engine == "Plotly (Interativo)":
                    fig = go.Figure()
                    
                    # Adicionar pontos Q-Q
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name=value_col
                    ))
                    
                    # Adicionar linha de referência
                    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Linha de Referência',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Gráfico Q-Q: {value_col}",
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis da Amostra"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Criar gráfico Q-Q
                    stats.probplot(data_no_na, plot=ax)
                    
                    ax.set_title(f"Gráfico Q-Q: {value_col}")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Interpretar o gráfico Q-Q
                st.subheader("Interpretação do Gráfico Q-Q")
                st.write("""
                O gráfico Q-Q (Quantil-Quantil) é usado para verificar se os dados seguem uma distribuição normal:
                - Se os pontos seguem aproximadamente a linha diagonal vermelha, os dados provavelmente seguem uma distribuição normal.
                - Desvios da linha diagonal indicam desvios da normalidade.
                - Curvas em forma de S indicam assimetria.
                - Curvas em forma de U indicam caudas mais pesadas ou mais leves que a distribuição normal.
                """)
                
                # Realizar teste de normalidade
                stat, p_value = stats.shapiro(data_no_na) if len(data_no_na) <= 5000 else stats.normaltest(data_no_na)
                
                st.write(f"Teste de Normalidade (Shapiro-Wilk para n≤5000, D'Agostino-Pearson para n>5000):")
                st.write(f"Estatística do teste: {stat:.4f}")
                st.write(f"Valor p: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.warning("Os dados provavelmente NÃO seguem uma distribuição normal (p < 0,05).")
                else:
                    st.success("Os dados provavelmente seguem uma distribuição normal (p >= 0,05).")
    
    # Mapa de Calor
    elif stat_plot_type == "Mapa de Calor":
        st.subheader("Mapa de Calor de Correlação")
        
        # Selecionar colunas para correlação
        corr_columns = st.multiselect(
            "Selecione as colunas numéricas para o mapa de calor:",
            numeric_columns,
            default=numeric_columns[:min(8, len(numeric_columns))]
        )
        
        if corr_columns and len(corr_columns) >= 2:
            # Calculcular matriz de correlação
            corr_matrix = data[corr_columns].corr()
            
            # Opções para o mapa de calor
            show_values = st.checkbox("Mostrar valores no mapa de calor", value=True)
            use_absolute = st.checkbox("Usar valores absolutos para cor", value=False)
            
            # Selecionar método de correlação
            corr_method = st.selectbox(
                "Método de correlação:",
                ["Pearson", "Spearman", "Kendall"]
            )
            
            # Atualizar matriz de correlação com base no método
            method_map = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}
            corr_matrix = data[corr_columns].corr(method=method_map[corr_method])
            
            # Criar mapa de calor
            if viz_engine == "Plotly (Interativo)":
                # Para valores absolutos
                if use_absolute:
                    z_values = np.abs(corr_matrix)
                    colorscale = 'Reds'
                else:
                    z_values = corr_matrix
                    colorscale = 'RdBu_r'
                
                # Criar mapa de calor com Plotly
                fig = px.imshow(
                    z_values,
                    x=corr_columns,
                    y=corr_columns,
                    color_continuous_scale=colorscale,
                    zmin=-1 if not use_absolute else 0,
                    zmax=1,
                    title=f"Mapa de Calor de Correlação ({corr_method})"
                )
                
                # Adicionar valores ao mapa de calor
                if show_values:
                    for i in range(len(corr_matrix.columns)):
                        for j in range(len(corr_matrix.columns)):
                            fig.add_annotation(
                                x=corr_matrix.columns[j],
                                y=corr_matrix.columns[i],
                                text=f"{corr_matrix.iloc[i, j]:.2f}",
                                showarrow=False,
                                font=dict(color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
                            )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Criar mapa de calor com seaborn
                mask = np.zeros_like(corr_matrix)
                if not st.checkbox("Mostrar valores duplicados", value=True):
                    mask[np.triu_indices_from(mask, k=1)] = True
                
                cmap = "Reds" if use_absolute else "coolwarm"
                sns.heatmap(
                    np.abs(corr_matrix) if use_absolute else corr_matrix,
                    annot=show_values,
                    mask=mask if not show_values else None,
                    cmap=cmap,
                    vmin=-1 if not use_absolute else 0,
                    vmax=1,
                    fmt=".2f",
                    linewidths=0.5,
                    ax=ax
                )
                
                plt.title(f"Mapa de Calor de Correlação ({corr_method})")
                plt.tight_layout()
                st.pyplot(fig)

# Gráficos Interativos
elif visualization_type == "Gráficos Interativos":
    st.header("Gráficos Interativos")
    
    interactive_plot_type = st.selectbox(
        "Selecione o Tipo de Gráfico Interativo",
        ["Gráfico 3D", "Mapa de Bolhas", "Linha do Tempo", "Gráfico Sunburst", "Gráfico de Área"]
    )
    
    # Gráfico 3D
    if interactive_plot_type == "Gráfico 3D":
        st.subheader("Gráfico de Dispersão 3D")
        
        if len(numeric_columns) < 3:
            st.warning("São necessárias pelo menos 3 colunas numéricas para criar um gráfico 3D.")
        else:
            x_col = st.selectbox("Selecione a coluna X:", numeric_columns, index=0)
            y_col = st.selectbox("Selecione a coluna Y:", numeric_columns, index=min(1, len(numeric_columns)-1))
            z_col = st.selectbox("Selecione a coluna Z:", numeric_columns, index=min(2, len(numeric_columns)-1))
            color_col = st.selectbox("Selecione a coluna de cor (opcional):", ["Nenhuma"] + categorical_columns)
            
            color = None if color_col == "Nenhuma" else color_col
            
            # Criar gráfico 3D
            fig = px.scatter_3d(
                data,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color,
                opacity=0.7,
                title=f"Gráfico de Dispersão 3D: {x_col} vs {y_col} vs {z_col}"
            )
            
            # Ajustar layout
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                ),
                width=800,
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Mapa de Bolhas
    elif interactive_plot_type == "Mapa de Bolhas":
        st.subheader("Mapa de Bolhas")
        
        x_col = st.selectbox("Selecione a coluna X:", numeric_columns, index=0)
        y_col = st.selectbox("Selecione a coluna Y:", numeric_columns, index=min(1, len(numeric_columns)-1))
        size_col = st.selectbox("Selecione a coluna de tamanho:", numeric_columns, index=min(2, len(numeric_columns)-1))
        color_col = st.selectbox("Selecione a coluna de cor (opcional):", ["Nenhuma"] + categorical_columns)
        
        color = None if color_col == "Nenhuma" else color_col
        
        # Criar mapa de bolhas
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color,
            hover_name=color if color else None,
            size_max=60,
            opacity=0.7,
            title=f"Mapa de Bolhas: {x_col} vs {y_col} (tamanho por {size_col})"
        )
        
        # Ajustar layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title=color if color else "",
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Linha do Tempo
    elif interactive_plot_type == "Linha do Tempo":
        st.subheader("Linha do Tempo")
        
        # Verificar se há colunas de data
        date_columns = [col for col in columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        
        if not date_columns:
            st.warning("Não foram encontradas colunas de data no conjunto de dados. Por favor, converta uma coluna para tipo de data primeiro.")
        else:
            date_col = st.selectbox("Selecione a coluna de data:", date_columns)
            value_col = st.selectbox("Selecione a coluna de valor:", numeric_columns)
            color_col = st.selectbox("Agrupar por (opcional):", ["Nenhuma"] + categorical_columns)
            
            color = None if color_col == "Nenhuma" else color_col
            
            # Preparar dados
            timeline_data = data.copy()
            timeline_data = timeline_data.sort_values(by=date_col)
            
            # Opções de agregação
            if color:
                freq = st.selectbox(
                    "Agregação temporal:",
                    ["Dados originais", "Diária", "Semanal", "Mensal", "Trimestral", "Anual"]
                )
                
                # Aplicar agregação se selecionada
                if freq != "Dados originais":
                    freq_map = {
                        "Diária": "D",
                        "Semanal": "W",
                        "Mensal": "M",
                        "Trimestral": "Q",
                        "Anual": "Y"
                    }
                    
                    agg_data = timeline_data.groupby([pd.Grouper(key=date_col, freq=freq_map[freq]), color_col])[value_col].mean().reset_index()
                    timeline_data = agg_data
            else:
                freq = st.selectbox(
                    "Agregação temporal:",
                    ["Dados originais", "Diária", "Semanal", "Mensal", "Trimestral", "Anual"]
                )
                
                # Aplicar agregação se selecionada
                if freq != "Dados originais":
                    freq_map = {
                        "Diária": "D",
                        "Semanal": "W",
                        "Mensal": "M",
                        "Trimestral": "Q",
                        "Anual": "Y"
                    }
                    
                    agg_data = timeline_data.groupby(pd.Grouper(key=date_col, freq=freq_map[freq]))[value_col].mean().reset_index()
                    timeline_data = agg_data
            
            # Criar gráfico de linha do tempo
            if color:
                fig = px.line(
                    timeline_data,
                    x=date_col,
                    y=value_col,
                    color=color,
                    markers=True,
                    title=f"Linha do Tempo: {value_col} ao longo do tempo, agrupado por {color}"
                )
            else:
                fig = px.line(
                    timeline_data,
                    x=date_col,
                    y=value_col,
                    markers=True,
                    title=f"Linha do Tempo: {value_col} ao longo do tempo"
                )
            
            # Adicionar intervalo de seleção
            fig.update_layout(
                xaxis_title="Data",
                yaxis_title=value_col,
                hovermode="x unified",
                width=800,
                height=500
            )
            
            # Adicionar controle deslizante para intervalo de datas
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1a", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico Sunburst
    elif interactive_plot_type == "Gráfico Sunburst":
        st.subheader("Gráfico Sunburst (Hierárquico)")
        
        if len(categorical_columns) < 2:
            st.warning("São necessárias pelo menos 2 colunas categóricas para criar um gráfico Sunburst.")
        else:
            path_cols = st.multiselect(
                "Selecione colunas para o caminho hierárquico (ordem importa):",
                categorical_columns,
                default=categorical_columns[:min(3, len(categorical_columns))]
            )
            
            value_col = st.selectbox("Selecione a coluna de valor:", numeric_columns)
            
            if path_cols and len(path_cols) >= 1 and value_col:
                # Criar gráfico Sunburst
                fig = px.sunburst(
                    data,
                    path=path_cols,
                    values=value_col,
                    title=f"Gráfico Sunburst: {value_col} por {' > '.join(path_cols)}"
                )
                
                # Ajustar layout
                fig.update_layout(
                    width=700,
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selecione pelo menos uma coluna categórica para o caminho e uma coluna numérica para os valores.")
    
    # Gráfico de Área
    elif interactive_plot_type == "Gráfico de Área":
        st.subheader("Gráfico de Área")
        
        # Verificar se há coluna de data
        date_columns = [col for col in columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        
        if not date_columns:
            # Se não houver coluna de data, usar qualquer coluna como x
            x_col = st.selectbox("Selecione a coluna X:", columns)
            sort_by_x = st.checkbox("Ordenar por valores do eixo X", value=True)
        else:
            x_col = st.selectbox("Selecione a coluna X (de preferência uma coluna de data):", columns, index=columns.index(date_columns[0]) if date_columns else 0)
            sort_by_x = st.checkbox("Ordenar por valores do eixo X", value=True)
        
        y_cols = st.multiselect(
            "Selecione as colunas Y (múltiplas para áreas empilhadas):",
            numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))]
        )
        
        stacked = st.checkbox("Usar áreas empilhadas", value=True)
        
        if x_col and y_cols:
            # Preparar dados
            area_data = data.copy()
            if sort_by_x:
                area_data = area_data.sort_values(by=x_col)
            
            # Criar gráfico de área
            if len(y_cols) == 1:
                # Gráfico de área simples
                fig = px.area(
                    area_data,
                    x=x_col,
                    y=y_cols[0],
                    title=f"Gráfico de Área: {y_cols[0]} por {x_col}"
                )
            else:
                # Gráfico de área com múltiplas séries
                if stacked:
                    # Áreas empilhadas
                    fig = go.Figure()
                    
                    for y_col in y_cols:
                        fig.add_trace(go.Scatter(
                            x=area_data[x_col],
                            y=area_data[y_col],
                            mode='lines',
                            stackgroup='one',
                            name=y_col,
                            hovertemplate=f"{y_col}: %{{y}}<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        title=f"Gráfico de Área Empilhada por {x_col}",
                        xaxis_title=x_col,
                        yaxis_title="Valor",
                        hovermode="x unified"
                    )
                else:
                    # Áreas sobrepostas
                    fig = go.Figure()
                    
                    for i, y_col in enumerate(y_cols):
                        fig.add_trace(go.Scatter(
                            x=area_data[x_col],
                            y=area_data[y_col],
                            mode='lines',
                            line=dict(width=0.5),
                            stackgroup='one',
                            groupnorm='percent' if st.checkbox("Normalizar para 100%", value=False) else None,
                            name=y_col
                        ))
                    
                    fig.update_layout(
                        title=f"Gráfico de Área por {x_col}",
                        xaxis_title=x_col,
                        yaxis_title="Valor",
                        hovermode="x unified"
                    )
            
            st.plotly_chart(fig, use_container_width=True)

# Visualização Personalizada
elif visualization_type == "Visualização Personalizada":
    st.header("Visualização Personalizada")
    
    st.write("""
    Crie uma visualização personalizada combinando diferentes elementos e configurações. 
    Você pode ajustar as cores, títulos, legendas e outros aspectos do gráfico.
    """)
    
    # Selecionar tipo de gráfico base
    base_chart = st.selectbox(
        "Selecione o tipo de gráfico base:",
        ["Gráfico de Dispersão", "Gráfico de Linhas", "Gráfico de Barras", "Histograma", "Boxplot"]
    )
    
    # Opções comuns
    st.subheader("Dados para o Gráfico")
    
    # Configurações específicas por tipo de gráfico
    if base_chart == "Gráfico de Dispersão":
        x_col = st.selectbox("Eixo X:", numeric_columns)
        y_col = st.selectbox("Eixo Y:", numeric_columns, index=min(1, len(numeric_columns)-1))
        color_col = st.selectbox("Coluna de Cor (opcional):", ["Nenhuma"] + categorical_columns)
        size_col = st.selectbox("Coluna de Tamanho (opcional):", ["Nenhuma"] + numeric_columns)
        
        # Opções avançadas em seção expansível
        with st.expander("Opções Avançadas"):
            title = st.text_input("Título do Gráfico:", f"{y_col} vs {x_col}")
            x_title = st.text_input("Título do Eixo X:", x_col)
            y_title = st.text_input("Título do Eixo Y:", y_col)
            
            # Seleção de paleta de cores
            color_scheme = st.selectbox(
                "Esquema de Cores:",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Reds", "Greens", "Purples", "Oranges"]
            )
            
            # Opções para eixos
            log_x = st.checkbox("Escala Logarítmica para Eixo X")
            log_y = st.checkbox("Escala Logarítmica para Eixo Y")
            
            # Opções para pontos
            opacity = st.slider("Opacidade dos Pontos:", 0.0, 1.0, 0.7, 0.1)
            
            # Opções para linha de tendência
            add_trendline = st.checkbox("Adicionar Linha de Tendência")
            trend_type = st.selectbox(
                "Tipo de Linha de Tendência:",
                ["ols", "lowess"],
                disabled=not add_trendline
            )
            
            # Opções para tamanho do gráfico
            width = st.slider("Largura do Gráfico:", 400, 1200, 800, 50)
            height = st.slider("Altura do Gráfico:", 300, 1000, 600, 50)
        
        # Preparar configurações
        color = None if color_col == "Nenhuma" else color_col
        size = None if size_col == "Nenhuma" else size_col
        
        # Criar gráfico de dispersão personalizado
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color,
            size=size,
            title=title,
            color_continuous_scale=color_scheme.lower() if color and color in numeric_columns else None,
            color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()) if color and color in categorical_columns else None,
            opacity=opacity,
            trendline=trend_type if add_trendline else None,
            log_x=log_x,
            log_y=log_y
        )
        
        # Atualizar layout
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            width=width,
            height=height
        )
        
        # Mostrar gráfico
        st.plotly_chart(fig, use_container_width=True)
    
    elif base_chart == "Gráfico de Linhas":
        x_col = st.selectbox("Eixo X:", columns)
        y_cols = st.multiselect("Eixo Y (múltiplos permitidos):", numeric_columns)
        
        # Verificar se a coluna x é datetime e oferecer agregação
        is_datetime = x_col in data.columns and pd.api.types.is_datetime64_any_dtype(data[x_col])
        if is_datetime:
            st.info(f"A coluna '{x_col}' é detectada como data/hora.")
            
            freq = st.selectbox(
                "Agregação temporal:",
                ["Dados originais", "Diária", "Semanal", "Mensal", "Trimestral", "Anual"]
            )
            
            agg_method = st.selectbox(
                "Método de Agregação:",
                ["Média", "Soma", "Mínimo", "Máximo", "Contagem"]
            )
            
            # Mapear seleções para funções pandas
            freq_map = {
                "Dados originais": None,
                "Diária": "D",
                "Semanal": "W",
                "Mensal": "M",
                "Trimestral": "Q",
                "Anual": "Y"
            }
            
            agg_map = {
                "Média": "mean",
                "Soma": "sum",
                "Mínimo": "min",
                "Máximo": "max",
                "Contagem": "count"
            }
        else:
            # Se não for datetime, apenas ordenar por x
            sort_by_x = st.checkbox("Ordenar por valores do eixo X", value=True)
            freq = "Dados originais"
            agg_method = "Média"
        
        # Opções avançadas
        with st.expander("Opções Avançadas"):
            title = st.text_input("Título do Gráfico:", f"Gráfico de Linhas: {', '.join(y_cols)} por {x_col}")
            x_title = st.text_input("Título do Eixo X:", x_col)
            y_title = st.text_input("Título do Eixo Y:", "Valor")
            
            # Opções para linhas
            line_mode = st.selectbox(
                "Modo da Linha:",
                ["Linhas e Marcadores", "Apenas Linhas", "Apenas Marcadores"]
            )
            
            # Mapear para modos plotly
            mode_map = {
                "Linhas e Marcadores": "lines+markers",
                "Apenas Linhas": "lines",
                "Apenas Marcadores": "markers"
            }
            
            line_shape = st.selectbox(
                "Forma da Linha:",
                ["Linear", "Spline", "Degraus Verticais", "Degraus Horizontais"]
            )
            
            # Mapear para shapes plotly
            shape_map = {
                "Linear": "linear",
                "Spline": "spline",
                "Degraus Verticais": "vh",
                "Degraus Horizontais": "hv"
            }
            
            # Seleção de paleta de cores
            color_scheme = st.selectbox(
                "Esquema de Cores:",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Reds", "Greens", "Purples", "Oranges"]
            )
            
            # Opções para eixos
            log_y = st.checkbox("Escala Logarítmica para Eixo Y")
            show_grid = st.checkbox("Mostrar Linhas de Grade", value=True)
            
            # Opções para tamanho do gráfico
            width = st.slider("Largura do Gráfico:", 400, 1200, 800, 50)
            height = st.slider("Altura do Gráfico:", 300, 1000, 500, 50)
        
        if x_col and y_cols:
            # Preparar dados
            if is_datetime and freq != "Dados originais":
                # Criar cópia para não modificar os dados originais
                plot_data = data.copy()
                result_dfs = []
                
                # Agregar por frequência temporal
                for y_col in y_cols:
                    temp_df = data[[x_col, y_col]].copy()
                    agg_df = temp_df.groupby(pd.Grouper(key=x_col, freq=freq_map[freq]))[y_col].agg(agg_map[agg_method]).reset_index()
                    agg_df.columns = [x_col, y_col]
                    result_dfs.append(agg_df)
                
                # Mesclar os DataFrames resultantes
                from functools import reduce
                plot_data = reduce(lambda left, right: pd.merge(left, right, on=x_col, how='outer'), result_dfs)
            else:
                plot_data = data.copy()
                if not is_datetime and 'sort_by_x' in locals() and sort_by_x:
                    plot_data = plot_data.sort_values(by=x_col)
            
            # Criar gráfico de linhas personalizado
            fig = go.Figure()
            
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=plot_data[x_col],
                    y=plot_data[y_col],
                    mode=mode_map[line_mode],
                    name=y_col,
                    line=dict(shape=shape_map[line_shape])
                ))
            
            # Atualizar layout
            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title=y_title,
                yaxis_type="log" if log_y else "linear",
                width=width,
                height=height,
                hovermode="x unified",
                colorway=px.colors.sequential.__getattribute__(color_scheme.lower())
            )
            
            # Atualizar grades
            fig.update_xaxes(showgrid=show_grid)
            fig.update_yaxes(showgrid=show_grid)
            
            # Adicionar seletor de intervalo se for datetime
            if is_datetime:
                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1a", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
            
            # Mostrar gráfico
            st.plotly_chart(fig, use_container_width=True)
    
    elif base_chart == "Gráfico de Barras":
        x_col = st.selectbox("Eixo X (Categorias):", categorical_columns)
        y_col = st.selectbox("Eixo Y (Valores):", numeric_columns)
        color_col = st.selectbox("Coluna de Cor (opcional):", ["Nenhuma"] + categorical_columns)
        
        # Opções avançadas
        with st.expander("Opções Avançadas"):
            title = st.text_input("Título do Gráfico:", f"{y_col} por {x_col}")
            x_title = st.text_input("Título do Eixo X:", x_col)
            y_title = st.text_input("Título do Eixo Y:", y_col)
            
            # Orientação
            orientation = st.radio("Orientação:", ["Vertical", "Horizontal"])
            
            # Agrupamento
            bar_mode = st.radio("Modo de Barras:", ["Agrupadas", "Empilhadas", "Empilhadas 100%"])
            
            # Mapear para modos plotly
            mode_map = {
                "Agrupadas": "group",
                "Empilhadas": "stack",
                "Empilhadas 100%": "relative"
            }
            
            # Seleção de paleta de cores
            color_scheme = st.selectbox(
                "Esquema de Cores:",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Reds", "Greens", "Purples", "Oranges"]
            )
            
            # Ordenação
            sort_bars = st.checkbox("Ordenar Barras por Valor")
            sort_direction = st.radio("Direção de Ordenação:", ["Ascendente", "Descendente"]) if sort_bars else "Ascendente"
            
            # Opções para eixos
            log_y = st.checkbox("Escala Logarítmica para Eixo Y")
            show_values = st.checkbox("Mostrar Valores nas Barras")
            
            # Opções para tamanho do gráfico
            width = st.slider("Largura do Gráfico:", 400, 1200, 800, 50)
            height = st.slider("Altura do Gráfico:", 300, 1000, 600, 50)
        
        # Preparar dados
        if sort_bars:
            # Agregar e ordenar dados
            if color_col != "Nenhuma":
                agg_data = data.groupby([x_col, color_col])[y_col].mean().reset_index()
                if sort_direction == "Ascendente":
                    order = agg_data.groupby(x_col)[y_col].sum().sort_values().index.tolist()
                else:
                    order = agg_data.groupby(x_col)[y_col].sum().sort_values(ascending=False).index.tolist()
            else:
                agg_data = data.groupby(x_col)[y_col].mean().reset_index()
                if sort_direction == "Ascendente":
                    order = agg_data.sort_values(by=y_col)[x_col].tolist()
                else:
                    order = agg_data.sort_values(by=y_col, ascending=False)[x_col].tolist()
        else:
            if color_col != "Nenhuma":
                agg_data = data.groupby([x_col, color_col])[y_col].mean().reset_index()
            else:
                agg_data = data.groupby(x_col)[y_col].mean().reset_index()
            order = None
        
        # Criar gráfico de barras personalizado
        if orientation == "Horizontal":
            fig = px.bar(
                agg_data,
                y=x_col,
                x=y_col,
                color=None if color_col == "Nenhuma" else color_col,
                title=title,
                orientation='h',
                barmode=mode_map[bar_mode],
                category_orders={x_col: order} if order else None,
                color_continuous_scale=color_scheme.lower() if color_col != "Nenhuma" and color_col in numeric_columns else None,
                color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()) if color_col != "Nenhuma" and color_col in categorical_columns else None,
            )
        else:
            fig = px.bar(
                agg_data,
                x=x_col,
                y=y_col,
                color=None if color_col == "Nenhuma" else color_col,
                title=title,
                barmode=mode_map[bar_mode],
                category_orders={x_col: order} if order else None,
                color_continuous_scale=color_scheme.lower() if color_col != "Nenhuma" and color_col in numeric_columns else None,
                color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()) if color_col != "Nenhuma" and color_col in categorical_columns else None,
            )
        
        # Atualizar layout
        fig.update_layout(
            xaxis_title=y_title if orientation == "Horizontal" else x_title,
            yaxis_title=x_title if orientation == "Horizontal" else y_title,
            yaxis_type="log" if log_y and orientation != "Horizontal" else "linear",
            xaxis_type="log" if log_y and orientation == "Horizontal" else "linear",
            width=width,
            height=height
        )
        
        # Adicionar valores nas barras
        if show_values:
            if orientation == "Horizontal":
                fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')
            else:
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        
        # Mostrar gráfico
        st.plotly_chart(fig, use_container_width=True)
    
    elif base_chart == "Histograma":
        value_col = st.selectbox("Selecione a Coluna para Histograma:", numeric_columns)
        color_col = st.selectbox("Agrupar Por (opcional):", ["Nenhuma"] + categorical_columns)
        
        # Opções avançadas
        with st.expander("Opções Avançadas"):
            title = st.text_input("Título do Gráfico:", f"Histograma de {value_col}")
            x_title = st.text_input("Título do Eixo X:", value_col)
            y_title = st.text_input("Título do Eixo Y:", "Contagem")
            
            # Opções para bins
            bins = st.slider("Número de Intervalos (Bins):", 5, 100, 20, 1)
            
            # Tipo de histograma
            hist_norm = st.selectbox(
                "Normalização do Histograma:",
                ["Contagem", "Probabilidade", "Densidade", "Densidade de Probabilidade"]
            )
            
            # Mapear para modos plotly
            norm_map = {
                "Contagem": None,
                "Probabilidade": "probability",
                "Densidade": "density",
                "Densidade de Probabilidade": "probability density"
            }
            
            # Opções para sobreposição
            if color_col != "Nenhuma":
                hist_mode = st.radio("Modo de Histograma:", ["Sobreposto", "Lado a Lado", "Empilhado"])
                # Mapear para modos plotly
                mode_map = {
                    "Sobreposto": "overlay",
                    "Lado a Lado": "group",
                    "Empilhado": "stack"
                }
            
            # Opções para exibição
            show_kde = st.checkbox("Mostrar Curva KDE", value=False)
            
            # Seleção de paleta de cores
            color_scheme = st.selectbox(
                "Esquema de Cores:",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Reds", "Greens", "Purples", "Oranges"]
            )
            
            # Opções para eixos
            log_y = st.checkbox("Escala Logarítmica para Eixo Y")
            
            # Opções para tamanho do gráfico
            width = st.slider("Largura do Gráfico:", 400, 1200, 800, 50)
            height = st.slider("Altura do Gráfico:", 300, 1000, 500, 50)
        
        if value_col:
            # Criar histograma personalizado
            if color_col != "Nenhuma" and show_kde:
                # Para KDE com cores, usamos matplotlib/seaborn
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for cat in data[color_col].unique():
                    subset = data[data[color_col] == cat]
                    sns.histplot(
                        data=subset,
                        x=value_col,
                        bins=bins,
                        kde=True,
                        label=cat,
                        alpha=0.5,
                        stat=norm_map[hist_norm] if norm_map[hist_norm] else "count"
                    )
                
                plt.title(title)
                plt.xlabel(x_title)
                plt.ylabel(y_title)
                plt.legend()
                
                if log_y:
                    plt.yscale('log')
                
                st.pyplot(fig)
            else:
                # Usar plotly para histograma
                color = None if color_col == "Nenhuma" else color_col
                
                fig = px.histogram(
                    data,
                    x=value_col,
                    color=color,
                    title=title,
                    nbins=bins,
                    histnorm=norm_map[hist_norm],
                    barmode=mode_map[hist_mode] if color != None else None,
                    color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower())
                )
                
                # Atualizar layout
                fig.update_layout(
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    yaxis_type="log" if log_y else "linear",
                    width=width,
                    height=height
                )
                
                # Adicionar curva KDE se solicitado
                if show_kde and color_col == "Nenhuma":
                    # Calcular densidade KDE
                    from scipy.stats import gaussian_kde
                    import numpy as np
                    
                    # Remover valores ausentes
                    kde_data = data[value_col].dropna()
                    
                    if len(kde_data) > 1:
                        kde = gaussian_kde(kde_data)
                        x_range = np.linspace(kde_data.min(), kde_data.max(), 1000)
                        y_kde = kde(x_range)
                        
                        # Normalizar KDE para corresponder à altura do histograma
                        if norm_map[hist_norm] is None:
                            hist_heights = np.histogram(kde_data, bins=bins)[0]
                            scale_factor = max(hist_heights) / max(y_kde)
                            y_kde = y_kde * scale_factor
                        
                        # Adicionar traço KDE
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_kde,
                            mode='lines',
                            name='KDE',
                            line=dict(color='red', width=2)
                        ))
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif base_chart == "Boxplot":
        y_col = st.selectbox("Selecione a Coluna de Valor:", numeric_columns)
        x_col = st.selectbox("Selecione a Coluna de Agrupamento (opcional):", ["Nenhuma"] + categorical_columns)
        color_col = st.selectbox("Coluna de Cor (opcional):", ["Nenhuma"] + categorical_columns) if x_col != "Nenhuma" else "Nenhuma"
        
        # Evitar usar a mesma coluna para x e cor
        if color_col == x_col:
            color_col = "Nenhuma"
        
        # Opções avançadas
        with st.expander("Opções Avançadas"):
            title = st.text_input("Título do Gráfico:", f"Boxplot de {y_col}" + (f" por {x_col}" if x_col != "Nenhuma" else ""))
            x_title = st.text_input("Título do Eixo X:", x_col if x_col != "Nenhuma" else "")
            y_title = st.text_input("Título do Eixo Y:", y_col)
            
            # Orientação
            orientation = st.radio("Orientação:", ["Vertical", "Horizontal"])
            
            # Opções de exibição
            show_points = st.checkbox("Mostrar Pontos Individuais", value=True)
            point_mode = st.selectbox(
                "Modo de Exibição de Pontos:",
                ["Todos", "Outliers", "Suspeitosos", "Outliers+Suspeitosos"],
                disabled=not show_points
            )
            
            # Mapear para modos plotly
            points_map = {
                "Todos": "all",
                "Outliers": "outliers",
                "Suspeitosos": "suspectedoutliers",
                "Outliers+Suspeitosos": "outliers+suspectedoutliers"
            }
            
            # Seleção de paleta de cores
            color_scheme = st.selectbox(
                "Esquema de Cores:",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Blues", "Reds", "Greens", "Purples", "Oranges"]
            )
            
            # Opção para mostrar notch
            show_notch = st.checkbox("Mostrar Notch (Intervalo de Confiança da Mediana)")
            
            # Opções para eixos
            log_y = st.checkbox("Escala Logarítmica para Eixo Y")
            
            # Opções para tamanho do gráfico
            width = st.slider("Largura do Gráfico:", 400, 1200, 800, 50)
            height = st.slider("Altura do Gráfico:", 300, 1000, 600, 50)
        
        # Preparar variáveis
        x = None if x_col == "Nenhuma" else x_col
        color = None if color_col == "Nenhuma" else color_col
        
        # Criar boxplot personalizado
        if orientation == "Horizontal":
            fig = px.box(
                data,
                y=x,
                x=y_col,
                color=color,
                title=title,
                points=points_map[point_mode] if show_points else False,
                notched=show_notch,
                color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()) if color else None
            )
        else:
            fig = px.box(
                data,
                x=x,
                y=y_col,
                color=color,
                title=title,
                points=points_map[point_mode] if show_points else False,
                notched=show_notch,
                color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()) if color else None
            )
        
        # Atualizar layout
        fig.update_layout(
            xaxis_title=y_title if orientation == "Horizontal" else x_title,
            yaxis_title=x_title if orientation == "Horizontal" else y_title,
            yaxis_type="log" if log_y and orientation != "Horizontal" else "linear",
            xaxis_type="log" if log_y and orientation == "Horizontal" else "linear",
            width=width,
            height=height
        )
        
        # Mostrar gráfico
        st.plotly_chart(fig, use_container_width=True)

# Adicionar recursos de exportação
st.sidebar.markdown("---")
with st.sidebar.expander("📥 Exportar Visualização"):
    st.write("Exporte o gráfico atual como imagem ou arquivo interativo.")
    
    export_format = st.selectbox(
        "Formato de Exportação:",
        ["PNG", "JPEG", "SVG", "HTML Interativo"]
    )
    
    st.write("""
    Para exportar o gráfico:
    1. Passe o mouse sobre o gráfico
    2. Clique no botão "Baixar" no canto superior direito
    3. Selecione o formato desejado
    """)
    
    st.markdown("---")
    st.markdown("### Dicas:")
    st.markdown("""
    - Gráficos Plotly são interativos, você pode:
      - Dar zoom selecionando uma área
      - Passar o mouse para ver detalhes
      - Clicar para destacar itens
      - Usar a barra de ferramentas para personalizar a visualização
    - Para resetar a visualização, clique duas vezes no gráfico
    """)