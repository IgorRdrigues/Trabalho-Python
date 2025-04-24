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

# Adicionar diretório pai ao caminho para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_download_link

st.title("📁 Exportação de Dados")

# Verificar se os dados foram carregados
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Por favor, carregue seus dados primeiro usando a seção Upload de Dados.")
    st.stop()

# Obter dados do estado da sessão
data = st.session_state.data

# Opções principais de exportação
st.header("Opções de Exportação")

export_what = st.radio(
    "O que você gostaria de exportar?",
    ["Conjunto de Dados Completo", "Conjunto de Dados Filtrado", "Estatísticas Resumidas", "Visualizações"]
)

# Exportar Conjunto de Dados Completo
if export_what == "Conjunto de Dados Completo":
    st.subheader("Exportar Conjunto de Dados Completo")
    
    # Exibir informações do conjunto de dados
    st.write(f"Conjunto de Dados: {st.session_state.filename}")
    st.write(f"Linhas: {data.shape[0]}, Colunas: {data.shape[1]}")
    
    # Selecionar formato de exportação
    export_format = st.selectbox(
        "Selecione o Formato de Exportação",
        ["CSV", "Excel", "JSON"]
    )
    
    # Opções adicionais
    with st.expander("Opções Avançadas de Exportação"):
        if export_format == "CSV":
            separator = st.selectbox("Separador:", [",", ";", "\\t", "|"])
            decimal = st.selectbox("Marcador decimal:", [".", ","])
            encoding = st.selectbox("Codificação:", ["utf-8", "latin1", "iso-8859-1", "cp1252"])
            include_index = st.checkbox("Incluir índice de linhas", value=False)
        
        elif export_format == "Excel":
            sheet_name = st.text_input("Nome da planilha:", "Dados")
            include_index = st.checkbox("Incluir índice de linhas", value=False)
            add_filters = st.checkbox("Adicionar filtros automáticos", value=True)
            
        elif export_format == "JSON":
            json_orient = st.selectbox(
                "Orientação JSON:",
                ["records", "columns", "index", "values", "table"],
                index=0,
                help="'records': lista de objetos (registros), 'columns': dicionário de listas, 'index': dicionário de dicionários"
            )
            include_index = st.checkbox("Incluir índice de linhas", value=False)
            date_format = st.selectbox("Formato de data:", ["iso", "epoch"])
            indent = st.slider("Indentação:", 0, 4, 2)
    
    # Criar link de download
    if export_format == "CSV":
        # Configurar opções avançadas se expandidas
        separator_map = {",": ",", ";": ";", "\\t": "\t", "|": "|"}
        sep = separator_map.get(separator, ",") if 'separator' in locals() else ","
        dec = decimal if 'decimal' in locals() else "."
        enc = encoding if 'encoding' in locals() else "utf-8"
        inc_idx = include_index if 'include_index' in locals() else False
        
        csv = data.to_csv(index=inc_idx, sep=sep, decimal=dec)
        b64 = base64.b64encode(csv.encode(enc)).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{st.session_state.filename}_exportacao.csv">Baixar Arquivo CSV</a>'
        
    elif export_format == "Excel":
        # Configurar opções avançadas se expandidas
        sheet = sheet_name if 'sheet_name' in locals() else "Dados"
        inc_idx = include_index if 'include_index' in locals() else False
        add_filt = add_filters if 'add_filters' in locals() else True
        
        # Criar arquivo Excel na memória
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=inc_idx, sheet_name=sheet)
            
            # Adicionar filtros se solicitado
            if add_filt:
                worksheet = writer.sheets[sheet]
                worksheet.autofilter(0, 0, data.shape[0], data.shape[1] - 1)
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{st.session_state.filename}_exportacao.xlsx">Baixar Arquivo Excel</a>'
        
    elif export_format == "JSON":
        # Configurar opções avançadas se expandidas
        orient = json_orient if 'json_orient' in locals() else "records"
        inc_idx = include_index if 'include_index' in locals() else False
        date_fmt = date_format if 'date_format' in locals() else "iso"
        ind = indent if 'indent' in locals() else 2
        
        # Converter para formato JSON
        json_str = data.to_json(orient=orient, date_format=date_fmt, indent=ind, index=inc_idx, force_ascii=False)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{st.session_state.filename}_exportacao.json">Baixar Arquivo JSON</a>'
    
    # Exibir link de download
    st.markdown(href, unsafe_allow_html=True)
    
    # Exibir prévia
    with st.expander("Prévia dos Dados"):
        st.dataframe(data.head(10))
        st.caption("Mostrando apenas as primeiras 10 linhas do conjunto de dados.")

# Exportar Conjunto de Dados Filtrado
elif export_what == "Conjunto de Dados Filtrado":
    st.subheader("Exportar Conjunto de Dados Filtrado")
    
    # Criar opções de filtro
    st.write("Aplique filtros ao conjunto de dados antes de exportar:")
    
    # Seletor de colunas para filtragem
    filter_columns = st.multiselect(
        "Selecione colunas para filtrar:",
        data.columns.tolist()
    )
    
    filtered_data = data.copy()
    
    # Criar filtros para cada coluna selecionada
    for column in filter_columns:
        st.write(f"Filtro para: {column}")
        
        # Diferentes tipos de filtro com base no tipo de dados
        if pd.api.types.is_numeric_dtype(data[column]):
            # Coluna numérica - filtrar por intervalo
            min_val = float(data[column].min())
            max_val = float(data[column].max())
            
            filter_range = st.slider(
                f"Intervalo para {column}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            
            filtered_data = filtered_data[(filtered_data[column] >= filter_range[0]) & 
                                         (filtered_data[column] <= filter_range[1])]
            
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            # Coluna de data/hora - filtrar por intervalo de datas
            min_date = data[column].min()
            max_date = data[column].max()
            
            filter_date_range = st.date_input(
                f"Intervalo de datas para {column}:",
                value=(min_date.date(), max_date.date())
            )
            
            if len(filter_date_range) == 2:
                start_date, end_date = filter_date_range
                filtered_data = filtered_data[(filtered_data[column].dt.date >= start_date) & 
                                             (filtered_data[column].dt.date <= end_date)]
            
        else:
            # Coluna categórica ou de objeto - filtrar por seleção
            unique_values = data[column].dropna().unique().tolist()
            
            if len(unique_values) <= 20:
                # Se houver poucos valores únicos, mostrar checkboxes
                st.write(f"Selecione os valores para {column}:")
                
                all_option = st.checkbox("Selecionar Todos", value=True, key=f"all_{column}")
                
                # Lógica para selecionar/desselecionar todos
                if all_option:
                    default_selection = unique_values
                else:
                    default_selection = []
                
                selected_values = st.multiselect(
                    f"Valores para {column}:",
                    unique_values,
                    default=default_selection,
                    key=f"select_{column}"
                )
                
                if selected_values:
                    filtered_data = filtered_data[filtered_data[column].isin(selected_values)]
            else:
                # Muitos valores únicos, usar entrada de texto para pesquisa
                search_term = st.text_input(f"Pesquisar em {column} (deixe vazio para todos):")
                
                if search_term:
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(search_term, case=False, na=False)]
    
    # Mostrar resultados do filtro
    st.write(f"Dados filtrados contêm {filtered_data.shape[0]} linhas (do original {data.shape[0]} linhas)")
    
    # Seleção de colunas para exportação
    st.write("Selecione colunas para incluir na exportação:")
    export_columns = st.multiselect(
        "Selecione colunas para exportar (deixe vazio para exportar todas):",
        data.columns.tolist()
    )
    
    if export_columns:
        export_data = filtered_data[export_columns].copy()
    else:
        export_data = filtered_data.copy()
    
    # Prévia dos dados filtrados
    with st.expander("Prévia dos Dados Filtrados"):
        st.dataframe(export_data.head(10))
        st.caption("Mostrando apenas as primeiras 10 linhas do conjunto de dados filtrado.")
    
    # Opções de exportação
    st.subheader("Opções de Exportação")
    
    # Selecionar formato de exportação
    export_format = st.selectbox(
        "Selecione o Formato de Exportação",
        ["CSV", "Excel", "JSON"]
    )
    
    # Opções avançadas por formato
    with st.expander("Opções Avançadas de Exportação"):
        if export_format == "CSV":
            separator = st.selectbox("Separador:", [",", ";", "\\t", "|"])
            decimal = st.selectbox("Marcador decimal:", [".", ","])
            encoding = st.selectbox("Codificação:", ["utf-8", "latin1", "iso-8859-1", "cp1252"])
            include_index = st.checkbox("Incluir índice de linhas", value=False)
        
        elif export_format == "Excel":
            sheet_name = st.text_input("Nome da planilha:", "Dados Filtrados")
            include_index = st.checkbox("Incluir índice de linhas", value=False)
            add_filters = st.checkbox("Adicionar filtros automáticos", value=True)
            
        elif export_format == "JSON":
            json_orient = st.selectbox(
                "Orientação JSON:",
                ["records", "columns", "index", "values", "table"],
                index=0,
                help="'records': lista de objetos (registros), 'columns': dicionário de listas, 'index': dicionário de dicionários"
            )
            include_index = st.checkbox("Incluir índice de linhas", value=False)
            date_format = st.selectbox("Formato de data:", ["iso", "epoch"])
            indent = st.slider("Indentação:", 0, 4, 2)
    
    # Nome do arquivo para download
    download_filename = st.text_input(
        "Nome do arquivo para download:",
        value=f"{st.session_state.filename.split('.')[0]}_filtrado"
    )
    
    # Gerar e exibir link de download
    if not export_data.empty:
        if export_format == "CSV":
            # Configurar opções avançadas se expandidas
            separator_map = {",": ",", ";": ";", "\\t": "\t", "|": "|"}
            sep = separator_map.get(separator, ",") if 'separator' in locals() else ","
            dec = decimal if 'decimal' in locals() else "."
            enc = encoding if 'encoding' in locals() else "utf-8"
            inc_idx = include_index if 'include_index' in locals() else False
            
            csv = export_data.to_csv(index=inc_idx, sep=sep, decimal=dec)
            b64 = base64.b64encode(csv.encode(enc)).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}.csv">Baixar Dados Filtrados (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "Excel":
            # Configurar opções avançadas se expandidas
            sheet = sheet_name if 'sheet_name' in locals() else "Dados Filtrados"
            inc_idx = include_index if 'include_index' in locals() else False
            add_filt = add_filters if 'add_filters' in locals() else True
            
            # Criar arquivo Excel na memória
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                export_data.to_excel(writer, index=inc_idx, sheet_name=sheet)
                
                # Adicionar filtros se solicitado
                if add_filt:
                    worksheet = writer.sheets[sheet]
                    worksheet.autofilter(0, 0, export_data.shape[0], export_data.shape[1] - 1)
            
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{download_filename}.xlsx">Baixar Dados Filtrados (Excel)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "JSON":
            # Configurar opções avançadas se expandidas
            orient = json_orient if 'json_orient' in locals() else "records"
            inc_idx = include_index if 'include_index' in locals() else False
            date_fmt = date_format if 'date_format' in locals() else "iso"
            ind = indent if 'indent' in locals() else 2
            
            # Converter para formato JSON
            json_str = export_data.to_json(orient=orient, date_format=date_fmt, indent=ind, index=inc_idx, force_ascii=False)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="{download_filename}.json">Baixar Dados Filtrados (JSON)</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Não há dados para exportar após aplicar os filtros.")

# Exportar Estatísticas Resumidas
elif export_what == "Estatísticas Resumidas":
    st.subheader("Exportar Estatísticas Resumidas")
    
    # Selecionar colunas para estatísticas
    stat_columns = st.multiselect(
        "Selecione colunas para estatísticas:",
        data.columns.tolist(),
        default=st.session_state.numeric_columns[:min(5, len(st.session_state.numeric_columns))]
    )
    
    if stat_columns:
        # Calcular estatísticas para colunas selecionadas
        stats_df = pd.DataFrame()
        
        for column in stat_columns:
            # Verificar se a coluna é numérica
            if pd.api.types.is_numeric_dtype(data[column]):
                # Calcular estatísticas
                stats = {
                    'Coluna': column,
                    'Média': data[column].mean(),
                    'Mediana': data[column].median(),
                    'Desvio Padrão': data[column].std(),
                    'Mínimo': data[column].min(),
                    'Máximo': data[column].max(),
                    'Contagem': data[column].count(),
                    'Valores Ausentes': data[column].isna().sum()
                }
                
                # Adicionar ao DataFrame de estatísticas
                stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)
            else:
                # Para colunas não numéricas, calcular estatísticas de frequência
                value_counts = data[column].value_counts()
                top_value = value_counts.index[0] if not value_counts.empty else None
                
                stats = {
                    'Coluna': column,
                    'Tipo': str(data[column].dtype),
                    'Valores Únicos': data[column].nunique(),
                    'Valor Mais Comum': top_value,
                    'Contagem do Mais Comum': value_counts.iloc[0] if not value_counts.empty else 0,
                    'Contagem': data[column].count(),
                    'Valores Ausentes': data[column].isna().sum()
                }
                
                # Adicionar ao DataFrame de estatísticas
                stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)
        
        # Exibir estatísticas
        st.write("Estatísticas Resumidas:")
        st.dataframe(stats_df)
        
        # Gerar relatório para exportar
        st.subheader("Exportar Relatório de Estatísticas")
        
        # Escolher formato de exportação
        export_format = st.selectbox(
            "Selecione o Formato de Exportação",
            ["CSV", "Excel", "HTML", "JSON"]
        )
        
        # Opções específicas para cada formato
        with st.expander("Opções Avançadas"):
            if export_format == "Excel":
                include_metadata = st.checkbox("Incluir metadados do conjunto de dados", value=True)
                include_correlation = st.checkbox("Incluir matriz de correlação", value=True)
                sheet_styles = st.checkbox("Aplicar estilos à planilha", value=True)
            
            elif export_format == "HTML":
                include_metadata = st.checkbox("Incluir metadados do conjunto de dados", value=True)
                include_charts = st.checkbox("Incluir gráficos básicos", value=True)
                html_style = st.selectbox("Estilo do relatório:", ["Padrão", "Minimalista", "Escuro"])
            
            elif export_format == "JSON":
                include_metadata = st.checkbox("Incluir metadados do conjunto de dados", value=True)
                include_correlation = st.checkbox("Incluir matriz de correlação", value=True)
                json_indent = st.slider("Indentação JSON:", 0, 4, 2)
        
        # Gerar relatório com base no formato
        if export_format == "CSV":
            csv = stats_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="relatorio_estatisticas.csv">Baixar Relatório de Estatísticas (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "Excel":
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Adicionar estatísticas
                stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)
                
                # Adicionar planilha de resumo do conjunto de dados
                if 'include_metadata' in locals() and include_metadata:
                    summary = pd.DataFrame({
                        'Informação': [
                            'Nome do Conjunto de Dados',
                            'Número de Linhas',
                            'Número de Colunas',
                            'Data de Geração',
                            'Total de Valores Ausentes'
                        ],
                        'Valor': [
                            st.session_state.filename,
                            data.shape[0],
                            data.shape[1],
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            data.isna().sum().sum()
                        ]
                    })
                    
                    summary.to_excel(writer, sheet_name='Resumo', index=False)
                
                # Adicionar matriz de correlação se tivermos colunas numéricas
                if 'include_correlation' in locals() and include_correlation:
                    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_columns) >= 2:
                        corr_matrix = data[numeric_columns].corr()
                        corr_matrix.to_excel(writer, sheet_name='Matriz de Correlação')
                
                # Formatar a pasta de trabalho
                if 'sheet_styles' in locals() and sheet_styles:
                    workbook = writer.book
                    
                    # Adicionar formatos
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'bg_color': '#D9E1F2',
                        'border': 1
                    })
                    
                    numeric_format = workbook.add_format({
                        'num_format': '#,##0.00',
                        'align': 'right'
                    })
                    
                    # Aplicar formatação à planilha de estatísticas
                    worksheet = writer.sheets['Estatísticas']
                    for col_num, value in enumerate(stats_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        worksheet.set_column(col_num, col_num, 18)
                        
                        # Aplicar formatação numérica a colunas específicas
                        if value in ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']:
                            for row_num in range(1, len(stats_df) + 1):
                                try:
                                    worksheet.write(row_num, col_num, stats_df.iloc[row_num-1][value], numeric_format)
                                except:
                                    pass
            
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="relatorio_estatisticas.xlsx">Baixar Relatório de Estatísticas (Excel)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "HTML":
            # Criar um relatório HTML formatado
            
            # Selecionar estilo com base na escolha do usuário
            style_map = {
                "Padrão": """
                    body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
                    h1 { color: #2C3E50; }
                    h2 { color: #3498DB; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th { background-color: #3498DB; color: white; text-align: left; padding: 8px; }
                    td { border: 1px solid #ddd; padding: 8px; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    .summary { background-color: #EBF5FB; padding: 10px; border-radius: 5px; }
                """,
                "Minimalista": """
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }
                    h1 { font-weight: 300; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
                    h2 { font-weight: 400; margin-top: 30px; color: #444; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }
                    th { border-bottom: 2px solid #ddd; text-align: left; padding: 10px; font-weight: 500; }
                    td { padding: 10px; border-bottom: 1px solid #eee; }
                    .summary { border-left: 3px solid #ccc; padding-left: 15px; margin: 20px 0; }
                """,
                "Escuro": """
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: #f0f0f0; }
                    h1 { color: #bb86fc; }
                    h2 { color: #03dac6; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th { background-color: #3700b3; color: white; text-align: left; padding: 8px; }
                    td { border: 1px solid #333; padding: 8px; }
                    tr:nth-child(even) { background-color: #2d2d2d; }
                    .summary { background-color: #252525; padding: 10px; border-radius: 5px; }
                """
            }
            
            selected_style = style_map.get('html_style' in locals() and html_style or "Padrão", style_map["Padrão"])
            
            # Gerar HTML para tabela de estatísticas
            stats_table = stats_df.to_html(index=False)
            
            # Gerar gráficos se solicitado
            chart_html = ""
            if 'include_charts' in locals() and include_charts:
                # Para cada coluna numérica nas estatísticas, criar um histograma
                numeric_stats = stats_df[stats_df['Coluna'].isin(data.select_dtypes(include=['number']).columns)]
                for _, row in numeric_stats.iterrows():
                    column_name = row['Coluna']
                    
                    # Criar figura
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(data[column_name].dropna(), bins=20, alpha=0.7)
                    ax.set_title(f"Histograma de {column_name}")
                    ax.set_xlabel(column_name)
                    ax.set_ylabel("Frequência")
                    
                    # Converter para base64
                    buffer = io.BytesIO()
                    plt.tight_layout()
                    fig.savefig(buffer, format='png')
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.read()).decode()
                    plt.close(fig)
                    
                    # Adicionar ao HTML
                    chart_html += f"""
                    <div class="chart">
                        <h3>Histograma de {column_name}</h3>
                        <img src="data:image/png;base64,{img_str}" alt="Histograma de {column_name}" style="max-width:100%;">
                    </div>
                    """
            
            # Metadata HTML
            metadata_html = ""
            if 'include_metadata' in locals() and include_metadata:
                metadata_html = f"""
                <div class="summary">
                    <h2>Resumo do Conjunto de Dados</h2>
                    <p><strong>Nome do Conjunto de Dados:</strong> {st.session_state.filename}</p>
                    <p><strong>Número de Linhas:</strong> {data.shape[0]}</p>
                    <p><strong>Número de Colunas:</strong> {data.shape[1]}</p>
                    <p><strong>Data de Geração:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p><strong>Total de Valores Ausentes:</strong> {data.isna().sum().sum()}</p>
                </div>
                """
            
            # Montar HTML completo
            html = f"""
            <html>
            <head>
                <title>Relatório de Estatísticas</title>
                <meta charset="UTF-8">
                <style>
                    {selected_style}
                    .chart {{ margin: 20px 0; padding: 10px; background-color: #fff; border-radius: 5px; }}
                    @media print {{
                        .chart {{ page-break-inside: avoid; }}
                        tr {{ page-break-inside: avoid; }}
                        h2 {{ page-break-before: always; }}
                        h1 {{ page-break-before: avoid; }}
                    }}
                </style>
            </head>
            <body>
                <h1>Relatório de Estatísticas</h1>
                {metadata_html}
                
                <h2>Estatísticas das Colunas</h2>
                {stats_table}
                
                {chart_html}
            </body>
            </html>
            """
            
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="relatorio_estatisticas.html">Baixar Relatório de Estatísticas (HTML)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "JSON":
            # Converter estatísticas para JSON
            stats_json = stats_df.to_json(orient="records")
            
            # Configurar indentação
            indent_val = json_indent if 'json_indent' in locals() else 2
            
            # Adicionar metadados
            report = {
                "estatisticas": json.loads(stats_json)
            }
            
            if 'include_metadata' in locals() and include_metadata:
                report["metadados"] = {
                    "arquivo": st.session_state.filename,
                    "linhas": data.shape[0],
                    "colunas": data.shape[1],
                    "data_geracao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "valores_ausentes_total": int(data.isna().sum().sum())
                }
            
            # Adicionar dados de correlação se disponíveis
            if 'include_correlation' in locals() and include_correlation:
                numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_columns) >= 2:
                    corr_matrix = data[numeric_columns].corr()
                    report["correlacao"] = json.loads(corr_matrix.to_json())
            
            # Converter para string e codificar
            json_str = json.dumps(report, indent=indent_val, ensure_ascii=False)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="relatorio_estatisticas.json">Baixar Relatório de Estatísticas (JSON)</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Por favor, selecione pelo menos uma coluna para estatísticas.")

# Exportar Visualizações
elif export_what == "Visualizações":
    st.subheader("Exportar Visualizações")
    
    # Escolher tipo de visualização
    viz_type = st.selectbox(
        "Selecione o Tipo de Visualização",
        ["Gráfico de Barras", "Gráfico de Linhas", "Gráfico de Dispersão", "Histograma", "Boxplot", "Mapa de Calor de Correlação"]
    )
    
    # Configurar visualização com base no tipo
    if viz_type == "Gráfico de Barras":
        # Configuração de gráfico de barras
        x_col = st.selectbox("Selecione o eixo X (Categorias):", st.session_state.categorical_columns)
        y_col = st.selectbox("Selecione o eixo Y (Valores):", st.session_state.numeric_columns)
        
        # Opções adicionais
        with st.expander("Opções do Gráfico"):
            orientation = st.radio("Orientação:", ["Vertical", "Horizontal"])
            color_col = st.selectbox("Coluna para cores (opcional):", ["Nenhuma"] + st.session_state.categorical_columns)
            color = None if color_col == "Nenhuma" else color_col
            
            # Opções de estilo
            title = st.text_input("Título do gráfico:", f"{y_col} por {x_col}")
            color_scheme = st.selectbox("Esquema de Cores:", 
                                    ["Viridis", "Plasma", "Blues", "Reds", "Greens", "Purples", "Oranges"])
            
            # Opções de tamanho
            width = st.slider("Largura (pixels):", 400, 1200, 800, 50)
            height = st.slider("Altura (pixels):", 300, 800, 500, 50)
            
            # Formato de saída
            output_format = st.selectbox("Formato de Saída:", ["PNG", "JPEG", "SVG", "PDF", "HTML Interativo"])
            dpi = st.slider("Resolução (DPI):", 72, 300, 150, 10) if output_format != "HTML Interativo" else 150
        
        # Criar visualização
        # Preparar dados para gráfico de barras
        if color:
            agg_data = data.groupby([x_col, color])[y_col].mean().reset_index()
        else:
            agg_data = data.groupby(x_col)[y_col].mean().reset_index()
        
        # Determinar qual biblioteca de visualização usar
        if output_format == "HTML Interativo":
            # Usar Plotly para saída HTML interativa
            if orientation == "Horizontal":
                fig = px.bar(
                    agg_data,
                    y=x_col,
                    x=y_col,
                    color=color,
                    title=title,
                    orientation='h',
                    color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()),
                    width=width,
                    height=height
                )
            else:
                fig = px.bar(
                    agg_data,
                    x=x_col,
                    y=y_col,
                    color=color,
                    title=title,
                    color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()),
                    width=width,
                    height=height
                )
            
            # Exibir visualização
            st.plotly_chart(fig, use_container_width=True)
            
            # Gerar HTML para baixar
            html = fig.to_html(include_plotlyjs="cdn")
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="grafico_barras.html">Baixar Gráfico Interativo (HTML)</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            # Usar Matplotlib para saída de imagem estática
            plt.figure(figsize=(width/100, height/100))
            
            if orientation == "Horizontal":
                if color:
                    grouped_data = agg_data.groupby(x_col)[y_col].mean()
                    ax = grouped_data.sort_values().plot.barh(color=plt.cm.__getattribute__(color_scheme.lower())(np.linspace(0, 1, len(grouped_data))))
                else:
                    ax = agg_data.sort_values(by=y_col).plot.barh(x=x_col, y=y_col, legend=False, color=plt.cm.__getattribute__(color_scheme.lower())(0.6))
                
                plt.xlabel(y_col)
                plt.ylabel(x_col)
            else:
                if color:
                    pivot_data = pd.pivot_table(agg_data, values=y_col, index=x_col, columns=color)
                    ax = pivot_data.plot.bar(color=plt.cm.__getattribute__(color_scheme.lower())(np.linspace(0, 1, len(pivot_data.columns))))
                else:
                    ax = agg_data.plot.bar(x=x_col, y=y_col, legend=False, color=plt.cm.__getattribute__(color_scheme.lower())(0.6))
                
                plt.xlabel(x_col)
                plt.ylabel(y_col)
            
            plt.title(title)
            plt.tight_layout()
            
            # Salvar a figura em buffer
            buf = io.BytesIO()
            plt.savefig(buf, format=output_format.lower(), dpi=dpi)
            buf.seek(0)
            
            # Criar link para download
            if output_format == "PDF":
                mime_type = "application/pdf"
            elif output_format == "SVG":
                mime_type = "image/svg+xml"
            else:
                mime_type = f"image/{output_format.lower()}"
            
            data_url = base64.b64encode(buf.read()).decode('utf-8')
            href = f'<a href="data:{mime_type};base64,{data_url}" download="grafico_barras.{output_format.lower()}">Baixar Gráfico ({output_format})</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Exibir a figura na interface
            st.pyplot(plt.gcf())
            plt.close()

    elif viz_type == "Gráfico de Linhas":
        # Configuração para gráfico de linhas
        # Verificar se temos colunas de data/hora
        datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        
        if datetime_cols:
            # Priorizar colunas de data para o eixo X
            x_col = st.selectbox("Selecione o eixo X (preferencialmente data):", datetime_cols + [col for col in data.columns if col not in datetime_cols])
        else:
            x_col = st.selectbox("Selecione o eixo X:", data.columns)
        
        # Permitir múltiplas colunas de valor
        y_cols = st.multiselect("Selecione o eixo Y (Valores, múltiplos permitidos):", st.session_state.numeric_columns)
        
        # Opções adicionais
        with st.expander("Opções do Gráfico"):
            # Opções de estilo
            title = st.text_input("Título do gráfico:", f"Tendência de {', '.join(y_cols)} ao longo de {x_col}")
            color_scheme = st.selectbox("Esquema de Cores:", 
                                    ["tab10", "Set1", "Set2", "Pastel1", "viridis", "plasma", "inferno", "magma", "cividis"])
            
            # Estilo de linha
            line_style = st.selectbox("Estilo de Linha:", ["Sólida", "Tracejada", "Pontilhada", "Traço e Ponto"])
            marker = st.checkbox("Mostrar Marcadores", value=True)
            
            # Mapeamento para estilos matplotlib
            style_map = {
                "Sólida": "-",
                "Tracejada": "--",
                "Pontilhada": ":",
                "Traço e Ponto": "-."
            }
            
            # Opções de tamanho
            width = st.slider("Largura (pixels):", 400, 1200, 900, 50)
            height = st.slider("Altura (pixels):", 300, 800, 500, 50)
            
            # Formato de saída
            output_format = st.selectbox("Formato de Saída:", ["PNG", "JPEG", "SVG", "PDF", "HTML Interativo"])
            dpi = st.slider("Resolução (DPI):", 72, 300, 150, 10) if output_format != "HTML Interativo" else 150
        
        if x_col and y_cols:
            # Preparar dados (ordenar por X se necessário)
            plot_data = data.copy()
            if pd.api.types.is_numeric_dtype(data[x_col]) or pd.api.types.is_datetime64_any_dtype(data[x_col]):
                plot_data = plot_data.sort_values(by=x_col)
            
            # Determinar qual biblioteca de visualização usar
            if output_format == "HTML Interativo":
                # Usar Plotly para saída HTML interativa
                fig = px.line(
                    plot_data,
                    x=x_col,
                    y=y_cols,
                    title=title,
                    width=width,
                    height=height,
                    markers=marker
                )
                
                # Ajustar linhas e layout
                for i, trace in enumerate(fig.data):
                    trace.line.dash = "solid" if line_style == "Sólida" else "dash" if line_style == "Tracejada" else "dot" if line_style == "Pontilhada" else "dashdot"
                
                # Exibir visualização
                st.plotly_chart(fig, use_container_width=True)
                
                # Gerar HTML para baixar
                html = fig.to_html(include_plotlyjs="cdn")
                b64 = base64.b64encode(html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="grafico_linhas.html">Baixar Gráfico Interativo (HTML)</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                # Usar Matplotlib para saída de imagem estática
                plt.figure(figsize=(width/100, height/100))
                
                # Obter mapa de cores
                if color_scheme in ["tab10", "Set1", "Set2", "Pastel1"]:
                    cmap = plt.cm.get_cmap(color_scheme)
                    colors = [cmap(i) for i in range(len(y_cols))]
                else:
                    cmap = plt.cm.get_cmap(color_scheme)
                    colors = [cmap(i) for i in np.linspace(0, 1, len(y_cols))]
                
                for i, y_col in enumerate(y_cols):
                    plt.plot(
                        plot_data[x_col], 
                        plot_data[y_col], 
                        label=y_col,
                        linestyle=style_map[line_style],
                        marker='o' if marker else None,
                        color=colors[i]
                    )
                
                plt.xlabel(x_col)
                plt.ylabel("Valor")
                plt.title(title)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Rotacionar rótulos do eixo X se for data ou texto
                if pd.api.types.is_datetime64_any_dtype(data[x_col]) or pd.api.types.is_object_dtype(data[x_col]):
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                # Salvar a figura em buffer
                buf = io.BytesIO()
                plt.savefig(buf, format=output_format.lower(), dpi=dpi)
                buf.seek(0)
                
                # Criar link para download
                if output_format == "PDF":
                    mime_type = "application/pdf"
                elif output_format == "SVG":
                    mime_type = "image/svg+xml"
                else:
                    mime_type = f"image/{output_format.lower()}"
                
                data_url = base64.b64encode(buf.read()).decode('utf-8')
                href = f'<a href="data:{mime_type};base64,{data_url}" download="grafico_linhas.{output_format.lower()}">Baixar Gráfico ({output_format})</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Exibir a figura na interface
                st.pyplot(plt.gcf())
                plt.close()
    
    elif viz_type == "Gráfico de Dispersão":
        # Configuração para gráfico de dispersão
        x_col = st.selectbox("Selecione o eixo X:", st.session_state.numeric_columns)
        y_col = st.selectbox("Selecione o eixo Y:", st.session_state.numeric_columns, index=min(1, len(st.session_state.numeric_columns)-1))
        
        # Opções adicionais
        with st.expander("Opções do Gráfico"):
            # Opções de agrupamento
            color_col = st.selectbox("Coluna para cores (opcional):", ["Nenhuma"] + st.session_state.categorical_columns)
            size_col = st.selectbox("Coluna para tamanho (opcional):", ["Nenhuma"] + st.session_state.numeric_columns)
            
            color = None if color_col == "Nenhuma" else color_col
            size = None if size_col == "Nenhuma" else size_col
            
            # Opções de linha de tendência
            add_trendline = st.checkbox("Adicionar Linha de Tendência", value=True)
            
            # Opções de estilo
            title = st.text_input("Título do gráfico:", f"Dispersão de {y_col} vs {x_col}")
            opacity = st.slider("Opacidade dos Pontos:", 0.0, 1.0, 0.7, 0.1)
            color_scheme = st.selectbox("Esquema de Cores:", 
                                    ["Viridis", "Plasma", "Blues", "Reds", "Greens", "Rainbow", "Cividis"])
            
            # Opções de tamanho
            width = st.slider("Largura (pixels):", 400, 1200, 800, 50)
            height = st.slider("Altura (pixels):", 300, 800, 600, 50)
            
            # Formato de saída
            output_format = st.selectbox("Formato de Saída:", ["PNG", "JPEG", "SVG", "PDF", "HTML Interativo"])
            dpi = st.slider("Resolução (DPI):", 72, 300, 150, 10) if output_format != "HTML Interativo" else 150
        
        if x_col and y_col:
            # Determinar qual biblioteca de visualização usar
            if output_format == "HTML Interativo":
                # Usar Plotly para saída HTML interativa
                fig = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    color=color,
                    size=size,
                    title=title,
                    opacity=opacity,
                    color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()),
                    width=width,
                    height=height,
                    trendline="ols" if add_trendline else None
                )
                
                # Exibir visualização
                st.plotly_chart(fig, use_container_width=True)
                
                # Gerar HTML para baixar
                html = fig.to_html(include_plotlyjs="cdn")
                b64 = base64.b64encode(html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="grafico_dispersao.html">Baixar Gráfico Interativo (HTML)</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                # Usar Matplotlib para saída de imagem estática
                plt.figure(figsize=(width/100, height/100))
                
                # Plotar pontos
                if color:
                    if pd.api.types.is_numeric_dtype(data[color]):
                        # Usar colormap para valor numérico
                        scatter = plt.scatter(
                            data[x_col], 
                            data[y_col], 
                            c=data[color],
                            alpha=opacity,
                            s=100 if not size else data[size]*20,
                            cmap=color_scheme.lower()
                        )
                        plt.colorbar(scatter, label=color)
                    else:
                        # Agrupar por categoria
                        for i, (name, group) in enumerate(data.groupby(color)):
                            plt.scatter(
                                group[x_col], 
                                group[y_col], 
                                label=name,
                                alpha=opacity,
                                s=100 if not size else group[size]*20
                            )
                        plt.legend(title=color)
                else:
                    plt.scatter(
                        data[x_col], 
                        data[y_col], 
                        alpha=opacity,
                        s=100 if not size else data[size]*20,
                        color=plt.cm.__getattribute__(color_scheme.lower())(0.6)
                    )
                
                # Adicionar linha de tendência
                if add_trendline:
                    # Remover NaN
                    valid_data = data[[x_col, y_col]].dropna()
                    
                    if len(valid_data) > 1:
                        from scipy import stats
                        
                        # Calcular linha de regressão
                        slope, intercept, r_value, _, _ = stats.linregress(valid_data[x_col], valid_data[y_col])
                        
                        # Plotar linha
                        x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
                        plt.plot(x_range, intercept + slope * x_range, 'r-', 
                                label=f'y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})')
                        plt.legend()
                
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Salvar a figura em buffer
                buf = io.BytesIO()
                plt.savefig(buf, format=output_format.lower(), dpi=dpi)
                buf.seek(0)
                
                # Criar link para download
                if output_format == "PDF":
                    mime_type = "application/pdf"
                elif output_format == "SVG":
                    mime_type = "image/svg+xml"
                else:
                    mime_type = f"image/{output_format.lower()}"
                
                data_url = base64.b64encode(buf.read()).decode('utf-8')
                href = f'<a href="data:{mime_type};base64,{data_url}" download="grafico_dispersao.{output_format.lower()}">Baixar Gráfico ({output_format})</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Exibir a figura na interface
                st.pyplot(plt.gcf())
                plt.close()
    
    # Opções para outros tipos de gráficos podem ser adicionadas de forma semelhante...
    elif viz_type in ["Histograma", "Boxplot", "Mapa de Calor de Correlação"]:
        st.info(f"A funcionalidade de exportação para {viz_type} está disponível. Configure suas opções e clique em 'Gerar Visualização'.")
        
        # Configurações comuns para esses tipos
        if viz_type == "Histograma":
            value_col = st.selectbox("Selecione a Coluna para Histograma:", st.session_state.numeric_columns)
            bins = st.slider("Número de Intervalos:", 5, 100, 20)
            
            with st.expander("Opções do Gráfico"):
                # Opções de estilo
                title = st.text_input("Título do gráfico:", f"Histograma de {value_col}")
                color = st.color_picker("Cor das Barras:", "#3498db")
                
                show_kde = st.checkbox("Mostrar Curva de Densidade (KDE)", value=True)
                kde_color = st.color_picker("Cor da Curva KDE:", "#e74c3c") if show_kde else "#e74c3c"
                
                # Opções de tamanho
                width = st.slider("Largura (pixels):", 400, 1200, 800, 50)
                height = st.slider("Altura (pixels):", 300, 800, 500, 50)
                
                # Formato de saída
                output_format = st.selectbox("Formato de Saída:", ["PNG", "JPEG", "SVG", "PDF", "HTML Interativo"])
                dpi = st.slider("Resolução (DPI):", 72, 300, 150, 10) if output_format != "HTML Interativo" else 150
        
            if st.button("Gerar Visualização"):
                if value_col:
                    # Determinar qual biblioteca de visualização usar
                    if output_format == "HTML Interativo":
                        # Usar Plotly para saída HTML interativa
                        fig = px.histogram(
                            data,
                            x=value_col,
                            title=title,
                            nbins=bins,
                            opacity=0.7,
                            width=width,
                            height=height
                        )
                        
                        # Adicionar curva KDE se solicitado
                        if show_kde:
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
                                hist_vals = np.histogram(kde_data, bins=bins)[0]
                                scale_factor = max(hist_vals) / max(y_kde)
                                y_kde = y_kde * scale_factor
                                
                                # Adicionar curva KDE ao gráfico
                                fig.add_scatter(
                                    x=x_range,
                                    y=y_kde,
                                    mode='lines',
                                    line=dict(color=kde_color, width=2),
                                    name='Densidade'
                                )
                        
                        # Exibir visualização
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Gerar HTML para baixar
                        html = fig.to_html(include_plotlyjs="cdn")
                        b64 = base64.b64encode(html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="histograma.html">Baixar Histograma Interativo (HTML)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        # Usar Matplotlib/Seaborn para saída de imagem estática
                        plt.figure(figsize=(width/100, height/100))
                        
                        # Plotar histograma com ou sem KDE
                        if show_kde:
                            import seaborn as sns
                            sns.histplot(data[value_col].dropna(), bins=bins, kde=True, color=color, kde_kws={'color': kde_color})
                        else:
                            plt.hist(data[value_col].dropna(), bins=bins, color=color, alpha=0.7)
                        
                        plt.xlabel(value_col)
                        plt.ylabel("Frequência")
                        plt.title(title)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        # Salvar a figura em buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format=output_format.lower(), dpi=dpi)
                        buf.seek(0)
                        
                        # Criar link para download
                        if output_format == "PDF":
                            mime_type = "application/pdf"
                        elif output_format == "SVG":
                            mime_type = "image/svg+xml"
                        else:
                            mime_type = f"image/{output_format.lower()}"
                        
                        data_url = base64.b64encode(buf.read()).decode('utf-8')
                        href = f'<a href="data:{mime_type};base64,{data_url}" download="histograma.{output_format.lower()}">Baixar Histograma ({output_format})</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Exibir a figura na interface
                        st.pyplot(plt.gcf())
                        plt.close()
        
        elif viz_type == "Boxplot":
            y_col = st.selectbox("Selecione a Coluna de Valor:", st.session_state.numeric_columns)
            x_col = st.selectbox("Selecione a Coluna de Agrupamento (opcional):", ["Nenhuma"] + st.session_state.categorical_columns)
            
            with st.expander("Opções do Gráfico"):
                # Opções de estilo
                title = st.text_input("Título do gráfico:", f"Boxplot de {y_col}" + (f" por {x_col}" if x_col != "Nenhuma" else ""))
                orientation = st.radio("Orientação:", ["Vertical", "Horizontal"])
                color_scheme = st.selectbox("Esquema de Cores:", 
                                        ["Viridis", "Plasma", "Blues", "Reds", "Greens", "Set1", "Set2", "Pastel1"])
                
                # Opções de personalização
                notch = st.checkbox("Mostrar Notch (Intervalo de Confiança da Mediana)", value=False)
                show_points = st.checkbox("Mostrar Pontos", value=True)
                
                # Opções de tamanho
                width = st.slider("Largura (pixels):", 400, 1200, 800, 50)
                height = st.slider("Altura (pixels):", 300, 800, 500, 50)
                
                # Formato de saída
                output_format = st.selectbox("Formato de Saída:", ["PNG", "JPEG", "SVG", "PDF", "HTML Interativo"])
                dpi = st.slider("Resolução (DPI):", 72, 300, 150, 10) if output_format != "HTML Interativo" else 150
            
            if st.button("Gerar Visualização"):
                if y_col:
                    # Preparar variáveis
                    x = None if x_col == "Nenhuma" else x_col
                    
                    # Determinar qual biblioteca de visualização usar
                    if output_format == "HTML Interativo":
                        # Usar Plotly para saída HTML interativa
                        if orientation == "Horizontal":
                            fig = px.box(
                                data,
                                y=x,
                                x=y_col,
                                title=title,
                                points="all" if show_points else False,
                                notched=notch,
                                color=x,
                                color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()) if x else None,
                                width=width,
                                height=height
                            )
                        else:
                            fig = px.box(
                                data,
                                x=x,
                                y=y_col,
                                title=title,
                                points="all" if show_points else False,
                                notched=notch,
                                color=x,
                                color_discrete_sequence=px.colors.sequential.__getattribute__(color_scheme.lower()) if x else None,
                                width=width,
                                height=height
                            )
                        
                        # Exibir visualização
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Gerar HTML para baixar
                        html = fig.to_html(include_plotlyjs="cdn")
                        b64 = base64.b64encode(html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="boxplot.html">Baixar Boxplot Interativo (HTML)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        # Usar Matplotlib/Seaborn para saída de imagem estática
                        plt.figure(figsize=(width/100, height/100))
                        
                        import seaborn as sns
                        
                        if x:
                            if orientation == "Horizontal":
                                sns.boxplot(x=y_col, y=x, data=data, palette=color_scheme.lower(), notch=notch, 
                                           showfliers=show_points, orient='h')
                            else:
                                sns.boxplot(x=x, y=y_col, data=data, palette=color_scheme.lower(), notch=notch,
                                           showfliers=show_points)
                        else:
                            if orientation == "Horizontal":
                                sns.boxplot(x=y_col, data=data, color=plt.cm.__getattribute__(color_scheme.lower())(0.6), 
                                          notch=notch, showfliers=show_points, orient='h')
                            else:
                                sns.boxplot(y=y_col, data=data, color=plt.cm.__getattribute__(color_scheme.lower())(0.6),
                                          notch=notch, showfliers=show_points)
                        
                        plt.title(title)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        # Salvar a figura em buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format=output_format.lower(), dpi=dpi)
                        buf.seek(0)
                        
                        # Criar link para download
                        if output_format == "PDF":
                            mime_type = "application/pdf"
                        elif output_format == "SVG":
                            mime_type = "image/svg+xml"
                        else:
                            mime_type = f"image/{output_format.lower()}"
                        
                        data_url = base64.b64encode(buf.read()).decode('utf-8')
                        href = f'<a href="data:{mime_type};base64,{data_url}" download="boxplot.{output_format.lower()}">Baixar Boxplot ({output_format})</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Exibir a figura na interface
                        st.pyplot(plt.gcf())
                        plt.close()
        
        elif viz_type == "Mapa de Calor de Correlação":
            # Selecionar colunas para correlação
            corr_columns = st.multiselect(
                "Selecione as colunas numéricas para o mapa de calor:",
                st.session_state.numeric_columns,
                default=st.session_state.numeric_columns[:min(8, len(st.session_state.numeric_columns))]
            )
            
            with st.expander("Opções do Gráfico"):
                # Opções de correlação
                corr_method = st.selectbox("Método de Correlação:", ["Pearson", "Spearman", "Kendall"])
                
                # Opções de estilo
                title = st.text_input("Título do gráfico:", "Mapa de Calor de Correlação")
                color_scheme = st.selectbox("Esquema de Cores:", 
                                        ["coolwarm", "viridis", "plasma", "RdBu", "RdBu_r", "BrBG", "PiYG", "PRGn", "PuOr", "RdGy", "RdYlBu", "RdYlGn", "Spectral"])
                show_values = st.checkbox("Mostrar Valores", value=True)
                
                # Opções de tamanho
                width = st.slider("Largura (pixels):", 400, 1200, 800, 50)
                height = st.slider("Altura (pixels):", 300, 800, 700, 50)
                
                # Formato de saída
                output_format = st.selectbox("Formato de Saída:", ["PNG", "JPEG", "SVG", "PDF", "HTML Interativo"])
                dpi = st.slider("Resolução (DPI):", 72, 300, 150, 10) if output_format != "HTML Interativo" else 150
            
            if st.button("Gerar Visualização") and corr_columns and len(corr_columns) >= 2:
                # Calcular matriz de correlação
                method_map = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}
                corr_matrix = data[corr_columns].corr(method=method_map[corr_method])
                
                # Determinar qual biblioteca de visualização usar
                if output_format == "HTML Interativo":
                    # Usar Plotly para saída HTML interativa
                    fig = px.imshow(
                        corr_matrix,
                        x=corr_columns,
                        y=corr_columns,
                        color_continuous_scale=color_scheme,
                        zmin=-1,
                        zmax=1,
                        title=title,
                        width=width,
                        height=height
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
                    
                    # Exibir visualização
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gerar HTML para baixar
                    html = fig.to_html(include_plotlyjs="cdn")
                    b64 = base64.b64encode(html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="correlacao.html">Baixar Mapa de Calor Interativo (HTML)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    # Usar Matplotlib/Seaborn para saída de imagem estática
                    plt.figure(figsize=(width/100, height/100))
                    
                    import seaborn as sns
                    
                    # Criar mapa de calor
                    mask = None
                    if not st.checkbox("Mostrar valores duplicados", value=True):
                        mask = np.zeros_like(corr_matrix)
                        mask[np.triu_indices_from(mask, k=1)] = True
                    
                    ax = sns.heatmap(
                        corr_matrix,
                        annot=show_values,
                        mask=mask,
                        cmap=color_scheme,
                        vmin=-1,
                        vmax=1,
                        fmt=".2f",
                        linewidths=0.5
                    )
                    
                    plt.title(title)
                    plt.tight_layout()
                    
                    # Salvar a figura em buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format=output_format.lower(), dpi=dpi)
                    buf.seek(0)
                    
                    # Criar link para download
                    if output_format == "PDF":
                        mime_type = "application/pdf"
                    elif output_format == "SVG":
                        mime_type = "image/svg+xml"
                    else:
                        mime_type = f"image/{output_format.lower()}"
                    
                    data_url = base64.b64encode(buf.read()).decode('utf-8')
                    href = f'<a href="data:{mime_type};base64,{data_url}" download="correlacao.{output_format.lower()}">Baixar Mapa de Calor ({output_format})</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Exibir a figura na interface
                    st.pyplot(plt.gcf())
                    plt.close()
    
    # Informações adicionais para exportação
    st.subheader("Dicas para Visualizações")
    st.markdown("""
    - **Exportação em HTML Interativo**: Permite interagir com o gráfico mesmo depois de exportado.
    - **Formatos de Imagem**: PNG é o melhor para apresentações e relatórios; SVG é ideal para publicações de alta qualidade.
    - **DPI (Pontos por Polegada)**: Valores mais altos produzem imagens de melhor qualidade, mas com tamanho de arquivo maior.
    - **Recomendações**:
        - Use PNG para uso geral e apresentações
        - Use SVG para publicações científicas ou impressão
        - Use HTML Interativo para compartilhar resultados interativos
    """)

# Opções de exportação adicionais no rodapé
st.markdown("---")
st.markdown("### Exportação do Projeto Completo")

with st.expander("Opções Avançadas de Exportação"):
    st.markdown("""
    Você pode exportar todo o trabalho de análise para reutilização futura ou compartilhamento com colegas.
    """)
    
    # Opções de exportação de projeto
    export_project_option = st.radio(
        "O que incluir no pacote de exportação:",
        ["Apenas Dados", "Dados + Metadados", "Projeto Completo (Dados + Configurações + Visualizações)"]
    )
    
    # Formato do projeto
    project_format = st.selectbox(
        "Formato do Pacote:",
        ["ZIP", "Pasta Compactada"]
    )
    
    # Adicionar descrição
    project_description = st.text_area(
        "Descrição do Projeto (opcional):",
        "Projeto de análise de dados criado com o aplicativo de Análise e Visualização de Dados."
    )
    
    if st.button("Preparar Exportação do Projeto"):
        st.info("Funcionalidade de exportação do projeto em desenvolvimento. Em uma versão futura, você poderá exportar todo o projeto em um único arquivo.")