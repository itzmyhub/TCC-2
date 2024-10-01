import pandas as pd

def gerar_relatorio_dados_csv(arquivo_csv, arquivo_saida):
    # Carregar o arquivo CSV
    dados = pd.read_csv(arquivo_csv)

    # Criar um dicionário para armazenar as informações dos atributos
    relatorio_atributos = []

    for coluna in dados.columns:
        # Obter o tipo de dado
        tipo_dado = dados[coluna].dtype
        
        # Calcular estatísticas descritivas para atributos numéricos
        if pd.api.types.is_numeric_dtype(dados[coluna]):
            estatisticas = dados[coluna].describe()
            estatisticas_dict = estatisticas.to_dict()
        else:
            estatisticas_dict = {}

        # Contar valores únicos e valores ausentes
        valores_unicos = dados[coluna].nunique()
        valores_ausentes = dados[coluna].isnull().sum()
        
        # Adicionar informações ao relatório
        relatorio_atributos.append({
            'Atributo': coluna,
            'Tipo de Dado': tipo_dado,
            'Valores Únicos': valores_unicos,
            'Valores Ausentes': valores_ausentes,
            'Estatísticas Descritivas': estatisticas_dict
        })

    # Converter o relatório para um DataFrame
    df_relatorio = pd.DataFrame(relatorio_atributos)
    
    # Exportar o DataFrame para um arquivo CSV
    df_relatorio.to_csv(arquivo_saida, index=False)
    print(f"Relatório exportado para {arquivo_saida}")

# Caminho para o arquivo CSV e arquivo de saída
arquivo_csv = 'dados_incendio_tratados.csv'
arquivo_saida = 'relatorio_atributos_tratados.csv'
gerar_relatorio_dados_csv(arquivo_csv, arquivo_saida)
