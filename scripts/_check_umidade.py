import pandas as pd

df = pd.read_csv('base_de_dados_com_umidade.csv')
print(f'Total linhas: {len(df)}')
nao_nulos = df['Umidade'].notna().sum()
nulos = df['Umidade'].isna().sum()
print(f'Umidade nao nula: {nao_nulos} ({nao_nulos/len(df)*100:.1f}%)')
print(f'Umidade NaN: {nulos} ({nulos/len(df)*100:.1f}%)')
print(f'Stats Umidade:')
print(df['Umidade'].describe())
