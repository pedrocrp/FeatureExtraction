import matplotlib.pyplot as plt
import pandas as pd

# Dados fornecidos
dados = {
    'Classe': ['Normal', 'COVID', 'Viral_Pneumonia', 'Lung_Opacity'],
    'Treino': [8153, 2892, 1076, 4809],
    'Teste': [2039, 724, 269, 1203]
}

# Criar DataFrame
df = pd.DataFrame(dados)

# Gráfico de barras agrupadas
plt.figure(figsize=(10, 6))
largura_barra = 0.35
x = range(len(df))

plt.bar([i - largura_barra/2 for i in x], df['Treino'], width=largura_barra, label='Treino')
plt.bar([i + largura_barra/2 for i in x], df['Teste'], width=largura_barra, label='Teste')

# Rótulos e ajustes
plt.xticks(ticks=x, labels=df['Classe'], rotation=15)
plt.xlabel('Classe')
plt.ylabel('Número de Imagens')
plt.title('Distribuição de Imagens por Classe (Treino vs Teste)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
