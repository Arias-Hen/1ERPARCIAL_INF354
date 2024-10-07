import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Asumiendo que ya tienes tus datos limpios en el DataFrame 'data_cleaned'
# Seleccionamos las columnas relevantes para el análisis
columnas_seleccionadas = tbl[['vote_average', 'vote_count', 'popularity']]

# Gráfico de dispersión entre vote_count y vote_average, coloreado por popularity
plt.figure(figsize=(12, 6))
sns.scatterplot(data=columnas_seleccionadas, x='vote_count', y='vote_average', hue='popularity', palette='viridis')
plt.title('Gráfico de Dispersión: vote_count vs vote_average (Coloreado por popularity)')
plt.xlabel('vote_count (Cantidad de Votos)')
plt.ylabel('vote_average (Promedio de Votos)')
plt.legend(title='popularidad', loc='upper left')
plt.show()

# Crear un mapa de calor para visualizar las correlaciones entre las columnas seleccionadas
plt.figure(figsize=(8, 6))
correlaciones = columnas_seleccionadas.corr()  # Calcular la matriz de correlación
sns.heatmap(correlaciones, annot=True, cmap='coolwarm', vmin=-1, vmax=1)  # Mapa de calor con anotaciones
plt.title('Mapa de Calor: Correlaciones entre vote_average, vote_count y popularity')
plt.show()
