import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Asumiendo que ya tienes el dataset limpio en 'data_cleaned'
# Seleccionar las columnas de interés
columnas_seleccionadas = tbl[['vote_average', 'vote_count', 'popularity']]

# Calcular medidas de tendencia central
media = columnas_seleccionadas.mean()      # Media
mediana = columnas_seleccionadas.median()  # Mediana
moda = columnas_seleccionadas.mode().iloc[0]  # Moda

# Imprimir los resultados
print("Media:\n", media)
print("\nMediana:\n", mediana)
print("\nModa:\n", moda)

# Crear el diagrama de cajas y bigotes (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=columnas_seleccionadas)
plt.title('Diagrama de Cajas y Bigotes de vote_average, vote_count y popularity')
plt.xlabel('Columnas')
plt.ylabel('Valores')
plt.show()
