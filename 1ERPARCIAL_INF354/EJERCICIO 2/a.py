import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, gamma

datos_vote_average = np.array(tbl['vote_average'].dropna())
datos_vote_count = np.array(tbl['vote_count'].dropna())
datos_popularity = np.array(tbl['popularity'].dropna())

#Calcular percentiles manualmente
def calcular_percentil(datos, percentil):
    datos_ordenados = sorted(datos)
    n = len(datos_ordenados)
    indice = (percentil / 100) * (n - 1)
    inferior = int(indice)
    superior = inferior + 1
    peso = indice - inferior

    if superior < n:
        return datos_ordenados[inferior] * (1 - peso) + datos_ordenados[superior] * peso
    else:
        return datos_ordenados[inferior]

# Calcular cuartiles (Q1, Q2, Q3)
def calcular_cuartiles(datos):
    Q1 = calcular_percentil(datos, 25)
    Q2 = calcular_percentil(datos, 50)
    Q3 = calcular_percentil(datos, 75)
    return Q1, Q2, Q3

cuartiles_vote_count = calcular_cuartiles(datos_vote_count)
cuartiles_popularity = calcular_cuartiles(datos_popularity)

#Graficar las distribuciones ajustadas

fig, ejes = plt.subplots(1, 3, figsize=(18, 5))

# Distribución normal para vote_average
media, desviacion = norm.fit(datos_vote_average)
x1 = np.linspace(min(datos_vote_average), max(datos_vote_average), 100)
pdf_normal = norm.pdf(x1, media, desviacion)
ejes[0].hist(datos_vote_average, bins=30, alpha=0.6, color='red', density=True)
ejes[0].plot(x1, pdf_normal, 'k-', label=f'Normal (media={media:.2f}, std={desviacion:.2f})')
ejes[0].set_title('Distribución de vote_average (Normal)')
ejes[0].legend()

# Distribución log-normal para vote_count
forma, loc, escala = lognorm.fit(datos_vote_count, floc=0)
x2 = np.linspace(min(datos_vote_count), max(datos_vote_count), 100)
pdf_lognormal = lognorm.pdf(x2, forma, loc, escala)
ejes[1].hist(datos_vote_count, bins=30, alpha=0.6, color='blue', density=True)
ejes[1].plot(x2, pdf_lognormal, 'k-', label='Log-Normal')
ejes[1].set_title('Distribución de vote_count (Log-Normal)')
ejes[1].legend()

# Distribución gamma para popularity
forma, loc, escala = gamma.fit(datos_popularity, floc=0)
x3 = np.linspace(min(datos_popularity), max(datos_popularity), 100)
pdf_gamma = gamma.pdf(x3, forma, loc, escala)
ejes[2].hist(datos_popularity, bins=30, alpha=0.6, color='green', density=True)
ejes[2].plot(x3, pdf_gamma, 'k-', label='Gamma')
ejes[2].set_title('Distribución de popularity (Gamma)')
ejes[2].legend()

plt.tight_layout()
plt.show()
