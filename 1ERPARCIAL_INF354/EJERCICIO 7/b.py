import random

# Definir funciones básicas del algoritmo genético

def crear_individuo(tamano):
    """Crea un individuo como una lista de bits (0 o 1)."""
    return [random.randint(0, 1) for _ in range(tamano)]

def crear_poblacion(n_individuos, tamano_individuo):
    """Crea una población de individuos."""
    return [crear_individuo(tamano_individuo) for _ in range(n_individuos)]

def evaluar_individuo(individuo):
    """Función de evaluación que retorna la suma de unos en el individuo."""
    return sum(individuo)

def seleccionar_padres(poblacion, fitness_poblacion):
    """Selección por torneo."""
    seleccionados = []
    for _ in range(len(poblacion)):
        # Selección de dos individuos al azar
        i1, i2 = random.sample(range(len(poblacion)), 2)
        # Seleccionar el de mayor fitness
        if fitness_poblacion[i1] > fitness_poblacion[i2]:
            seleccionados.append(poblacion[i1])
        else:
            seleccionados.append(poblacion[i2])
    return seleccionados

def cruzar_individuos(padre1, padre2):
    """Cruce de un punto entre dos padres."""
    punto = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

def mutar_individuo(individuo, prob_mutacion):
    """Mutación de bits en el individuo."""
    for i in range(len(individuo)):
        if random.random() < prob_mutacion:
            individuo[i] = 1 - individuo[i]  # Invierte el bit (0 -> 1, 1 -> 0)
    return individuo

# Definir el algoritmo genético

def algoritmo_genetico(n_individuos, tamano_individuo, n_generaciones, prob_cruce, prob_mutacion):
    # Crear población inicial
    poblacion = crear_poblacion(n_individuos, tamano_individuo)
    
    for gen in range(n_generaciones):
        # Evaluar la población
        fitness_poblacion = [evaluar_individuo(ind) for ind in poblacion]
        
        # Registrar estadísticas
        promedio = sum(fitness_poblacion) / len(fitness_poblacion)
        minimo = min(fitness_poblacion)
        maximo = max(fitness_poblacion)
        
        print(f"Generación {gen}: Promedio: {promedio}, Mínimo: {minimo}, Máximo: {maximo}")
        
        # Selección
        padres = seleccionar_padres(poblacion, fitness_poblacion)
        
        # Cruzar la población seleccionada
        nueva_poblacion = []
        for i in range(0, len(padres), 2):
            if i + 1 < len(padres) and random.random() < prob_cruce:
                hijo1, hijo2 = cruzar_individuos(padres[i], padres[i+1])
                nueva_poblacion.append(hijo1)
                nueva_poblacion.append(hijo2)
            else:
                nueva_poblacion.append(padres[i])
                if i + 1 < len(padres):
                    nueva_poblacion.append(padres[i+1])
        
        # Mutación
        poblacion = [mutar_individuo(ind, prob_mutacion) for ind in nueva_poblacion]
    
    # Retornar la población final
    return poblacion

# Ejecutar el algoritmo genético

if __name__ == "__main__":
    # Parámetros
    n_individuos = 13
    tamano_individuo = 100
    n_generaciones = 50
    prob_cruce = 0.8
    prob_mutacion = 0.05
    
    # Ejecutar el algoritmo genético
    poblacion_final = algoritmo_genetico(n_individuos, tamano_individuo, n_generaciones, prob_cruce, prob_mutacion)
