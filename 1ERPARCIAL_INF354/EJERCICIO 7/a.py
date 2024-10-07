import random
from deap import base, creator, tools, algorithms

# Crear los tipos básicos de fitness y de individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizar
creator.create("Individual", list, fitness=creator.FitnessMax)

# Definir las funciones básicas
def evalOneMax(individual):
    return sum(individual),  # Fitness es el número de unos en el individuo

def mutFlipBit(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])
    return individual,

# Configuración de DEAP
toolbox = base.Toolbox()
# Crear un gen binario aleatorio (0 o 1)
toolbox.register("attr_bool", random.randint, 0, 1)
# Crear un individuo (lista de bits)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
# Crear la población (lista de individuos)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Función de evaluación (el objetivo es maximizar el número de unos)
toolbox.register("evaluate", evalOneMax)
# Selección por torneo
toolbox.register("select", tools.selTournament, tournsize=3)
# Cruce en un punto
toolbox.register("mate", tools.cxOnePoint)
# Mutación: invertir bits
toolbox.register("mutate", mutFlipBit, indpb=0.05)

# Parámetros del algoritmo genético
def main():
    # Crear población inicial de 300 individuos
    population = toolbox.population(n=13)
    
    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Accedemos al primer valor de la tupla de fitness
    stats.register("avg", lambda x: sum(x)/len(x))
    stats.register("min", min)
    stats.register("max", max)

    # Ejecutar el algoritmo genético usando eaSimple
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, 
                                              stats=stats, verbose=True)
    return population, logbook

if __name__ == "__main__":
    main()
