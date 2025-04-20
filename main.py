import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from mochilaACO import KnapsackProblem, ACOKnapsack
from plot import plot_progress, display_results, average_convergence, average_time

def leer_datos_excel(archivo):
    """
    Lee los datos del problema desde un archivo Excel

    :param archivo: Ruta al archivo Excel
    :return: Tupla (weights, values, quantities, capacity)
    """
    df = pd.read_excel(archivo)

    weights = df['Peso_kg'].tolist()
    values = df['Valor'].tolist()
    quantities = df['Cantidad'].tolist()

    # Se puede modificar
    capacity = 2.45  # capacidad de la mochila, 2.45kg por defecto

    return weights, values, quantities, capacity

def main():
    # Datos del problema
    archivo = 'Mochila_capacidad_maxima_2.45kg.xlsx'
    weights, values, quantities, capacity = leer_datos_excel(archivo)

    iter = 30  # Cambiado a 30 ejecuciones
    resultados = []
    best_generation = [[] for _ in range(iter)]
    tiempos = []
    converge = 0
    

    for i in range(iter):
        problem = KnapsackProblem(weights, values, quantities, capacity)
        aco = ACOKnapsack(
            ant_count=30,
            generations=100,
            alpha=0.8,
            beta=1.5,
            gamma=2,
            rho=1,
            q=1
        )
        end_time = 0

        start_time = time.time()
        solution, best_generation[i] = aco.solve(problem)
        end_time = time.time() - start_time
        tiempos.append(end_time)
        
        converge = detectar_convergencia(best_generation[i])

        resultados.append({
            'iteracion': i + 1,
            'solution': solution,
            'execution_time': end_time,
            'convergence_generation': converge
        })
        
        

        print(f" Iteración {i + 1}:")
        print(f"  Mejor solución: {solution}")
        print(f"  Tiempo de ejecución: {end_time :.2f} s")
        print(f"  Empieza a converger en la generacion: {converge}")
        print("-" * 40)

    display_results(resultados)
    #average_convergence(best_generation[1], aco.generations)
    average_time(tiempos)

    average_convergence(best_generation)
    
    
def detectar_convergencia(best_generations):
    max_val = max(best_generations)
    for i in range(len(best_generations)):
        if all(v == max_val for v in best_generations[i:]):
            return i + 1  
    return len(best_generations)

if __name__ == "__main__":
    main()