import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from mochilaACO import KnapsackProblem, ACOKnapsack
from plot import plot_progress, display_results, average_convergence, convergence

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

    for i in range(iter):
        problem = KnapsackProblem(weights, values, quantities, capacity)
        aco = ACOKnapsack(
            ant_count=30,
            generations=100,
            alpha=1,
            beta=1.5,
            gamma=2.5,
            rho=0.8,
            q=2
        )

        start_time = time.time()
        solution, best_generation[i] = aco.solve(problem)
        end_time = time.time()

        resultados.append({
            'iteracion': i + 1,
            'solution': solution,
            'execution_time': end_time - start_time,
            'convergence_generation': aco.convergence_generation
        })

        print(f" Iteración {i + 1}:")
        print(f"  Mejor solución: {solution}")
        print(f"  Tiempo de ejecución: {end_time - start_time:.2f} s")
        print(f"  Generación de convergencia: {aco.convergence_generation}")
        print("-" * 40)

    display_results(resultados)
    #average_convergence(best_generation[1], aco.generations)

    
    average_convergence(best_generation)

if __name__ == "__main__":
    main()