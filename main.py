import pandas as pd
import time
from mochilaACO import KnapsackProblem, ACOKnapsack
from plot import plot_progress, display_results, average_convergence

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
    capacity = 2.45  # capacidad de la mochila, 28kg por defecto

    return weights, values, quantities, capacity

import time  # Asegúrate de importar time si no está ya importado

def main():
    # Datos del problema
    archivo = 'Mochila_capacidad_maxima_2.45kg.xlsx'  # nombre del archivo
    weights, values, quantities, capacity = leer_datos_excel(archivo)


    # Crear instancia del problema
    problem = KnapsackProblem(weights, values, quantities, capacity)
    # Configurar algoritmo ACO (nueva instancia en cada iteración)
    aco = ACOKnapsack(
        ant_count=50,
        generations=100,
        alpha=0.8,
        beta=1,
        gamma=2,
        rho=0.6,
        q=2
    )
    
    
    solution = aco.solve(problem)


    # Imprimir los resultados al finalizar
    for resultado in solution:
        print(f" Iteración {resultado['iteracion']}:")
        print(f"  Mejor solución: {resultado['solution']}")
        print(f"  Peso: {resultado['total_weight']}")
        print(f"  Tiempo de ejecución: {resultado['execution_time']:.4f} segundos")
        print(f"  Generación de convergencia: {resultado['convergence_generation']}")
        print("-" * 40)
        

    display_results(solution)
    average_convergence(aco.evolution_iter,aco.generations)

if __name__ == "__main__":
    main()

    