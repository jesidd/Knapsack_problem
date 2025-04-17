import pandas as pd
import time
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
    capacity = 2.45  # capacidad de la mochila, 28kg por defecto

    return weights, values, quantities, capacity

import time  # Asegúrate de importar time si no está ya importado

def main():
    # Datos del problema
    archivo = 'Mochila_capacidad_maxima_2.45kg.xlsx'  # nombre del archivo
    weights, values, quantities, capacity = leer_datos_excel(archivo)    
    
    #numero de iteraciones
    iter = 10
    # Ejecutar el problema 10 veces y guardar los datos
    resultados = []
    
    best_generation = [[] for _ in range(iter)]
    
    for i in range(iter):
        # Crear instancia del problema
        problem = KnapsackProblem(weights, values, quantities, capacity)
        # Configurar algoritmo ACO (nueva instancia en cada iteración)
        aco = ACOKnapsack(
            ant_count=30,
            generations=100,
            alpha=0.8,
            beta=1.5,
            gamma=2.5,
            rho=0.9,
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
        print(f"  Tiempo de ejecución: {end_time - start_time}")
        print(f"  Generación de convergencia: {aco.convergence_generation}")
        print("-" * 40)
    
    

    display_results(resultados)
    average_convergence(best_generation[1],aco.generations)
    #convergence(aco)
    

if __name__ == "__main__":
    main()