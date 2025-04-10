import pandas as pd
import time
from mochilaACO import KnapsackProblem, ACOKnapsack
from plot import plot_progress, display_results 

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
    capacity = 28  # capacidad de la mochila, 28kg por defecto

    return weights, values, quantities, capacity

import time  # Asegúrate de importar time si no está ya importado

def main():
    # Datos del problema
    archivo = 'Mochila_capacidad_maxima_28kg.xlsx'  # nombre del archivo
    weights, values, quantities, capacity = leer_datos_excel(archivo)    
    
    # Ejecutar el problema 10 veces y guardar los datos
    resultados = []
    for i in range(10):
        # Crear instancia del problema
        problem = KnapsackProblem(weights, values, quantities, capacity)
        # Configurar algoritmo ACO (nueva instancia en cada iteración)
        aco = ACOKnapsack(
            ant_count=50,
            generations=100,
            alpha=0.8,
            beta=1.5,
            gamma=2.5,
            rho=0.7,
            q=100
        )
        
        start_time = time.time()
        solution = aco.solve(problem)
        end_time = time.time()
        resultados.append({
            'iteracion': i + 1,
            'solution': solution,
            'execution_time': end_time - start_time,
            'convergence_generation': aco.convergence_generation
        })

    # Imprimir los resultados al finalizar
    for resultado in resultados:
        print(f" Iteración {resultado['iteracion']}:")
        print(f"  Mejor solución: {resultado['solution']}")
        print(f"  Tiempo de ejecución: {resultado['execution_time']:.4f} segundos")
        print(f"  Generación de convergencia: {resultado['convergence_generation']}")
        print("-" * 40)
    
    
    # Mostrar resultados detallados
    # display_results(
    #     solution, 
    #     weights, 
    #     values, 
    #     quantities, 
    #     capacity, 
    #     aco.execution_time, 
    #     aco.convergence_generation
    # )
    # Mostrar gráfico de progreso
    #plot_progress(aco.fitness_history, aco.convergence_generation)
    #display_total(resultados)
    display_results(resultados)

if __name__ == "__main__":
    main()

    