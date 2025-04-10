import pandas as pd
import time
from mochilaACO import KnapsackProblem  # Reutilizamos la clase del problema
from mochilaSA import SimulatedAnnealingKnapsack  # Asegúrate de tener esta clase
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

def main():
    # Datos del problema
    archivo = 'Mochila_capacidad_maxima_28kg.xlsx'
    weights, values, quantities, capacity = leer_datos_excel(archivo)    
    
    # Ejecutar el problema 10 veces
    resultados = []
    for i in range(100):
        problem = KnapsackProblem(weights, values, quantities, capacity)
        sa = SimulatedAnnealingKnapsack(
            initial_temp=1000,
            final_temp=1,
            alpha=0.9,
            iterations_per_temp=500
        )
        
        start_time = time.time()
        solution = sa.solve(problem)
        end_time = time.time()

        resultados.append({
            'iteracion': i + 1,
            'solution': solution,
            'execution_time': end_time - start_time,
            'convergence_generation': sa.convergence_generation
        })

    # Imprimir los resultados
    for resultado in resultados:
        print(f" Iteración {resultado['iteracion']}:")
        print(f"  Mejor solución: {resultado['solution']}")
        print(f"  Tiempo de ejecución: {resultado['execution_time']:.4f} segundos")
        print(f"  Generación de convergencia: {resultado['convergence_generation']}")
        print("-" * 40)

    # Mostrar resumen y gráfico
    display_results(resultados)
    #plot_progress(sa.fitness_history, sa.convergence_generation)  # opcional

if __name__ == "__main__":
    main()
