import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_progress(fitness_history, convergence_generation=None):
    """
    Grafica el progreso del mejor valor a través de las generaciones
    
    :param fitness_history: Lista con los mejores valores de cada generación
    :param convergence_generation: Generación donde se logró convergencia
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    
    # Marcar el punto de convergencia
    if convergence_generation is not None:
        plt.axvline(x=convergence_generation, color='r', linestyle='--', 
                  label=f'Convergencia en gen {convergence_generation}')
        plt.plot(convergence_generation, fitness_history[convergence_generation-1], 
               'ro', markersize=10)
    
    plt.title('Convergencia del algoritmo ACO para el problema de la mochila')
    plt.xlabel('Generación')
    plt.ylabel('Mejor valor')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def average_convergence(average_iter, num_iter):
    plt.plot(range(1, num_iter + 1), average_iter, color='green')
    plt.xlabel('Iteración')
    plt.ylabel('Valor promedio de mejor solución')
    plt.title('Convergencia promedio del método ACO')
    plt.grid()
    plt.show()
    
def convergence(aco):
    evolution_array = np.array(aco.evolution_iter)  # 30 × 100
    promedio_generacional = evolution_array.mean(axis=0)  # Promedio en cada generación

    plt.plot(range(len(promedio_generacional)), promedio_generacional, color='red', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Promedio del mejor valor')
    plt.title('Convergencia promedio en 30 ejecuciones')
    plt.grid(True)
    plt.show()

def display_results(resultados):

    # Crear un DataFrame a partir de resultados
    df = pd.DataFrame([{
        'iteracion': r['iteracion'],
        'valor_solucion': r['solution']['value'],
        'tiempo_ejecucion': r['execution_time'],
        'convergencia': r['convergence_generation']
    } for r in resultados])

    # Calcular estadísticas
    tabla_resumen = {
        '| Promedio': df['valor_solucion'].mean(),
        '| Mínimo': df['valor_solucion'].min(),
        '| Máximo': df['valor_solucion'].max(),
        '| Varianza |': df['valor_solucion'].var()     
    }

    tabla_resumen2 = {
        '| Tiempo Promedio de Ejecución': df['tiempo_ejecucion'].mean(),
        '| Iteración Promedio de Convergencia': df['convergencia'].mean()
    }

    # Mostrar tabla como DataFrame
    resumen_df = pd.DataFrame(tabla_resumen, index=['Valor |'])
    resumen_df2 = pd.DataFrame(tabla_resumen2, index=['Tiempo/Iteración |'])
    
    print(resumen_df)
    print("\n")
    print(resumen_df2)

    # Gráfico opcional
    plt.figure(figsize=(10, 6))
    plt.bar(df['iteracion'], df['valor_solucion'], color='lightgreen')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la Solución')
    plt.title('Valor de la Solución por Iteración')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


