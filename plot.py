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
    

def average_convergence(best_generation):
    # Gráfico  Convergencia promedio
    generaciones = len(best_generation[0])
    promedio_generacional = np.mean(best_generation, axis=0)
    plt.plot(promedio_generacional, color='blue')
    plt.title("Convergencia promedio")
    plt.xlabel("Generación")
    plt.ylabel("Valor promedio del mejor individuo")
    plt.tight_layout()
    plt.show()
    

def average_time(tiempos):
    # Estadísticas
    media_tiempo = np.mean(tiempos)
    varianza_tiempo = np.var(tiempos)

    # Gráfica
    plt.figure(figsize=(10, 5))
    ejecuciones = list(range(1, len(tiempos) + 1))  # Para mostrar ejecuciones desde 1 hasta 30
    plt.plot(ejecuciones, tiempos, marker='o', linestyle='-', label='Tiempo por ejecución')
    plt.axhline(media_tiempo, color='red', linestyle='--', label=f'Media = {media_tiempo:.2f}s')
    
    plt.title("Tiempo Total por Ejecución (30 corridas)")
    plt.xlabel("Ejecución")
    plt.ylabel("Tiempo (segundos)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def average_ejecution(best_generation):
    mejores_finales = [max(ejecucion) for ejecucion in best_generation]
    promedio_finales = np.mean(mejores_finales)
    plt.plot(mejores_finales, marker='o')
    plt.axhline(promedio_finales, color='r', linestyle='--', label='Promedio')
    plt.title("Solución final por ejecución")
    plt.xlabel("Ejecución")
    plt.ylabel("Mejor valor final")
    plt.legend()
    plt.tight_layout()
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
        '| Iteración Promedio de Convergencia': df['convergencia'].mean(axis=0)
    }

    # Mostrar tabla como DataFrame
    resumen_df = pd.DataFrame(tabla_resumen, index=['Valores globales |'])
    resumen_df2 = pd.DataFrame(tabla_resumen2, index=['Tiempo/Iteración |'])
    
    print(resumen_df)
    print("\n")
    print(resumen_df2)
    

    # Gráfico opcional
    plt.figure(figsize=(10, 6))
    plt.bar(df['iteracion'], df['valor_solucion'], color='lightgreen')
    plt.axhline(tabla_resumen['| Promedio'], color='r', linestyle='--', label='Promedio')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la Solución')
    plt.title('Valor de la Solución por Iteración')
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()


