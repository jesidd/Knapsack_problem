import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from mochilaACO import KnapsackProblem, ACOKnapsack
from plot import plot_progress, display_results, average_convergence, average_time, deseempeño_exp

def leer_datos_excel(archivo):
    df = pd.read_excel(archivo)
    weights = df['Peso_kg'].tolist()
    values = df['Valor'].tolist()
    quantities = df['Cantidad'].tolist()
    capacity = 2.45
    return weights, values, quantities, capacity

def main():
    archivo = 'Mochila_capacidad_maxima_2.45kg.xlsx'
    weights, values, quantities, capacity = leer_datos_excel(archivo)

    iter = 30
    resultados = []
    tiempos = []
    valores_exp=[[0 for _ in range(iter)] for _ in range(2)]

    # Listas separadas para las generaciones de cada experimento
    best_generation_exp1 = []
    best_generation_exp2 = []

    # parámetros
    ant_count=[30,30]
    generations=[150,100]
    alpha=[0.8,0.8]
    beta=[1.5, 1.5]
    gamma=[2.6,2]
    rho=[0.5,1]
    q=[1,1]

    for j in range(2):
        for i in range(iter):
            problem = KnapsackProblem(weights, values, quantities, capacity)
            aco = ACOKnapsack(
                ant_count=ant_count[j],
                generations=generations[j],
                alpha=alpha[j],
                beta=beta[j],
                gamma=gamma[j],
                rho=rho[j],
                q=q[j]
            )

            start_time = time.time()
            solution, generaciones_valores = aco.solve(problem)
            end_time = time.time() - start_time

            valores_exp[j][i] = max(generaciones_valores)
            tiempos.append(end_time)
            converge = detectar_convergencia(generaciones_valores)

            if j == 0:
                best_generation_exp1.append(generaciones_valores)
            else:
                best_generation_exp2.append(generaciones_valores)

            resultados.append({
                'iteracion': i + 1,
                'solution': solution,
                'execution_time': end_time,
                'convergence_generation': converge
            })

            print(f" Iteración {i + 1} del Experimento {j+1}:")
            print(f"  Mejor solución: {solution}")
            print(f"  Tiempo de ejecución: {end_time :.2f} s")
            print(f"  Empieza a converger en la generación: {converge}")
            print("-" * 40)
        
        display_results(resultados, j+1)
        average_time(tiempos, j+1)

    

    # Normalizar para graficar convergencia promedio
    def truncar_generaciones(lista):
        min_len = min(len(gen) for gen in lista)
        return [gen[:min_len] for gen in lista]

    average_convergence(truncar_generaciones(best_generation_exp1), exp=1)
    average_convergence(truncar_generaciones(best_generation_exp2), exp=2)

    deseempeño_exp(valores_exp[0], valores_exp[1])

def detectar_convergencia(best_generations):
    max_val = max(best_generations)
    for i in range(len(best_generations)):
        if all(v == max_val for v in best_generations[i:]):
            return i + 1
    return len(best_generations)

if __name__ == "__main__":
    main()
