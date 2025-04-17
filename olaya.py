import random
import pandas as pd
import time
import matplotlib.pyplot as plt

# Leer archivo Excel
file_path = 'Mochila_capacidad_maxima_2.45kg.xlsx'
df = pd.read_excel(file_path)

df['Peso_kg'] = df['Peso_kg'].astype(str).str.replace(',', '.').astype(float)

pesos = df['Peso_kg'].tolist()
valores = df['Valor'].tolist()
cantidades = df['Cantidad'].tolist()
objetos = df['Id'].tolist()
capacidad_max = 2.45

# Parámetros ACO
num_hormigas = 30
num_iteraciones = 100
evaporacion = 0.5
intensificacion = 2
inicial_feromonas = 1.0

def construir_solucion(feromonas):
    solucion = [0] * len(pesos)
    peso_total = 0
    while True:
        i = random.randint(0, len(pesos)-1)
        if solucion[i] < cantidades[i]:
            prob = feromonas[i] / sum(feromonas)
            if random.random() < prob:
                if peso_total + pesos[i] <= capacidad_max:
                    solucion[i] += 1
                    peso_total += pesos[i]
                else:
                    break
        if all(solucion[j] >= cantidades[j] for j in range(len(pesos))):
            break
    return solucion

def valor_solucion(sol):
    return sum(sol[i] * valores[i] for i in range(len(sol)))

def peso_solucion(sol):
    return sum(sol[i] * pesos[i] for i in range(len(sol)))

mejores_valores = []
tiempos = []
mejor_sol_global = None
mejor_valor_global = 0
evolucion_iteraciones = [[] for _ in range(num_iteraciones)]

for ejec in range(1, 31):
    feromonas = [inicial_feromonas] * len(pesos)
    mejor_sol = None
    mejor_valor = 0
    valores_por_iteracion = []
    start = time.time()

    for it in range(num_iteraciones):
        soluciones = []
        for a in range(num_hormigas):
            sol = construir_solucion(feromonas)
            soluciones.append(sol)

        for sol in soluciones:
            valor = valor_solucion(sol)
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_sol = sol[:]

        valores_por_iteracion.append(mejor_valor)

        # Evaporar feromonas
        feromonas = [f * (1-evaporacion) for f in feromonas]

        # Reforzar feromonas
        for i in range(len(feromonas)):
            if mejor_sol[i] > 0:
                feromonas[i] += intensificacion * mejor_sol[i]

    end = time.time()
    tiempo = end - start
    tiempos.append(tiempo)
    mejores_valores.append(mejor_valor)

    print(f"Ejecución {ejec}: Mejor valor = {mejor_valor} | Tiempo = {tiempo:.2f} seg")

    if mejor_valor > mejor_valor_global:
        mejor_valor_global = mejor_valor
        mejor_sol_global = mejor_sol[:]

    for i in range(num_iteraciones):
        evolucion_iteraciones[i].append(valores_por_iteracion[i])

# Resultados Finales
print("\nResultados Finales del ACO en 30 ejecuciones:")
print(f"Promedio de valores obtenidos: {sum(mejores_valores)/len(mejores_valores):.2f}")
print(f"Mejor valor global: {mejor_valor_global}")
print(f"Tiempo promedio por ejecución: {sum(tiempos)/len(tiempos):.2f} segundos")
print(f"Tiempo total: {sum(tiempos):.2f} segundos")

print("\nMejor solución encontrada:")
for i in range(len(mejor_sol_global)):
    if mejor_sol_global[i] > 0:
        print(f"- {mejor_sol_global[i]} unidad(es) de '{objetos[i]}' | Peso unitario: {pesos[i]}kg | Valor unitario: {valores[i]}")

print(f"\nPeso total: {peso_solucion(mejor_sol_global):.2f} kg")
print(f"Valor total: {valor_solucion(mejor_sol_global)}")

# Gráfica 1: Valor por ejecución
plt.plot(range(1, 31), mejores_valores, marker='o')
plt.xlabel('Ejecución')
plt.ylabel('Mejor valor encontrado')
plt.title('Resultados ACO - 30 Ejecuciones')
plt.grid()
plt.show()

print("Tamaño",len(evolucion_iteraciones))
# Gráfica 2: Convergencia promedio
promedio_por_iteracion = [sum(valores)/len(valores) for valores in evolucion_iteraciones]
print("promedio ",promedio_por_iteracion)

plt.plot(range(1, num_iteraciones + 1), promedio_por_iteracion, color='green')
plt.xlabel('Iteración')
plt.ylabel('Valor promedio de mejor solución')
plt.title('Convergencia promedio del método ACO')
plt.grid()
plt.show()
