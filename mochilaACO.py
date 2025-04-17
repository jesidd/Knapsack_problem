import random
import numpy as np

class KnapsackProblem:
    def __init__(self, weights, values, quantities, capacity):
        """
        Inicializa el problema de la mochila con unidades limitadas
        :param weights: lista de pesos de los objetos
        :param values: lista de valores de los objetos
        :param quantities: lista de cantidades disponibles de cada objeto
        :param capacity: capacidad de la mochila
        """
        self.weights = weights
        self.values = values
        self.quantities = quantities
        self.capacity = capacity
        self.n_items = len(weights)
        # Inicializar el vector de feromonas con valores uniformes
        self.pheromone = [1 / self.n_items for _ in range(self.n_items)]
        # Inicializar la información heurística (valor/peso)
        self.eta = [values[i] / weights[i] if weights[i] > 0 else 0 for i in range(self.n_items)]


class ACOKnapsack:
    def __init__(self, ant_count=30, generations=100, alpha=1.0, beta=2.0, gamma=2.0, 
                 rho=0.5, q=100):
        """
        :param ant_count: número de hormigas
        :param generations: número de generaciones
        :param alpha: importancia de la feromona
        :param beta: importancia de la heurística (relación valor/peso)
        :param gamma: importancia del valor
        :param rho: coeficiente de evaporación de feromona
        :param q: intensidad de la feromona para actualización
        """
        self.ant_count = ant_count
        self.generations = generations
        self.alpha = alpha
        self.beta = beta  # Para la relación valor/peso
        self.gamma = gamma  # Para el valor absoluto
        self.rho = rho
        self.Q = q
        # Métricas de rendimiento
        self.fitness_history = []
        self.convergence_generation = None
        self.execution_time = None
        self.evolution_iter = [[] for _ in range(30)]

    def _update_pheromone(self, problem, ants):
        """
        Actualiza los niveles de feromona en todas las posiciones
        :param problem: instancia de KnapsackProblem
        :param ants: lista de hormigas _AntKnapsack
        """
        # Evaporación de feromona
        for i in range(problem.n_items):
            problem.pheromone[i] *= self.rho
        
        # Añadir feromona según las soluciones
        for ant in ants:
            for i in range(problem.n_items):
                problem.pheromone[i] += ant.pheromone_delta[i]

    def solve(self, problem):
        """
        Resuelve el problema de la mochila usando el algoritmo ACO
        :param problem: instancia de KnapsackProblem
        :return: mejor solución y su valor
        """
        import time
        start_time = time.time()
        
        
        self.fitness_history = []
        last_improvement_gen = 0

        best_solution = None
        best_value = 0
        best_generations = []
        for gen in range(self.generations):
            # Crear población de hormigas
            ants = [_AntKnapsack(self, problem) for _ in range(self.ant_count)]
            
            # Construir solución para cada hormiga
            for ant in ants:
                ant.construct_solution()
                
                # Calcular feromonas a depositar
                ant.update_pheromone_delta()
                
                # Actualizar mejor solución global
                if ant.total_value > best_value and ant.total_weight <= problem.capacity:
                    best_value = ant.total_value
                    best_solution = ant.solution.copy()
                    last_improvement_gen = gen
                    
                
            # Guardar el mejor valor de esta generación (aunque no haya mejorado)     
            best_generations.append(best_value)
            
            # Actualizar feromonas
            self._update_pheromone(problem, ants)
            
            # Guardar mejor valor de la generación
            self.fitness_history.append(best_value)
            
        # Calcular tiempo de ejecución
        self.execution_time = time.time() - start_time
        # Registrar la generación de convergencia
        self.convergence_generation = last_improvement_gen + 1

        return {'items': best_solution, 'value': best_value}, best_generations
    
    
class _AntKnapsack:
    def __init__(self, aco, problem):
        """
        Inicializa una hormiga para resolver el problema de la mochila
        :param aco: instancia de ACOKnapsack
        :param problem: instancia de KnapsackProblem
        """
        self.colony = aco
        self.problem = problem
        self.solution = [0] * problem.n_items  # Cuántas unidades seleccionadas de cada objeto
        self.total_weight = 0.0
        self.total_value = 0.0
        self.pheromone_delta = [0.0] * problem.n_items  # Delta de feromona a añadir por cada objeto
        # Lista de objetos permitidos para selección (inicialmente todos)
        self.allowed = list(range(problem.n_items))
    
    def select_next(self):
        """
        Selecciona el siguiente objeto para añadir a la mochila
        basado en probabilidades calculadas por feromonas y heurística
        :return: índice del objeto seleccionado o None si no hay selección
        """
        if not self.allowed:
            return None
        
        # Calcular denominador (suma total para normalización)
        denominator = 0.0
        for i in self.allowed:
            # Factores para objetos permitidos
            pheromone_factor = self.problem.pheromone[i] ** self.colony.alpha  # Feromona
            heuristic_factor = self.problem.eta[i] ** self.colony.beta         # Valor/Peso
            value_factor = self.problem.values[i] ** self.colony.gamma         # Valor absoluto
            # Penalizar objetos ya muy usados
            #diversity_factor = (1.0 - self.solution[i] / self.problem.quantities[i]) ** 0.5
            #Comenté el "diversity_factor" para hacer pruebas
            
            denominator += pheromone_factor * heuristic_factor * value_factor #* diversity_factor
        
        if denominator == 0:
            return None
        
        # Calcular probabilidades normalizadas
        probabilities = [0.0] * self.problem.n_items
        for i in self.allowed:
            pheromone_factor = self.problem.pheromone[i] ** self.colony.alpha
            heuristic_factor = self.problem.eta[i] ** self.colony.beta
            value_factor = self.problem.values[i] ** self.colony.gamma
            #diversity_factor = (1.0 - self.solution[i] / self.problem.quantities[i]) ** 0.5
            
            probabilities[i] = (pheromone_factor * heuristic_factor * value_factor ) / denominator # "* diversity_factor" despues de value factor
        
        # Selección por ruleta
        selected = None
        rand = random.random()
        for i in range(self.problem.n_items):
            rand -= probabilities[i]
            if rand <= 0:
                selected = i
                break
        
        # Si por precisión numérica no se seleccionó, tomar el último permitido
        if selected is None and self.allowed:
            selected = self.allowed[-1]
            
        return selected

    def construct_solution(self):
        """
        Construye la solución completa añadiendo objetos hasta llenar la mochila
        o hasta que no queden objetos candidatos válidos
        """
        # Mientras queden objetos permitidos
        while self.allowed:
            # Seleccionar siguiente objeto
            item = self.select_next()
            if item is None:
                break
                
            # Intentar añadir el objeto a la solución
            if (self.solution[item] < self.problem.quantities[item] and 
                self.total_weight + self.problem.weights[item] <= self.problem.capacity):
                # Añadir una unidad del objeto
                self.solution[item] += 1
                self.total_weight += self.problem.weights[item]
                self.total_value += self.problem.values[item]
            else:
                # Si no se puede añadir, eliminar de permitidos
                if item in self.allowed:
                    self.allowed.remove(item)
                continue
            
            # Actualizar lista de objetos permitidos
            self.allowed = [
                i for i in range(self.problem.n_items) if (
                    self.solution[i] < self.problem.quantities[i] and
                    self.problem.weights[i] + self.total_weight <= self.problem.capacity
                )
            ]
            
            # Para añadir diversidad, posibilidad de terminar antes
            if random.random() < 0.05 and self.total_value > 0:
                break
            
    def update_pheromone_delta(self):
        """
        Calcula la cantidad de feromona a depositar para cada objeto
        basada en la calidad de la solución
        """
        if self.total_value <= 0:
            return
        
        # Para cada tipo de objeto en la solución
        for i in range(self.problem.n_items):
            if self.solution[i] > 0:
                # La feromona es proporcional al valor aportado por este tipo de objeto
                value_contribution = self.solution[i] * self.problem.values[i]
                self.pheromone_delta[i] = (self.colony.Q * value_contribution / self.total_value) * \
                 (self.solution[i] / self.problem.quantities[i])