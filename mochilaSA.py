import math
import random
import time

class SimulatedAnnealingKnapsack:
    def __init__(self, initial_temp=1000, final_temp=1, alpha=0.95, iterations_per_temp=1000):
        """
        Inicializa el algoritmo de Enfriamiento Simulado
        :param initial_temp: temperatura inicial
        :param final_temp: temperatura final
        :param alpha: factor de enfriamiento
        :param iterations_per_temp: número de iteraciones por temperatura
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp
        self.convergence_generation = None
        self.execution_time = None
        self.fitness_history = []

    def _random_solution(self, problem):
        """
        Genera una solución inicial aleatoria válida
        """
        solution = [0] * problem.n_items
        remaining_capacity = problem.capacity

        indices = list(range(problem.n_items))
        random.shuffle(indices)

        for i in indices:
            max_qty = min(problem.quantities[i], remaining_capacity // problem.weights[i])
            if max_qty > 0:
                qty = random.randint(0, max_qty)
                solution[i] = qty
                remaining_capacity -= qty * problem.weights[i]
        
        return solution

    def _get_value_weight(self, solution, problem):
        total_value = sum(solution[i] * problem.values[i] for i in range(problem.n_items))
        total_weight = sum(solution[i] * problem.weights[i] for i in range(problem.n_items))
        return total_value, total_weight

    def _neighbor(self, solution, problem):
        """
        Genera una solución vecina a partir de una solución actual
        """
        new_solution = solution.copy()
        index = random.randint(0, problem.n_items - 1)

        if random.random() < 0.5 and new_solution[index] > 0:
            new_solution[index] -= 1
        elif new_solution[index] < problem.quantities[index]:
            new_solution[index] += 1
        
        _, weight = self._get_value_weight(new_solution, problem)
        if weight <= problem.capacity:
            return new_solution
        else:
            return solution  # Si no es válida, retornar la original

    def solve(self, problem):
        start_time = time.time()

        current_solution = self._random_solution(problem)
        current_value, _ = self._get_value_weight(current_solution, problem)

        best_solution = current_solution.copy()
        best_value = current_value

        temp = self.initial_temp
        iteration = 0
        last_improvement_gen = 0

        while temp > self.final_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self._neighbor(current_solution, problem)
                neighbor_value, _ = self._get_value_weight(neighbor, problem)
                delta = neighbor_value - current_value

                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_solution = neighbor
                    current_value = neighbor_value

                    if current_value > best_value:
                        best_solution = current_solution.copy()
                        best_value = current_value
                        last_improvement_gen = iteration

            self.fitness_history.append(best_value)
            temp *= self.alpha
            iteration += 1

        self.execution_time = time.time() - start_time
        self.convergence_generation = last_improvement_gen + 1

        return {
            'items': best_solution,
            'value': best_value
        }
