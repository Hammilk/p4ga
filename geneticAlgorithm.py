import random as rd
from typing import Callable, List

import numpy as np

ALPHA = 0.0000000000000001


class GeneticAlgorithm:
    def __init__(
        self,
        eval_function,
        pop_low: float,
        pop_high: float,
        maximization: bool = True,
        penalty_type=0,
    ):
        self.eval_function = eval_function
        self.maximization = maximization
        self.pop_low = pop_low
        self.pop_high = pop_high
        self.fitness_evaluations = 0
        self.penalty_type = penalty_type
        self.generation_num = 1

    def evaluate(self, vector):
        self.fitness_evaluations += 1
        return self.eval_function(vector)

    def initialize_population(
        self,
        population_size: int,
        chromosome_length: int,
    ) -> np.ndarray:
        population = np.random.uniform(
            self.pop_low, self.pop_high, (population_size, chromosome_length)
        )
        return population

    def initialize_population_binary(
        self, population_size: int, chromosome_length: int
    ) -> np.ndarray:
        return np.random.randint(0, 2, size=(population_size, chromosome_length))

    def proportional_selection(self, population: np.ndarray):
        selection_vector = np.zeros(len(population))
        non_zero_constant = ALPHA

        # Calculate fitness
        for x in range(len(population)):
            if self.maximization:
                selection_vector[x] = self.evaluate(population[x])
            else:
                selection_vector[x] = 1 / (
                    self.evaluate(population[x]) + non_zero_constant
                )

        return selection_vector

    def roulette_wheel(self, population: np.ndarray, selection_vector: np.ndarray):
        total_fitness = np.sum(selection_vector)
        selection_vector = np.cumsum(selection_vector / total_fitness)
        next_population = list()
        for _ in range(len(population)):
            random_float: float = np.random.random()
            for i in range(len(selection_vector)):
                if random_float < selection_vector[i]:
                    next_population.append(population[i])
                    break
        return np.array(next_population)

    def crossover_selection(
        self,
        population: np.ndarray,
        crossover_probability,
        crossover_function: Callable[[list[np.ndarray]], list[np.ndarray]],
        num_parents=2,
    ) -> np.ndarray:

        selected_parents = list()
        selected_indices = list()

        for i in range(len(population)):
            if np.random.random() < crossover_probability:
                selected_parents.append(population[i])
                selected_indices.append(i)
                if len(selected_parents) == num_parents:
                    offspring = crossover_function(selected_parents)
                    for j, child in zip(selected_indices, offspring):
                        population[j] = child
                    selected_parents = []
                    selected_indices = []
        return population

    def crossover(self, parents: list):
        mom = parents[0]
        dad = parents[1]
        crossover_point = np.random.randint(1, len(dad))
        daughter = np.concatenate((dad[:crossover_point], mom[crossover_point:]))
        son = np.concatenate((mom[0:crossover_point], dad[crossover_point:]))
        return [daughter, son]

    def uniform_crossover(self, parents: list):
        mom = parents[0]
        dad = parents[1]
        daughter = list()
        son = list()
        for allele1, allele2 in zip(mom, dad):
            if np.random.random() < 0.5:
                daughter.append(allele1)
                son.append(allele2)
            else:
                daughter.append(allele2)
                son.append(allele1)
        return [np.array(daughter), np.array(son)]

    def multi_parent_cycle(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        offspring_list = []

        num_parents = len(parents)
        genome_length = len(parents[0])

        for parent_start_index in range(num_parents):
            offspring = []
            for gene_index in range(genome_length):
                parent_index = (parent_start_index + gene_index) % num_parents
                allele = parents[parent_index][gene_index]
                offspring.append(allele)
            offspring_list.append(np.array(offspring))
        return offspring_list

    def mutation(self, population, probability=0.05):
        population = population.copy()

        def mutate(gene):
            mutation_increment = (
                np.random.random() * 0.10 * (self.pop_high - self.pop_low)
            ) * np.random.choice([-1, 1])
            mutated_gene = gene + mutation_increment
            return np.clip(mutated_gene, self.pop_low, self.pop_high)

        for i in range(len(population)):
            for j in range(len(population[i])):
                if np.random.random() < probability:
                    population[i][j] = mutate(population[i][j])

        return population

    def binary_bitwise_mutation(self, population, probability=0.05):
        population = population.copy()

        for i in range(len(population)):
            for j in range(len(population[i])):
                if np.random.random() < probability:
                    population[i][j] = population[i][j] ^ 1
        self.generation_num += 1
        return population

    def best(self, population):
        fitnesses = list(map(lambda x: self.evaluate(x), population))
        sorted_indices = np.argsort(fitnesses)
        if self.maximization:
            return population[sorted_indices[-1]]
        else:
            return population[sorted_indices[0]]


def knapsack(
    items: List,
    knapsack_capacity: int,
    item_selection: List,
    penalty_type=0,
    generation_num=1,
) -> int:
    """
    Description: Eval Function for 0/1 knapsack problem
    Params:
        items: tuple of (weight, value)
        knapsack_capacity: total weight knapsack can hold
        item_selection: binary vector of selected items
    """
    weight = 0
    value = 0
    for item, select in zip(items, item_selection):
        weight += select * item[0]
        value += select * item[1]
    if weight > knapsack_capacity:
        if penalty_type == 0:
            value -= 1000
        else:
            value -= np.log(generation_num) * (weight - knapsack_capacity)
    return value


def make_knapsack_eval_fn(ga, items, capacity, penalty_type):
    def eval_fn(vector):
        return knapsack(
            items=items,
            knapsack_capacity=capacity,
            item_selection=vector,
            penalty_type=penalty_type,
            generation_num=ga.generation_num,
        )

    return eval_fn


def main():
    import statistics

    crossover_probability = 0.8
    mutation_chance = 0.05
    iterations = 50
    pop_size = 40
    model_runs = 25
    knapsack_capacity = 150

    # Generate item values and weights
    num_items = 20
    item_list = [(rd.randint(1, 20), rd.randint(1, 1000)) for _ in range(num_items)]

    results = []

    for _ in range(model_runs):
        # Step 1: Create GA instance
        ga = GeneticAlgorithm(
            eval_function=None,
            pop_low=0,
            pop_high=1,
            maximization=True,
            penalty_type=0,
        )

        # Step 2: Attach dynamic evaluation function
        ga.eval_function = make_knapsack_eval_fn(
            ga, items=item_list, capacity=knapsack_capacity, penalty_type=0
        )

        # Step 3: Initialize binary population
        population = ga.initialize_population_binary(pop_size, num_items)

        # Step 4: Run GA loop
        for _ in range(iterations):
            selected = ga.roulette_wheel(
                population, ga.proportional_selection(population)
            )
            crossed = ga.crossover_selection(
                selected, crossover_probability, ga.crossover
            )
            population = ga.binary_bitwise_mutation(crossed, mutation_chance)

        # Step 5: Record best result
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))

    # Step 6: Compute summary statistics
    proportional_avg_best_fitness = sum(x[1] for x in results) / len(results)
    proportional_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = max(
        results, key=lambda x: x[1]
    )  # use max because it's a maximization problem

    print(f"One Point Best: {best}")
    print(f"One Point Avg Best Fitness: {proportional_avg_best_fitness}")
    print(f"One Point Avg Num Evals: {proportional_avg_num_evals}")

    # ------------------------.5 Uniform Crossover----------------------------#

    results = []

    for _ in range(model_runs):
        # Step 1: Create GA instance
        ga = GeneticAlgorithm(
            eval_function=None,
            pop_low=0,
            pop_high=1,
            maximization=True,
            penalty_type=0,
        )

        # Step 2: Attach dynamic evaluation function
        ga.eval_function = make_knapsack_eval_fn(
            ga, items=item_list, capacity=knapsack_capacity, penalty_type=0
        )

        # Step 3: Initialize binary population
        population = ga.initialize_population_binary(pop_size, num_items)

        # Step 4: Run GA loop
        for _ in range(iterations):
            selected = ga.roulette_wheel(
                population, ga.proportional_selection(population)
            )
            crossed = ga.crossover_selection(
                selected, crossover_probability, ga.uniform_crossover
            )
            population = ga.binary_bitwise_mutation(crossed, mutation_chance)

        # Step 5: Record best result
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))

    # Step 6: Compute summary statistics
    proportional_avg_best_fitness = sum(x[1] for x in results) / len(results)
    proportional_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = max(
        results, key=lambda x: x[1]
    )  # use max because it's a maximization problem

    print(f"One Point Best: {best}")
    print(f"One Point Avg Best Fitness: {proportional_avg_best_fitness}")
    print(f"One Point Avg Num Evals: {proportional_avg_num_evals}")

    # --------------------------------------Multi Parent------------------------------------#

    results = []

    for _ in range(model_runs):
        # Step 1: Create GA instance
        ga = GeneticAlgorithm(
            eval_function=None,
            pop_low=0,
            pop_high=1,
            maximization=True,
            penalty_type=0,
        )

        # Step 2: Attach dynamic evaluation function
        ga.eval_function = make_knapsack_eval_fn(
            ga, items=item_list, capacity=knapsack_capacity, penalty_type=0
        )

        # Step 3: Initialize binary population
        population = ga.initialize_population_binary(pop_size, num_items)

        # Step 4: Run GA loop
        for _ in range(iterations):
            selected = ga.roulette_wheel(
                population, ga.proportional_selection(population)
            )
            crossed = ga.crossover_selection(
                selected, crossover_probability, ga.crossover
            )
            population = ga.binary_bitwise_mutation(crossed, mutation_chance)

        # Step 5: Record best result
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))

    # Step 6: Compute summary statistics
    proportional_avg_best_fitness = sum(x[1] for x in results) / len(results)
    proportional_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = max(
        results, key=lambda x: x[1]
    )  # use max because it's a maximization problem

    print(f"One Point Best: {best}")
    print(f"One Point Avg Best Fitness: {proportional_avg_best_fitness}")
    print(f"One Point Avg Num Evals: {proportional_avg_num_evals}")

    # -------------------------------------Dynamic Penalty-------------------------------#

    results = []

    for _ in range(model_runs):
        # Step 1: Create GA instance
        ga = GeneticAlgorithm(
            eval_function=None,
            pop_low=0,
            pop_high=1,
            maximization=True,
            penalty_type=1,
        )

        # Step 2: Attach dynamic evaluation function
        ga.eval_function = make_knapsack_eval_fn(
            ga, items=item_list, capacity=knapsack_capacity, penalty_type=1
        )

        # Step 3: Initialize binary population
        population = ga.initialize_population_binary(pop_size, num_items)

        # Step 4: Run GA loop
        for _ in range(iterations):
            selected = ga.roulette_wheel(
                population, ga.proportional_selection(population)
            )
            crossed = ga.crossover_selection(
                selected, crossover_probability, ga.crossover
            )
            population = ga.binary_bitwise_mutation(crossed, mutation_chance)

        # Step 5: Record best result
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))

    # Step 6: Compute summary statistics
    proportional_avg_best_fitness = sum(x[1] for x in results) / len(results)
    proportional_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = max(
        results, key=lambda x: x[1]
    )  # use max because it's a maximization problem

    print(f"One Point Best: {best}")
    print(f"One Point Avg Best Fitness: {proportional_avg_best_fitness}")
    print(f"One Point Avg Num Evals: {proportional_avg_num_evals}")

    # --------------------------------Extreme Nothing Fits---------------------#

    # Still to make, just change the list
    results = []

    for _ in range(model_runs):
        # Step 1: Create GA instance
        ga = GeneticAlgorithm(
            eval_function=None,
            pop_low=0,
            pop_high=1,
            maximization=True,
            penalty_type=0,
        )

        # Step 2: Attach dynamic evaluation function
        ga.eval_function = make_knapsack_eval_fn(
            ga, items=item_list, capacity=knapsack_capacity, penalty_type=0
        )

        # Step 3: Initialize binary population
        population = ga.initialize_population_binary(pop_size, num_items)

        # Step 4: Run GA loop
        for _ in range(iterations):
            selected = ga.roulette_wheel(
                population, ga.proportional_selection(population)
            )
            crossed = ga.crossover_selection(
                selected, crossover_probability, ga.crossover
            )
            population = ga.binary_bitwise_mutation(crossed, mutation_chance)

        # Step 5: Record best result
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))

    # Step 6: Compute summary statistics
    proportional_avg_best_fitness = sum(x[1] for x in results) / len(results)
    proportional_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = max(
        results, key=lambda x: x[1]
    )  # use max because it's a maximization problem

    print(f"One Point Best: {best}")
    print(f"One Point Avg Best Fitness: {proportional_avg_best_fitness}")
    print(f"One Point Avg Num Evals: {proportional_avg_num_evals}")

    # --------------------------------Everything Fits--------------------------#

    results = []

    for _ in range(model_runs):
        # Step 1: Create GA instance
        ga = GeneticAlgorithm(
            eval_function=None,
            pop_low=0,
            pop_high=1,
            maximization=True,
            penalty_type=0,
        )

        # Step 2: Attach dynamic evaluation function
        ga.eval_function = make_knapsack_eval_fn(
            ga, items=item_list, capacity=knapsack_capacity, penalty_type=0
        )

        # Step 3: Initialize binary population
        population = ga.initialize_population_binary(pop_size, num_items)

        # Step 4: Run GA loop
        for _ in range(iterations):
            selected = ga.roulette_wheel(
                population, ga.proportional_selection(population)
            )
            crossed = ga.crossover_selection(
                selected, crossover_probability, ga.crossover
            )
            population = ga.binary_bitwise_mutation(crossed, mutation_chance)

        # Step 5: Record best result
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))

    # Step 6: Compute summary statistics
    proportional_avg_best_fitness = sum(x[1] for x in results) / len(results)
    proportional_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = max(
        results, key=lambda x: x[1]
    )  # use max because it's a maximization problem

    print(f"One Point Best: {best}")
    print(f"One Point Avg Best Fitness: {proportional_avg_best_fitness}")
    print(f"One Point Avg Num Evals: {proportional_avg_num_evals}")


if __name__ == "__main__":
    main()
