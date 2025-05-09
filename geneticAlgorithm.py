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
        penalty: Callable = lambda x: x,
    ):
        self.eval_function = eval_function
        self.maximization = maximization
        self.pop_low = pop_low
        self.pop_high = pop_high
        self.fitness_evaluations = 0
        self.penalty = penalty
        # add penalty as an init parameter

    def evaluate(self, vector):
        self.fitness_evaluations += 1
        result = self.eval_function(vector)
        # add self.penalty
        # add binary_convert_to_float
        return result

    def binary_convert_to_float(self, binary_string: str):
        reversed_string: str = binary_string[::-1]
        float_number = 0
        for bit_number in range(len(reversed_string)):
            float_number += reversed_string[bit_number] * (2**bit_number)

        value = self.pop_low + (
            float_number
            * ((self.pop_high - self.pop_low) / (2 ** len(binary_string) - 1))
        )
        return value

    def dynamic_penalty(self):
        print("foo")

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
        self, population_size: int, chromosome_length: int, precision_length: int
    ) -> np.ndarray:

        def generate_binary_strings(length: int):
            binary_string = ""
            for _ in range(length):
                random_bit = str(rd.randint(0, 1))
                binary_string += random_bit
            return binary_string

        population = []
        for _ in range(population_size):
            pop_vector = []
            for _ in range(chromosome_length):
                pop_vector.append(generate_binary_strings(precision_length))
            population.append(pop_vector)
        return np.array(population)

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

    def truncation_selection(
        self, population: np.ndarray, truncation: float
    ) -> np.ndarray:
        truncation_index: int = int(len(population) * truncation)
        fitnesses = list(map(lambda x: self.evaluate(x), population))
        sorted_indices = (
            np.argsort(fitnesses)[::-1] if self.maximization else np.argsort(fitnesses)
        )
        sorted_population = population[sorted_indices]
        sampled_population = sorted_population[:truncation_index]
        truncated_population = list(sampled_population)
        for _ in range(0, len(population) - len(truncated_population)):
            random_index = int(len(sampled_population) * np.random.random())
            truncated_population.append(sampled_population[random_index])
        return np.array(truncated_population)

    def deterministic_tournament_selection(self, population: np.ndarray) -> np.ndarray:
        winners = list()
        for _ in range(0, len(population)):
            fighter1 = population[np.random.randint(0, len(population))]
            fighter2 = population[np.random.randint(0, len(population))]
            score1 = self.evaluate(fighter1)
            score2 = self.evaluate(fighter2)
            if self.maximization == False:
                score1 = (1 / score1) + ALPHA
                score2 = (1 / score2) + ALPHA
            if score1 > score2:
                winners.append(fighter1)
            else:
                winners.append(fighter2)
        return np.array(winners)

    def stochastic_binary_tournament_selection(
        self, population: np.ndarray, probability: float = 0.5
    ) -> np.ndarray:
        winners = list()
        for _ in range(0, len(population)):
            fighter1 = population[np.random.randint(0, len(population))]
            fighter2 = population[np.random.randint(0, len(population))]
            score1 = self.evaluate(fighter1)
            score2 = self.evaluate(fighter2)
            if self.maximization == False:
                score1 = (1 / score1) + ALPHA
                score2 = (1 / score2) + ALPHA

            if score1 > score2:
                stronger, weaker = fighter1, fighter2
            else:
                stronger, weaker = fighter2, fighter1

            tourney_float = np.random.random()
            if tourney_float < probability:
                winners.append(stronger)
            else:
                winners.append(weaker)
        return np.array(winners)

    def linear_ranking_selection(self, population: np.ndarray, max=1.5):
        if len(population) <= 1:
            return np.ones(len(population))

        fitnesses = [self.evaluate(x) for x in population]
        sorted_indices = (
            np.argsort(fitnesses) if self.maximization else np.argsort(fitnesses)[::-1]
        )

        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(population) + 1)

        min_val = 2 - max
        probabilities = [
            (min_val + ((r - 1) / (len(population) - 1)) * (max - min_val))
            / len(population)
            for r in ranks
        ]

        return np.array(probabilities)

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

    def best(self, population):
        fitnesses = list(map(lambda x: self.evaluate(x), population))
        sorted_indices = np.argsort(fitnesses)
        sorted_population = population[sorted_indices]
        return sorted_population[0]


def knapsack(self, items: List, knapsack_capacity: int) -> int:
    print("foo")


def shekel_foxholes(x: np.ndarray) -> float:
    a = np.array(
        [
            [
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
            ],
            [
                -32,
                -32,
                -32,
                -32,
                -32,
                -16,
                -16,
                -16,
                -16,
                -16,
                0,
                0,
                0,
                0,
                0,
                16,
                16,
                16,
                16,
                16,
                32,
                32,
                32,
                32,
                32,
            ],
        ]
    )
    sum = 0
    for i in range(25):
        first_diff = x[0] - a[0, i]
        second_diff = x[1] - a[1, i]
        sum += 1 / (i + 1 + ((first_diff**6) + (second_diff**6)))
    return 1 / (0.002 + sum)


def main():
    import statistics

    min_bound = -65.536
    max_bound = 65.536
    crossover_probability = 0.8
    mutation_chance = 0.05
    iterations = 50
    pop_size = 40
    model_runs = 25

    # Proportional Selection
    results = []
    for _ in range(model_runs):
        ga = GeneticAlgorithm(shekel_foxholes, min_bound, max_bound, False)
        population = ga.initialize_population(pop_size, 2)
        for _ in range(iterations):
            population = ga.mutation(
                ga.crossover_selection(
                    ga.roulette_wheel(
                        population, ga.proportional_selection(population)
                    ),
                    crossover_probability,
                ),
                mutation_chance,
            )
        # Determines best fitness of model run
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))

    # Determines average of model runs
    proportional_avg_best_fitness = sum(x[1] for x in results) / len(results)
    proportional_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = min(results, key=lambda x: x[1])
    print(f"Proportional Best: {best}")
    print(f"Proportional Avg Best Fitness: {proportional_avg_best_fitness}")
    print(f"Proportional Avg Num Evals: {proportional_avg_num_evals}")

    # Truncation Selection
    truncation_cutoffs = [0.25, 0.5, 0.75]
    for cutoff in truncation_cutoffs:
        for _ in range(model_runs):
            ga = GeneticAlgorithm(shekel_foxholes, min_bound, max_bound, False)
            population = ga.initialize_population(pop_size, 2)
            results = []
            for _ in range(iterations):
                population = ga.mutation(
                    ga.crossover_selection(
                        ga.truncation_selection(population, cutoff),
                        crossover_probability,
                    ),
                    mutation_chance,
                )
            best_pop = ga.best(population)
            best_fitness = ga.evaluate(best_pop)
            results.append((best_pop, best_fitness, ga.fitness_evaluations))
        trunc_avg_best_fitness = sum(x[1] for x in results) / len(results)
        trunc_avg_num_evals = sum(x[2] for x in results) / len(results)
        best = min(results, key=lambda x: x[1])
        print(f"Truncation {cutoff} Best: {best}")
        print(f"Truncation {cutoff} Avg Best Fitness: {trunc_avg_best_fitness}")
        print(f"Truncation {cutoff} Avg Num Evals: {trunc_avg_num_evals}")

    # Deterministic Tournament Selection
    for _ in range(model_runs):
        ga = GeneticAlgorithm(shekel_foxholes, min_bound, max_bound, False)
        population = ga.initialize_population(pop_size, 2)
        results = []
        for _ in range(iterations):
            population = ga.mutation(
                ga.crossover_selection(
                    ga.deterministic_tournament_selection(population),
                    crossover_probability,
                ),
                mutation_chance,
            )
        best_pop = ga.best(population)
        best_fitness = ga.evaluate(best_pop)
        results.append((best_pop, best_fitness, ga.fitness_evaluations))
    dt_avg_best_fitness = sum(x[1] for x in results) / len(results)
    dt_avg_num_evals = sum(x[2] for x in results) / len(results)
    best = min(results, key=lambda x: x[1])
    print(f"Determinstic Tournament Best: {best}")
    print(f"Deterministic Tournament Avg Best Fitness: {dt_avg_best_fitness}")
    print(f"Deterministic Tournament Avg Num Evals: {dt_avg_num_evals}")

    # Stochastic Binary Tournament Selection
    tournament_probabilities = [0.9, 0.8, 0.7]
    for prob in tournament_probabilities:
        for _ in range(model_runs):
            ga = GeneticAlgorithm(shekel_foxholes, min_bound, max_bound, False)
            population = ga.initialize_population(pop_size, 2)
            results = []
            for _ in range(iterations):
                population = ga.mutation(
                    ga.crossover_selection(
                        ga.stochastic_binary_tournament_selection(population, prob),
                        crossover_probability,
                    ),
                    mutation_chance,
                )
            best_pop = ga.best(population)
            best_fitness = ga.evaluate(best_pop)
            results.append((best_pop, best_fitness, ga.fitness_evaluations))
        sbt_avg_best_fitness = sum(x[1] for x in results) / len(results)
        sbt_avg_num_evals = sum(x[2] for x in results) / len(results)
        best = min(results, key=lambda x: x[1])
        print(f"Stochastic Tournament {prob} Best: {best}")
        print(f"Stochastic Tournament {prob} Avg Best Fitness: {sbt_avg_best_fitness}")
        print(f"Stochastic Tournament {prob} Avg Num Evals: {sbt_avg_num_evals}")

    # Linear Ranking Selection
    max_copies = [2.0, 1.7, 1.4]
    for val in max_copies:
        for _ in range(model_runs):
            ga = GeneticAlgorithm(shekel_foxholes, min_bound, max_bound, False)
            population = ga.initialize_population(pop_size, 2)
            results = []
            for _ in range(iterations):
                population = ga.mutation(
                    ga.crossover_selection(
                        ga.roulette_wheel(
                            population, ga.linear_ranking_selection(population, val)
                        ),
                        crossover_probability,
                    ),
                    mutation_chance,
                )
            best_pop = ga.best(population)
            best_fitness = ga.evaluate(best_pop)
            results.append((best_pop, best_fitness, ga.fitness_evaluations))
        rank_avg_best_fitness = sum(x[1] for x in results) / len(results)
        rank_avg_num_evals = sum(x[2] for x in results) / len(results)
        best = min(results, key=lambda x: x[1])
        print(f"Linear Ranking {val} Best: {best}")
        print(f"Linear Ranking {val} Avg Best Fitness: {rank_avg_best_fitness}")
        print(f"Linear Ranking {val} Avg Num Evals: {rank_avg_num_evals}")


if __name__ == "__main__":
    main()
