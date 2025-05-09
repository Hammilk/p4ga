# Project 1
# David Pham
import os
import random
import sys
from decimal import Decimal, getcontext

import numpy as np

np.random.seed(42)

width = os.get_terminal_size().columns
f = open("output_log", "w")


class GeneticAlgorithm:
    def __init__(
        self,
        eval_function,
        population_size=40,
        mutation_probability=0.05,
        crossover_probability=0.8,
        population_lower_bound=-5,
        population_upper_bound=9,
        mutation_increment=10,
        iteration_number=50,
        maximization=True,
        binary_flag=False,
        binary_length=10,
    ):
        self.population_size = population_size
        self.vector_length = len(eval_function)
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.eval_function = eval_function
        self.population_lower_bound = population_lower_bound
        self.population_upper_bound = population_upper_bound
        self.binary_flag = binary_flag
        self.binary_length = binary_length
        self.mutation_increment = mutation_increment
        self.iteration_number = iteration_number
        self.maximization = maximization
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        This should initialize the first population into random numbers between user-defined upper and lower bound.
        """

        def generate_binary_strings(length: int):
            binary_string = ""
            for index in range(length):
                random_bit = str(random.randint(0, 1))
                binary_string += random_bit
            return binary_string

        if not self.binary_flag:
            population = np.random.uniform(
                self.population_lower_bound,
                self.population_upper_bound,
                (self.population_size, self.vector_length),
            )
        else:
            population = []
            for idx in range(self.population_size):
                pop_vector = []
                for idx2 in range(self.vector_length):
                    pop_vector.append(generate_binary_strings(self.binary_length))
                population.append(pop_vector)

        return population

    def selection(self):
        selection_vector = np.zeros(self.population_size)
        # next_population = np.zeros((self.population_size, self.vector_length))
        next_population = []

        # non_zero_constant needed in case where division by zero occurs
        non_zero_constant = 0.000000000000000000000000000000000000000001

        # Calculate fitness
        for x in np.arange(0, self.population_size):
            if self.maximization:
                selection_vector[x] = self.evaluate(self.population[x])
            else:
                # Invert fitness for minimization problems
                selection_vector[x] = 1 / (
                    self.evaluate(self.population[x]) + non_zero_constant
                )

        # Calculate fitness probabilities
        total_fitness = np.sum(selection_vector)
        for idx in np.arange(0, np.shape(selection_vector)[0]):
            selection_vector[idx] = (selection_vector[idx]) / total_fitness

        # Calculate cumulative probabilities
        temp_sum = 0
        for idx in np.arange(0, np.shape(selection_vector)[0]):
            selection_vector[idx] = selection_vector[idx] + temp_sum
            temp_sum = selection_vector[idx]

        # Roulette wheel selection
        # for pop_index in np.arange(0, np.shape(next_population)[0]):
        for pop_index in range(0, np.shape(self.population)[0]):
            random_float = np.random.random()
            for fitness_index in np.arange(0, np.shape(selection_vector)[0]):
                if random_float < selection_vector[fitness_index]:
                    # next_population[pop_index] = self.population[fitness_index]
                    next_population.append(self.population[fitness_index])
                    break

        self.population = next_population

    def evaluate(self, vector):
        result = 0
        variable_index = np.arange(0, len(self.eval_function))

        for term in variable_index:
            if not self.binary_flag:
                value = vector[term]
            else:
                value = self.binary_convert_to_float(vector[term])

            # result += (value**(self.eval_function[term])[1]) * (self.eval_function[term])[0] #Adds polynomial terms
            result += value**2

        return result

    def binary_convert_to_float(self, binary_string: str):
        reversed_string: str = binary_string[::-1]
        float_number = 0
        for bit_number in range(len(reversed_string)):
            float_number += int(reversed_string[bit_number]) * (2**bit_number)

        value = self.population_lower_bound + (
            float_number
            * (
                (self.population_upper_bound - self.population_lower_bound)
                / (2**self.binary_length - 1)
            )
        )
        return value

    def crossover_selection(self):
        # inner function actually crosses chromosomes
        def crossover(idx3, paired_index1):
            vector1 = self.population[idx3]
            vector2 = self.population[paired_index1]

            # Randomly determine crossover point
            crossover_point = np.random.randint(1, np.shape(vector1)[0])
            parent1 = np.concatenate(
                (vector1[0:crossover_point], vector2[crossover_point:])
            )
            parent2 = np.concatenate(
                (vector2[0:crossover_point], vector1[crossover_point:])
            )

            self.population[idx3] = parent1
            self.population[paired_index1] = parent2

        # Driver
        crossover_counter = 0
        paired_index = 0
        # Determines pairs for crossover
        for idx in np.arange(0, np.shape(self.population)[0]):
            crossover_chance = np.random.random()
            if crossover_chance < self.crossover_probability:
                crossover_counter += 1
                if crossover_counter % 2 == 0:
                    crossover(idx, paired_index)
                else:
                    paired_index = idx

    def mutation(self):
        # Inner function actual mutate
        def mutate(gene):
            mutation_increment = (
                np.random.random()
                * 0.10
                * (self.population_upper_bound - self.population_lower_bound)
            ) * np.random.choice([-1, 1])
            if gene + mutation_increment > self.population_upper_bound:
                return self.population_upper_bound
            elif gene + mutation_increment < self.population_lower_bound:
                return self.population_lower_bound
            else:
                return gene + mutation_increment

        def bitwise_mutate(gene):
            mutated_gene1 = gene
            for gene_index1 in range(0, self.binary_length):
                mutation_chance1 = np.random.random()
                if mutation_chance1 < self.mutation_probability:
                    if mutated_gene1[gene_index1] == "0":
                        mutated_gene1 = (
                            mutated_gene1[0:gene_index1]
                            + "1"
                            + mutated_gene1[gene_index1 + 1 :]
                        )
                    else:
                        mutated_gene1 = (
                            mutated_gene1[0:gene_index1]
                            + "0"
                            + mutated_gene1[gene_index1 + 1 :]
                        )

            return mutated_gene1

        # Driver for mutation
        if self.binary_flag == False:
            for pop_index in np.arange(0, np.shape(self.population)[0]):
                for gene_index in np.arange(0, np.shape(self.population[pop_index])[0]):
                    mutation_chance = np.random.random()
                    if mutation_chance < self.mutation_probability:
                        mutated_gene = mutate(self.population[pop_index][gene_index])
                        self.population[pop_index][gene_index] = mutated_gene
        else:
            for pop_index in range(0, np.shape(self.population)[0]):
                for gene_index in range(0, np.shape(self.population[pop_index])[0]):
                    new_gene = bitwise_mutate(self.population[pop_index][gene_index])
                    self.population[pop_index][gene_index] = new_gene

    def proportional_selection(self):
        print("foo")

    def truncation_selection(self):
        print("foo")

    def deterministic_tournament_selection(self):
        print("foo")

    def linear_ranking_selection(self):
        print("foo")

    def stochastic_binary_tournament_selection(self):
        print("foo")

    def stat(self):
        max_fitness = -9999999.0
        max_chromosome = np.zeros(np.shape(self.population[0]))
        min_chromosome = np.zeros(np.shape(self.population[0]))
        min_fitness = 1000000.0
        fitness_sum = 0

        for chromosome in self.population:

            fitness = self.evaluate(chromosome)
            fitness_sum += fitness
            if fitness >= max_fitness:
                max_fitness = fitness
                max_chromosome = chromosome
            if fitness <= min_fitness:
                min_fitness = fitness
                min_chromosome = chromosome

        average_fitness = fitness_sum / self.population_size
        prediction_results = [
            min_chromosome,
            min_fitness,
            max_chromosome,
            max_fitness,
            average_fitness,
        ]

        return prediction_results

    interval_stats = []

    # Driver
    def predictA(self):
        iterations = 0
        best_of_run_predict = ["", 999999999999999999, 0]
        while iterations < self.iteration_number:
            self.selection()
            self.crossover_selection()
            self.mutation()
            iterations += 1
            run_stat = self.stat()
            if run_stat[1] < best_of_run_predict[1]:
                float_representation = []
                for gene in run_stat[0]:
                    float_representation.append(self.binary_convert_to_float(gene))
                best_of_run_predict = [run_stat[0], run_stat[1], float_representation]

        return best_of_run_predict


def main():

    import argparse

    parser = argparse.ArgumentParser(description="To enter GA parameters")

    parser.add_argument(
        "-p", dest="population_size", type=int, default=40, help="Set population size"
    )
    parser.add_argument(
        "-c",
        dest="crossover_chance",
        type=float,
        default=0.8,
        help="Set crossover chance",
    )
    parser.add_argument(
        "-m",
        dest="mutation_chance",
        type=float,
        default=0.05,
        help="Set mutation chance",
    )
    parser.add_argument(
        "-i", dest="iterations", type=int, default=50, help="Set iteration"
    )

    args = parser.parse_args()

    first_term = [1, 2]  # Encodes to 1x^2
    second_term = [1, 2]
    third_term = [1, 2]
    fourth_term = [1, 2]
    eval_function = [first_term, second_term, third_term, fourth_term]

    # Stats
    import statistics as st

    best_of_run = []  # this is for best vector

    min_fitness = 99999999
    best_run_stat = []  # this is for statistics for vectors

    for run in np.arange(0, 30):
        model_run = GeneticAlgorithm(
            eval_function,
            maximization=False,
            population_size=args.population_size,
            crossover_probability=args.crossover_chance,
            mutation_probability=args.mutation_chance,
            iteration_number=args.iterations,
            binary_flag=True,
            binary_length=10,
            population_lower_bound=-5,
            population_upper_bound=9,
        )
        best_vector = model_run.predictA()

        best_run_stat.append(best_vector[1])  # copies run's best fitness

        if best_vector[1] <= min_fitness:
            best_of_run = best_vector  # overwrites best_of_run with the best vector

    print(f"Chromosome with best fitness over 30 runs (binary): {best_of_run[0]}")
    print(f"Chromosome with best fitness over 30 runs (float): {best_of_run[2]}")
    print(f"Fitness of best chromosome: {best_of_run[1]}")

    print(f"Average of Best-of-Run Fitness is {st.mean(best_run_stat)}")
    print(f"Stdev of Best-of-Run Fitness is {st.stdev(best_run_stat)}")
    f.close()


if __name__ == "__main__":
    main()


"""



#Test and driver class here
#Drives Part A stats for a single run
partA_vector = prediction.predictA()
print(f'Best chromosomes across all runs {partA_vector[0]}')
print(f'Best fitness across all runs {partA_vector[1]}')
#Part B 30 independent run
import statistics as st
best_of_generation_fitness = []
average_of_average_of_generation_fitness = []
best_of_run = []
for run in np.arange(0, 30):
    run_average_best_of_generation = []
    run_average_average_of_generation = []
    model_run = GeneticAlgorithm(eval_function, maximization = False, population_size=40, crossover_probability=.8, mutation_probability=.05, iteration_number=50)
    best_vector = prediction.predictA()
    for element in model_run.interval_stats:
        run_average_best_of_generation.append(element[1])
        run_average_average_of_generation.append(element[4])
    best_of_generation_fitness.append(st.mean(run_average_best_of_generation))
    average_of_average_of_generation_fitness.append(st.mean(run_average_average_of_generation))
    best_of_run.append(best_vector[1])

print(f'Average of Best of Generation Fitness is {st.mean(best_of_generation_fitness)}')
print(f'Stdev of Best of Generation Fitness is {st.stdev(best_of_generation_fitness)}')
print(f'Average of Average of Generation Fitness {st.mean(average_of_average_of_generation_fitness)}')
print(f'Stdev of Average of Generation Fitness {st.stdev(average_of_average_of_generation_fitness)}')
print(f'Average of Best-of-Run Fitness is {st.mean(best_of_run)}')
print(f'Stdev of Best-of-Run Fitness is {st.stdev(best_of_run)}')

"""
