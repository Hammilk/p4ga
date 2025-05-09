#Project 1
#David Pham
import numpy as np
import os
import random

np.random.seed(42)

class GeneticAlgorithm:
    def __init__(self, eval_function, population_size=40, mutation_probability=.05,
                 crossover_probability=.8, population_lower_bound=-5,
                 population_upper_bound=9, iteration_number=50, maximization=True,
                 binary_flag=False, binary_length=10):
        self.population_size=population_size
        self.vector_length = len(eval_function)
        self.mutation_probability = mutation_probability
        self.crossover_probability = mutation_probability
        self.population_lower_bound = population_lower_bound
        self.population_upper_bound = population_upper_bound
        self.iteration_number = iteration_number
        self.maximization = maximization
        self.binary_flag = binary_flag
        self.binary_length = binary_length
        self.eval_function = eval_function
        self.population = self.initialize_population()

    def initialize_population(self):
        def generate_binary_strings(length: int):
            binary_string = ""
            for index in range(length):
                random_bit = str(random.randint(0, 1))
                binary_string += random_bit
            return binary_string

        population = []
        for idx in range(self.population_size):
            pop_vector = []
            for idx2 in range(self.vector_length):
                pop_vector.append(generate_binary_strings(self.binary_length))
            population.append(pop_vector)

        return population

    def selection(self):
        selection_vector = np.zeros(self.population_size)
        next_population = []
        non_zero_constant = .0000000000000000000000000001

        for x in range(self.population_size):
            selection_vector[x] = 1 / (self.evaluate(self.population[x]) + non_zero_constant)

        total_fitness = np.sum(selection_vector)
        for idx in range(np.shape(selection_vector)[0]):
            selection_vector[idx] = (selection_vector[idx])/total_fitness

        temp_sum = 0
        for idx in range(0, np.shape(selection_vector)[0]):
            selection_vector[idx] = selection_vector[idx] + temp_sum
            temp_sum = selection_vector[idx]

        for pop_index in range(np.shape(self.population)[0]):
            random_float = np.random.random()
            for fitness_index in range(np.shape(selection_vector)[0]):
                print(self.population[38])
                if random_float < selection_vector[fitness_index]:
                    next_population.append(self.population[fitness_index])
                    break

            self.population = next_population

    def evaluate(self, vector):
        result = 0
        variable_index = range(len(self.eval_function))

        for term in variable_index:
            result += self.binary_convert_to_float(vector[term]) ** 2

        return result

    def binary_convert_to_float(self, binary_string: str):
        reversed_string: str = binary_string[::-1]
        float_number = 0
        for bit_number in range(len(reversed_string)):
            float_number += (int(reversed_string[bit_number]) * (2 ** bit_number))
        value = self.population_lower_bound + (float_number * ((self.population_upper_bound - self.population_lower_bound)/(2**self.binary_length -1)))
        return value

    def crossover_selection(self):
        def crossover(idx3, paired_index1):
            vector1 = self.population[idx3]
            vector2 = self.population[paired_index1]

            crossover_point = np.random.randint(1, np.shape(vector1)[0])
            parent1 = np.concatenate((vector1[0:crossover_point], vector2[crossover_point:]))
            parent2 = np.concatenate((vector2[0:crossover_point], vector2[crossover_point:]))
            
            self.population[idx3] = parent1
            self.population[paired_index1] = parent2
        crossover_counter = 0
        paired_index = 0
        for idx in range(np.shape(self.population)[0]):
            crossover_chance = np.random.random()
            if crossover_chance < self.crossover_probability:
                crossover_counter += 1
                if crossover_counter % 2 == 0:
                    crossover(idx, paired_index)
                else:
                    paired_index = idx

    def mutation(self):
        def mutate(gene):
            mutated_gene1 = gene
            for gene_index1 in range(self.binary_length):
                mutation_chance1 = np.random.random()
                if mutation_chance1 < self.mutation_probability:
                    if mutated_gene1[gene_index1] == '0':
                        mutated_gene1 = mutated_gene1[0:gene_index1] + '1' + mutated_gene1[gene_index1 + 1:]
                    else:
                        mutated_gene1 = mutated_gene1[0:gene_index1] + '0' + mutated_gene1[gene_index1 + 1:]
            return mutated_gene1

        for pop_index in range(np.shape(self.population)[0]):
            for gene_index in range(np.shape(self.population[pop_index])[0]):
                new_gene = mutate(self.population[pop_index][gene_index])
                self.population[pop_index][gene_index] = new_gene


    def stat(self):
        fitness_sum = 0
        min_fitness = 10000
        min_chromosome = ""
        for chromosome in self.population:
            fitness = self.evaluate(chromosome)
            fitness_sum += fitness
            if fitness <= min_fitness:
                min_fitness = fitness
                min_chromosome = chromosome
        return [min_chromosome, min_fitness]

    def driver(self):
        best_of_run = ['', 10000]
        iterations = 0
        while iterations < self.iteration_number:
            self.selection()
            self.crossover_selection()
            self.mutation()
            run_stat = self.stat()
            iterations += 1
            if run_stat[1] < best_of_run[1]:
                best_of_run = [run_stat[0], run_stat[1]]
        return best_of_run

def main():
    first_term = [1, 2]
    second_term = [1, 2]
    third_term = [1, 2]
    fourth_term = [1, 2]
    eval_function = [first_term, second_term, third_term, fourth_term]

    test_model = GeneticAlgorithm(eval_function, maximization=False, population_size=40,
                                  crossover_probability=.8, mutation_probability=.05, iteration_number=50,
                                  binary_flag=True, binary_length=10, population_lower_bound=-5, population_upper_bound=9)
    test_model.selection()


    """
    import statistics as st
    best_of_runs = []

    min_fitness = 1000
    best_run_stat = []

    for run in range(30):
        model_run = GeneticAlgorithm(eval_function, maximization=False, population_size=40,
                                     crossover_probability=.8, mutation_probability=.05, iteration_number=50,
                                     binary_flag=True, binary_length=10, population_lower_bound=-5, population_upper_bound=9)
        best_vector = model_run.driver()
        best_run_stat.append(best_vector)
        
        if best_vector[1] <= min_fitness:
            best_of_runs = best_vector

    float_representation = []
    convert = GeneticAlgorithm(eval_function, maximization=False, binary_flag=True, binary_length=10)
    for gene in best_of_runs[0]:
        float_representation.append(convert.binary_convert_to_float(gene))

    print(f'Chromosome with best fitness over 30 runs (binary): {best_of_runs[0]}')
    print(f'Chromosome with best fitness over 30 runs (float): {float_representation}')
    print(f'Fitness of best chromosome: {best_run_stat[1]}')
    """

if __name__ == "__main__":
    main()









