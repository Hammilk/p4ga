#Project 1
#David Pham
import numpy as np
import os

np.random.seed(42)

width = os.get_terminal_size().columns

class GeneticAlgorithm:
    def __init__(self, eval_function, population_size=40, mutation_probability=.05, crossover_probability=.8, population_lower_bound = -5, 
                 population_upper_bound = 9, mutation_increment = 10, iteration_number = 50, maximization = True):
        self.population_size = population_size
        self.vector_length = len(eval_function)
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.eval_function = eval_function
        self.population_lower_bound = population_lower_bound
        self.population_upper_bound = population_upper_bound
        self.population = self.initialize_population()
        self.mutation_increment = mutation_increment
        self.iteration_number = iteration_number
        self.maximization = maximization

    def initialize_population(self):
        """
        This should initialize the first population into random numbers between user-defined upper and lower bound. 
        """
        population = np.random.uniform(self.population_lower_bound, self.population_upper_bound, (self.population_size, self.vector_length))
        return population

    def selection(self):
        selection_vector = np.zeros(self.population_size)
        #non_zero_constant needed in case where division by zero occurs
        non_zero_constant = .000000000000000000000000000000000000000001

        #Calculate fitness
        for x in np.arange(0, self.population_size):
            if self.maximization == True:
                selection_vector[x] = self.evaluate(self.population[x])
            else:
                #Invert fitness for minimization problems
                selection_vector[x] = 1 / (self.evaluate(self.population[x]) + non_zero_constant)

        #Calculate fitness probabilities
        total_fitness = np.sum(selection_vector)
        for idx in np.arange(0, np.shape(selection_vector)[0]):
            selection_vector[idx] = (selection_vector[idx])/total_fitness
        next_population = np.zeros((self.population_size, self.vector_length))

        #Calculate cumulative probabilities
        temp_sum = 0
        for idx in np.arange(0, np.shape(selection_vector)[0]):
            selection_vector[idx] = selection_vector[idx] + temp_sum
            temp_sum = selection_vector[idx]

        #Roulette wheel selection
        for pop_index in np.arange(0, np.shape(next_population)[0]):
            random_float = np.random.random()
            for fitness_index in np.arange(0, np.shape(selection_vector)[0]):
                if random_float < selection_vector[fitness_index]:
                    next_population[pop_index] = self.population[fitness_index]
                    break

        self.population = next_population

    def evaluate(self, vector):
        result = 0
        #variable_index = np.arange(len(self.eval_function)-1) #leaves out last element which is constant
        variable_index = np.arange(0, len(self.eval_function))
        for term in variable_index:
            result += (vector[term]**(self.eval_function[term])[1]) * (self.eval_function[term])[0] #Adds polynomial terms
        return result

    def crossover_selection(self):
        #inner function actually crosses chromosomes
        def crossover(idx, paired_index):
            vector1 = self.population[idx]
            vector2 = self.population[paired_index]
            #Randomly determine crossover point, this may chance
            crossover_point = np.random.randint(1, np.shape(vector1)[0])
            #TODO Check slicing
            parent1 = np.concatenate((vector1[0:crossover_point], vector2[crossover_point:]))
            parent2 = np.concatenate((vector2[0:crossover_point], vector1[crossover_point:]))

            self.population[idx] = parent1
            self.population[paired_index] = parent2
        #Driver
        crossover_counter = 0
        paired_index = 0
        #Determines pairs for crossover
        for idx in np.arange(0, np.shape(self.population)[0]):
            crossover_chance = np.random.random()
            if crossover_chance < self.crossover_probability:
                crossover_counter += 1
                if crossover_counter % 2 == 0:
                    crossover(idx, paired_index)
                else:
                    paired_index = idx

    
    def mutation(self):
        #Inner function actual mutate
        def mutate(gene):
            mutation_increment = (np.random.random() * .10 * (self.population_upper_bound - self.population_lower_bound)) * np.random.choice([-1, 1])
            if(gene + mutation_increment > self.population_upper_bound):
                return self.population_upper_bound
            elif(gene + mutation_increment < self.population_lower_bound):
                return self.population_lower_bound
            else:
                return gene + mutation_increment

        #Driver for mutation  
        for pop_index in np.arange(0, np.shape(self.population)[0]):
            for gene_index in np.arange(0, np.shape(self.population[pop_index])[0]):
                mutation_chance = np.random.random()
                if(mutation_chance < self.mutation_probability):
                    mutated_gene = mutate(self.population[pop_index][gene_index])
                    self.population[pop_index][gene_index] = mutated_gene
    #Function to help with stats reporting 
    def stat(self):
        max_fitness = -999999999999999
        max_chromosome = np.zeros(np.shape(self.population[0]))
        min_chromosome = np.zeros(np.shape(self.population[0]))
        min_fitness = 1000000000000000
        fitness_sum = 0

        for chromosome in self.population:
            fitness = self.evaluate(chromosome)
            fitness_sum += fitness
            if fitness > max_fitness:
                max_fitness = fitness
                max_chromosome = chromosome
            if fitness < min_fitness:
                min_fitness = fitness
                min_chromosome = chromosome
        average_fitness = fitness_sum / self.population_size
        prediction_results = (min_chromosome, min_fitness, max_chromosome, max_fitness, average_fitness)
        return prediction_results

    interval_stats = []

    #Driver
    def predictA(self):
        iterations = 0
        prediction_results = []
        prediction_results.append(self.stat())
        first_run = self.stat()
        """
        print('Interval 0')
        print(f'Chromosome with lowest fitness is {first_run[2]} with fitness of {first_run[3]}')
        print(f'Chromosome with highest fitness is {first_run[0]} with fitness of {first_run[1]}')
        print(f'Average fitness is {first_run[4]}')
        print('-' * width)
        """
        self.interval_stats.append([first_run[0], first_run[1], first_run[2], first_run[3], first_run[4]])


        best_of_run = [first_run[0], first_run[1]]
        while iterations < self.iteration_number:
            self.selection()
            self.crossover_selection()
            self.mutation()
            iterations += 1
            run_stat = self.stat()
            if run_stat[1] < best_of_run[1]:
                best_of_run = [run_stat[0], run_stat[1]]
            if iterations % 10 == 0:
                """
                print(f'Interval {iterations}')
                print(f'Chromosome with lowest fitness is {run_stat[2]} with fitness of {run_stat[3]}')
                print(f'Chromosome with highest fitness is {run_stat[0]} with fitness of {run_stat[1]}')
                print(f'Average fitness is {run_stat[4]}')
                print('-' * width)
                """
                self.interval_stats.append([run_stat[0], run_stat[1], run_stat[2], run_stat[3], run_stat[4]])

            
        return best_of_run


#Can add constant if i felt like it
first_term = [1, 2] #Encodes to 1x^2
second_term = [1, 2]
third_term = [1, 2]
third_term = [1, 2]
fourth_term = [1, 2]
testVector = [1, 2, 3, 4]



#Test and driver class here
eval_function = [first_term, second_term, third_term, fourth_term]
prediction = GeneticAlgorithm(eval_function, maximization = False, population_size=40, crossover_probability=.8, mutation_probability=.05, iteration_number=50)
#Drives Part A stats for a single run
"""
partA_vector = prediction.predictA()
print(f'Best chromosomes across all runs {partA_vector[0]}')
print(f'Best fitness across all runs {partA_vector[1]}')
"""
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










