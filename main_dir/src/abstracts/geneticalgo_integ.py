from copy import deepcopy
import os
import random

from .fitness import *
from .grapher import draw
import yaml

def conf_load(filename):
    with open(filename, 'r') as stream:
        try:
            ga_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return ga_params
def conf_save(filename, ga_params):
    with open(filename, 'w+') as out:
        doc = yaml.safe_dump(ga_params, out)

def string_format(bitstring):
    return str(bitstring)[1:-1].translate(dict.fromkeys(map(ord, ', '), None))

def assign_fitness(generated, actual, function):
    result = 0
    if function == 0:
        result = lc_substring(string_format(generated), string_format(actual), len(generated), len(actual))
    elif function == 1:
        result = lc_subsequence(string_format(generated), string_format(actual))
    elif function == 2:
        g_copy = deepcopy(generated)
        a_copy = deepcopy(actual)
        while len(g_copy) < len(a_copy):
            g_copy.insert(0, 0)
        while len(a_copy) < len(g_copy):
            a_copy.insert(0, 0)
        result = hamming_distance(string_format(g_copy), string_format(a_copy))
    # print(result)
    return result

class SNPGeneticAlgo:

    pop = None
    inout_pairs = []  # input-output spike train pairs (Example: {'inputs':('index': 0, 'input':'0101010'),'out':'0101010'})

    def create_population(self, count, initsystem):
        # Flush population and insert the original RSSNP
        self.pop = [{'system': initsystem, 'fitness': 0, 'out_pairs': []}]

        for _ in range(0, count - 1):
            # Create a random network.
            system = deepcopy(initsystem)
            system = system.randomize()

            # Check if generated system is deterministic
            while not system.isValid():
                system = deepcopy(initsystem)
                system = system.randomize()

            system.out_spiketrain = []

            # Add the network to our population.
            self.pop.append({
                'system': system,
                'fitness': 0,
                'out_pairs': []
            })

        return self.pop

    def selection(self, selection_func):
        parents = []
        if selection_func == 0:
            # Get top 50%
            parents = self.pop[:int(len(self.pop)/2)]
        elif selection_func == 1:
            # Get random 25%
            total_fitness = 0
            for chrom in self.pop:
                total_fitness += chrom['fitness']

            if total_fitness != 0:
                i = 0
                while len(parents) != int(len(self.pop)/4):
                    if random.randint(0,total_fitness) <= self.pop[-i]['fitness'] and not (self.pop[-i] in parents):    # chance to become parent is fitness/total fitness
                        parents.insert(0,self.pop[-i])
                    i = (i + 1) % len(self.pop)
            else:
                parents = self.pop[:int(len(self.pop)/4)]
        elif selection_func == 2:
            # Get random 25% and top 25%
            total_fitness = 0

            parents = self.pop[:int(len(self.pop)/4)]
            for chrom in self.pop:
                if chrom not in parents:
                    total_fitness += chrom['fitness']

            if total_fitness != 0:
                i = 0
                while len(parents) != int(len(self.pop)/4):
                    if random.randint(0,total_fitness) <= self.pop[-i]['fitness'] and not (self.pop[-i] in parents):    # chance to become parent is fitness/total fitness
                        parents.insert(0,self.pop[-i])
                    i = (i + 1) % len(self.pop)
            else:
                parents = self.pop[:int(len(self.pop)/2)]

        return parents

    def crossover(self, mutation_rate, selection_func):
        '''
        Performs crossover of 2 parents over the population to create a new population
        '''

        population_size = len(self.pop)
        # Get only parents
        parents = self.selection(selection_func)
        # delete half of the population
        self.pop = self.pop[:(int(len(self.pop)/2))]

        i = 0
        while True:
            cross_counter = 0
            while True:
                # Get parents
                parent1 = deepcopy(parents[i % len(parents)])  # best parent
                parent2 = deepcopy(parents[(i + 1) % len(parents)]) # 2nd best
                
                # Choose random rule to swap
                index1 = random.randint(0, parent1['system'].m - 1)
                index2 = random.randint(0, parent2['system'].m - 1)

                backup1 = deepcopy(parent1)
                backup2 = deepcopy(parent2)
                # Swap rules
                parent1['system'].rule[index1], parent2['system'].rule[index2] = parent2['system'].rule[index2], parent1['system'].rule[index1]
                # Mutate
                parent1['system'].randomize(mutation_rate)
                parent2['system'].randomize(mutation_rate)

                # Check if the children RSSNPs are deterministic
                if parent1['system'].isValid() and parent2['system'].isValid():
                    parent1['system'].out_spiketrain = []
                    parent2['system'].out_spiketrain = []
                    self.pop.extend([parent1,parent2])
                    break
                elif cross_counter < 10:
                    #print("Mutation failed")
                    cross_counter += 1
                    parent1 = deepcopy(backup1)
                    parent2 = deepcopy(backup2)
                elif cross_counter == 10:   # crossover wont work anymore - just copy the parents' original elements
                    print("Stopping crossover -- Copying parents instead")
                    parent1 = deepcopy(backup1)
                    parent2 = deepcopy(backup2)
                    parent1['system'].out_spiketrain = []
                    parent2['system'].out_spiketrain = []
                    self.pop.extend([parent1,parent2])
                    break

            if len(self.pop) == population_size:
                return

            i += 1

    def evaluate(self, chromosome, function):
        '''
        Evaluates the system based on the generated spike trains
        Function legend:
        0: Longest Common Substring
        1: Longest Common Subsequence
        2: Hamming Distance (must be same length)
        '''
        chromosome['out_pairs'] = []
        chromosome['fitness'] = 0

        # Compute for the total output bits
        total_length = 0
        for spike_train in self.inout_pairs:
            total_length += len(spike_train['output'])

        #chromosome['system'] = {}
        for pair in self.inout_pairs:
            maxSteps = 3*len(pair['output'])
            chromosome['system'].in_spiketrain = pair['inputs']
            # print(chromosome['system'].in_spiketrain)
            chromosome['system'].out_spiketrain = []
            config = deepcopy(chromosome['system'].configuration_init)
            
            # simulate the rssnp
            chromosome['out_pairs'].append((chromosome['system'].main((config, chromosome['system'].ruleStatus), maxSteps), pair['output']))
            chromosome['fitness'] += int(assign_fitness(chromosome['system'].out_spiketrain, pair['output'], function)/len(pair['output'])*100)
        chromosome['fitness'] = int(chromosome['fitness']/len(self.inout_pairs))

        # print(chromosome)

    def simulate(self, system, size, function, generations, mutation_rate, path_name, run_index, selection_func):
        '''
            Performs a complete run of the genetic algorithm framework
        '''
        
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        # Generate initial population
        copy_sys = deepcopy(system)
        self.create_population(size, copy_sys)
        filename = path_name
        ga_params = conf_load(filename);
  
        ga_params["population_size"] = str(size)
        ga_params["mutation_rate"] = str(mutation_rate)
        ga_params["fitness_function"] = str(function)
        ga_params["selection_function"] = str(selection_func)


        for generation in range(0, generations):
            current_gen = ga_params["runs"][run_index]["generations"][generation]
            # # Create folder
            # folder = path_name + "/" + "Run" + str(run_index) + "/" + "Generation" + str(generation) + "/"
            # if not os.path.exists(folder):
            #     os.makedirs(folder)           

            print("Evaluating Gen "+str(generation)+"...")
            
            # Calculate fitness of each element
            i = 0
            max_fitness = 0
            chromosome_index = []
            for chrom in self.pop:
                #print("Chromosome:",i)
                i += 1
                self.evaluate(chrom, function)
                result_fitness = chrom['fitness']
                if  result_fitness >= max_fitness:
                    max_fitness = result_fitness 
                    if result_fitness  == max_fitness:
                        chromosome_index.append(i)
                    else:
                        chromosome_index = []
                        chromosome_index.append(i)


            current_gen['best_fitness_result'] = max_fitness
            current_gen['best_chromosome_indexes'] = chromosome_index
            
            # Sort population acc. to fitness level
            self.pop = sorted(self.pop, key=lambda k: k['fitness'], reverse=True)
                
            # Crossover and mutation
            #print("Crossover:",generation)
            self.crossover(mutation_rate, selection_func)
        conf_save(filename, ga_params)

        return current_gen['best_fitness_result']

            