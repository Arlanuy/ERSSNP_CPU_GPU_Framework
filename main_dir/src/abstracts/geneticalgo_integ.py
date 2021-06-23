from copy import deepcopy
import os
import random

from .fitness import *
from .grapher import draw
from .rssnp import assign_rssnp
import yaml, time

def timer_write(ga_name, start, finish):
    timer_out_cpu = open(os.getcwd()+ "\\timer_directory\\cputestoutreal.txt", "a+")
    timer_out_cpu.write(ga_name + " CPU time is " + str(finish - start) + "\n")
    timer_out_cpu.close()

def timer_write_run(run_index):
    timer_out_cpu = open(os.getcwd()+ "\\timer_directory\\cputestoutreal.txt", "a+")
    timer_out_cpu.write(" Run index is " + str(run_index) + "\n")
    timer_out_cpu.close()

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

def class_to_yaml(ga_params_rssnp, rssnp):
    ga_params_rssnp['neurons'] = rssnp['system'].n
    ga_params_rssnp['synapses'] = rssnp['system'].l 
    ga_params_rssnp['rules'] = rssnp['system'].rule
    ga_params_rssnp['init_config'] = rssnp['system'].configuration_init
    ga_params_rssnp['rule_status'] = rssnp['system'].ruleStatus
    ga_params_rssnp['input_neurons']  = rssnp['system'].inputs
    ga_params_rssnp['output_neuron'] = rssnp['system'].outputs


def string_format(bitstring):
    return str(bitstring)[1:-1].translate(dict.fromkeys(map(ord, ', '), None))

def assign_fitness(generated, actual, function):
    result = 0
    #print("generated ", string_format(generated), "actual ", string_format(actual))
    if function == 0:
        result = lc_subsequence(string_format(generated), string_format(actual))      
    elif function == 1:
        result = lc_substring(string_format(generated), string_format(actual))
    elif function == 2:
        result = edit_distance2(string_format(generated), string_format(actual))
        #result = edit_distance2(string_format(generated), string_format(actual), len(generated), len(actual))
    #print("result ", result)
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

    def use_population(self, count, last_gen_chromosomes):
        # Flush population and insert the original RSSNP
        #self.pop = [{'system': initsystem, 'fitness': 0, 'out_pairs': []}]
        self.pop = []
        for chromosome in range(0, count):
            # Create a network from previous generation.
            system = assign_rssnp(last_gen_chromosomes[chromosome])
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

            start = time.perf_counter()
            for chrom in self.pop:
                total_fitness += chrom['fitness']
            finish = time.perf_counter()
            timer_write("Selection", start, finish)

            if total_fitness != 0:
                i = 0
                while len(parents) != int(len(self.pop)/4):
                    if random.randint(0,total_fitness) <= self.pop[-i]['fitness'] and not (self.pop[-i] in parents):    # chance to become parent is fitness/total fitness
                        parents.insert(0,self.pop[-i])
                    i = (i + 1) % len(self.pop)
            else:
                parents = self.pop[:int(len(self.pop)/4)]
            #print("chose this selection 1")
        elif selection_func == 2:
            # Get random 25% and top 25%
            total_fitness = 0

            parents = self.pop[:int(len(self.pop)/4)]
            
            start = time.perf_counter()
            for chrom in self.pop:
                if chrom not in parents:
                    total_fitness += chrom['fitness']
            finish = time.perf_counter()
            timer_write("Selection", start, finish)

            if total_fitness != 0:
                i = 0
                while len(parents) != int(len(self.pop)/2):
                    if random.randint(0,total_fitness) <= self.pop[-i]['fitness'] and not (self.pop[-i] in parents):    # chance to become parent is fitness/total fitness
                        parents.insert(0,self.pop[-i])
                    i = (i + 1) % len(self.pop)
            else:
                parents = self.pop[:int(len(self.pop)/2)]
            #print("chose this selection 2")
        print("parents returned by selection are ", len(parents))
        return parents

    def crossover(self, mutation_rate, selection_func):
        '''
        Performs crossover of 2 parents over the population to create a new population
        '''
        
        population_size = len(self.pop)
        # Get only parents
        parents = self.selection(selection_func)
        #print("parents are ", parents)
        # delete half of the population
        self.pop = self.pop[:(int(len(self.pop)/2))]
        #print("parent 2 is ", parents[1])
        i = 0
        while True:
            cross_counter = 0
            #print("passed outer loop here")

            while True:
                # Get parents
                #print("passed inner loop here")
                parent1 = deepcopy(parents[i % len(parents)])  # best parent
                parent2 = deepcopy(parents[(i + 1) % len(parents)]) # 2nd best
                #print("parent 1 is ", parent1)
                #print("parent 2 is ", parent2)
                # Choose random rule to swap
                backup1 = deepcopy(parent1)
                backup2 = deepcopy(parent2)
                start = time.perf_counter()
                index1 = random.randint(0, parent1['system'].m - 1)
                index2 = random.randint(0, parent2['system'].m - 1)
                # Swap rules
                parent1['system'].rule[index1], parent2['system'].rule[index2] = parent2['system'].rule[index2], parent1['system'].rule[index1]
                finish = time.perf_counter()
                timer_write("Crossover", start, finish)
                # Mutate
                parent1['system'].randomize(mutation_rate)
                parent2['system'].randomize(mutation_rate)

                # Check if the children RSSNPs are deterministic
                if parent1['system'].isValid() and parent2['system'].isValid():
                    parent1['system'].out_spiketrain = []
                    parent2['system'].out_spiketrain = []
                    self.pop.extend([parent1,parent2])
                    #print("passed first")
                    if len(self.pop) > population_size:
                        while len(self.pop) > population_size:
                            self.pop = self.pop[:-1]
                        return
                    break
                elif cross_counter < 10:
                    print("Mutation failed")
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

            #print("comparing", len(self.pop), population_size)
            if len(self.pop) == population_size and not i == 0:
                return

            i += 1
        #print("finished crossover here")


    def evaluate(self, chromosome, function):
        '''
        Evaluates the system based on the generated spike trains
        Function legend:
        0: Longest Common Substring
        1: Longest Common Subsequence
        2: Edit distance
        '''
        
        chromosome['out_pairs'] = []
        chromosome['fitness'] = 0

        # Compute for the total output bits
        total_dataset_lengths = 0
        for spike_train in self.inout_pairs:
            total_dataset_lengths += len(spike_train['output'])

        #chromosome['system'] = {}
        total_output_lengths = 0
        for pair in self.inout_pairs:
            maxSteps = 3*len(pair['output'])
            chromosome['system'].in_spiketrain = pair['inputs']
            # print(chromosome['system'].in_spiketrain)
            chromosome['system'].out_spiketrain = []
            config = deepcopy(chromosome['system'].configuration_init)
            
            # simulate the rssnp
            chromosome['out_pairs'].append((chromosome['system'].main((config, chromosome['system'].ruleStatus), maxSteps), pair['output']))
            maxlen = len(chromosome['system'].out_spiketrain)
            total_output_lengths += maxlen
            minlen = min(len(chromosome['system'].out_spiketrain), len(pair['output']))
            #value = += int(assign_fitness(chromosome['system'].out_spiketrain, pair['output'], function))
            if function == 2:
                #print("First string is ", chromosome['system'].out_spiketrain, " Second string is ", pair['output'])
                #print("first is ", len(chromosome['system'].out_spiketrain), " second is ", int(assign_fitness(chromosome['system'].out_spiketrain, pair['output'], function)))
                value = assign_fitness(chromosome['system'].out_spiketrain, pair['output'], function)
                #print("maxlen value is ", maxlen, " while minus value is ", value)
                chromosome['fitness'] +=  int((maxlen - value)/maxlen * len(self.inout_pairs))
            
            else:
                chromosome['fitness'] += assign_fitness(chromosome['system'].out_spiketrain, pair['output'], function)/minlen*100
            #print("add value is ", value)
            #print(" len pair is ", len(pair['output']))
            #chromosome['fitness'] += (value/len(pair['output']))*100
            #print("actual value is ", (value/len(pair['output']))*100)
        print("how many is ", len(self.inout_pairs))
        print("sum in cpu is ", chromosome['fitness'])
        print("total dataset is ", total_dataset_lengths, " and total output is ", total_output_lengths)
        if function == 2:
            delete_point = (1-(total_dataset_lengths/total_output_lengths)) * len(self.inout_pairs)
            maxlen = total_output_lengths/len(self.inout_pairs)
            print("maxlen is ", maxlen, " and delete point is ", delete_point)
            delete_point_total = (maxlen - delete_point)/maxlen * len(self.inout_pairs)
            print("delete point total is ", delete_point_total)
            chromosome['fitness'] = int(chromosome['fitness']/len(self.inout_pairs))# + delete_point_total)
        else:
            chromosome['fitness'] = int(chromosome['fitness']/len(self.inout_pairs))
        
        
        # print(chromosome)


    def simulate(self, system, size, function, generations, mutation_rate, path_name, run_index, selection_func, start_from_gen = False):
        '''
            Performs a complete run of the genetic algorithm framework
        '''

        filename = path_name
        ga_params = conf_load(filename)
        print("run index at ", run_index)
        start = 0
        timer_write_run(run_index)
        # Generate initial population
        if start_from_gen == False:
            copy_sys = deepcopy(system)
            self.create_population(size, copy_sys)
            

        else:
            print("Continuing using the run,generation number ", ga_params['generation_index_continue'])
            run_gen_array = ga_params['generation_index_continue'].split(',')
            run_start = int(run_gen_array[0])
            generation_start = int(run_gen_array[1])
            print(" run-gen ", run_start, generation_start)
            self.use_population(ga_params['runs'][run_start]['population_size'], ga_params['runs'][run_start]['generations'][generation_start]['rssnp_chromosomes'] )

        whole_run_best_fitness = 0
        current_gen = None
        for generation in range(start, start + generations + ga_params['gens_pending']):
            print("gen baby gen " + str(generation))
            print("run index is " + str(run_index) + " gen index is " + str(generation))
            current_gen = ga_params["runs"][run_index]["generations"][generation]
            # # Create folder
            # folder = path_name + "/" + "Run" + str(run_index) + "/" + "Generation" + str(generation) + "/"
            # if not os.path.exists(folder):
            #     os.makedirs(folder)           

            print("Evaluating Gen "+str(generation)+"...")
            
            # Calculate fitness of each element
            chrom_index = 0
            max_fitness = 0
            chromosome_indexes = []
            for chrom in self.pop:
                #print("Chromosome:",i)

                self.evaluate(chrom, function)
                
                result_fitness = chrom['fitness']
                ga_params['runs'][run_index]['generations'][generation]['rssnp_chromosomes'][chrom_index]['chrom_fitness'] = result_fitness
                print("result fitness is " + str(result_fitness))
                if  result_fitness >= max_fitness:   
                    if result_fitness  == max_fitness:
                        chromosome_indexes.append(chrom_index)
                    else:
                        chromosome_indexes = []
                        chromosome_indexes.append(chrom_index)
                    max_fitness = result_fitness 
                #current_gen['rssnp_chromosomes'][i] = chrom['system']
                ga_params_chrom = ga_params['runs'][run_index]['generations'][generation]['rssnp_chromosomes'][chrom_index]
                class_to_yaml(ga_params_chrom, chrom)
                chrom_index += 1



            current_gen['best_fitness_result'] = max_fitness
            if current_gen['best_fitness_result'] > whole_run_best_fitness:
                whole_run_best_fitness = max_fitness

            print("fitness got is " + str(current_gen['best_fitness_result']))
            current_gen['best_chromosome_indexes'] = chromosome_indexes
            print("best chromosome indexes are  " + str(current_gen['best_chromosome_indexes']))
            #print("ga_params at gen " + str(generation) + " is " + str(ga_params))
            conf_save(filename, ga_params)
            # Sort population acc. to fitness level
            self.pop = sorted(self.pop, key=lambda k: k['fitness'], reverse=True)
                
            # Crossover and mutation
            print("Crossover:",generation)
            self.crossover(mutation_rate, selection_func)

        ga_params['runs'][run_index]['max_fitness_in_run'] = whole_run_best_fitness
        print("whole run fitness is ", str(whole_run_best_fitness), " at index ", run_index)
        conf_save(filename, ga_params)
        print("went here")
        
        return whole_run_best_fitness

            