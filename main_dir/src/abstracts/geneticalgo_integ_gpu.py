from src.abstracts.rssnp import *
from src.abstracts.gpu_fitness import *
from src.abstracts.gpu_crossover import *
from src.abstracts.gpu_selection import *
from src.abstracts.gpu_timer import timer_write_run

import yaml, numpy, random, os, time

max_numpy_arraylen = 32



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

#adds padding of N zeros to the list a
def based_init(a,N):
	b = numpy.zeros(a.shape[0] + N)
	b[0:a.shape[0]] = a
	return b

def get_param(array,limit):             #limit - number of params to get array-the list of all possible params
    param = []

    while(len(param)<limit):
        r = []
        j = 0
        temp = random.choice(array)

        for i in array:
            if((i[0] == temp[0] and i[1] == temp[1]) or (i[0] == temp[1] and i[1] == temp[0])):
                r.append(i)

        
        for i in r:
            array.remove(i)

        param.append(temp)
        print(param)


    return param





class SNPGeneticAlgoGPU:	

	pop = None

	def assign_fitness(self, output_dataset, output_spike_train, function, len_dataset = 0, max_row_width = 0, max_col_width = 0, output_dataset_lengths = None, output_rssnp_lengths = None):
	    result = 0

	    
	    if function == 0:
	    	result = GPUlcs(output_dataset, output_spike_train, len_dataset)


	    elif function == 1:
	        result = GPULCSubStr(output_dataset, output_spike_train, len_dataset)

	    elif function == 2:
	    	result = GPUeditDistDP0(output_dataset, output_spike_train, max_row_width, max_col_width, len_dataset, output_dataset_lengths, output_rssnp_lengths)



	    return result



	



	def selection(self, selection_func):
		
		parents = []
		total_fitness_list = []
		if selection_func == 0:
		    # Get top 50%
		    parents = self.pop[:int(len(self.pop)/2)]
		elif selection_func == 1:
		    # Get random 25%
		    total_fitness = 0
		    
		    for chrom in self.pop:

			     total_fitness_list.append(chrom['fitness'])

			
		    total_fitness = init_tf_adder(total_fitness_list, len(total_fitness_list))


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
		    for chrom in self.pop[int(len(self.pop)/4):]:
		    	total_fitness_list.append(chrom['fitness'])


		    total_fitness = init_tf_adder(total_fitness_list, len(total_fitness_list))

		    if total_fitness != 0:

		        i = 0
		        while len(parents) != int(len(self.pop)/2):
		            if random.randint(0,total_fitness) <= self.pop[-i]['fitness'] and not (self.pop[-i] in parents):    # chance to become parent is fitness/total fitness
		                parents.insert(0,self.pop[-i])
		            i = (i + 1) % len(self.pop)
		    else:
		        parents = self.pop[:int(len(self.pop)/2)]


		
		return parents



	def crossover(self, mutation_rate, selection_func):
		
		population_size = len(self.pop)
		# Get only parents
		parents = self.selection(selection_func)
		len_orig_parents = len(self.pop)
		# delete half of the population
		self.pop = self.pop[:(int(len(self.pop)/2))]
		

		i = 0
		while True:
			cross_counter = 0
	
			num_crosses = len(parents) * 4

			random_rule_parents_limit = getrandommax(self.pop, len_orig_parents)
			while True:
				
				parent1 = deepcopy(parents[i % len(parents)])  # best parent
				parent2 = deepcopy(parents[(i + 1) % len(parents)]) # 2nd best
				
				
				backup1 = deepcopy(parent1)
				backup2 = deepcopy(parent2)
				# Swap rules
	
				crossover_indexes = crossover_gpu_defined(len(parents), len_orig_parents, num_crosses, self.pop, random_rule_parents_limit)
				rule1 = int(crossover_indexes[i % len(parents)]) 
				rule2 = int(crossover_indexes[(i + 1) % len(parents)])
				rule1 %= len(parent1['system'].rule)
				rule2 %= len(parent2['system'].rule)
				parent1['system'].rule[rule1], parent2['system'].rule[rule2] = parent2['system'].rule[rule2], parent1['system'].rule[rule1]

				# Mutate
				parent1['system'].randomize(mutation_rate)
				parent2['system'].randomize(mutation_rate)
				
				if parent1['system'].isValid() and parent2['system'].isValid():
					parent1['system'].out_spiketrain = []
					parent2['system'].out_spiketrain = []
					self.pop.extend([parent1,parent2])
					
					if len(self.pop) > population_size:
						while len(self.pop) > population_size:
							self.pop = self.pop[:-1]
					random_rule_parents_limit = getrandommax(self.pop, len_orig_parents)
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
				    random_rule_parents_limit = getrandommax(self.pop, len_orig_parents)
				    break



			if len(self.pop) == population_size and not i == 0:
				return

			i += 1
		

		
	#arranges the dataset for the GPU implementation of the fitness functions
	def dataset_arrange(self, test_case_name):
	    input = open(test_case_name, 'r')
	    dataset = []
	    man = True
	    for line in input:
	        dataset_line = []
	        numbers = line.strip('\n').split(',')
	        for number in numbers:
	            for character in number:
	                dataset_line.append(character)
	        dataset.append(dataset_line)

	    return dataset

	def dataset_arrange2(self, dataset_size, filename, max_spike_size = 10):
		dataset = numpy.zeros(shape=(dataset_size, max_spike_size), dtype=numpy.int32)
		line_index = 0
		max_row_width = 0
		max_col_width = 0
		input = open(filename, 'r')
		for line in input:
		    numbers = line.strip('\n').split(',')

		    spike_index = 0
		    for number in numbers[-1]:
		        dataset[line_index][spike_index] = int(number)
		        spike_index += 1
		    if max_row_width < spike_index:
		        max_row_width = spike_index
		    if max_col_width < spike_index:
		        max_col_width = spike_index
		    line_index += 1
		return numpy.array(dataset), max_row_width, max_col_width

	def dataset_arrange3(self, dataset_size, filename, max_spike_size = 10):
		
		dataset = numpy.zeros(shape=(int(dataset_size/max_numpy_arraylen) + 1, max_numpy_arraylen, max_spike_size), dtype=numpy.int32)
		line_index = 0
		max_row_width = 0
		input = open(filename, 'r')
		for line in input:
		    numbers = line.strip('\n').split(',')

		    spike_index = 0
		    for number in numbers[-1]:
		        dataset[int(line_index/max_numpy_arraylen)][int(line_index%max_numpy_arraylen)][spike_index] = int(number)
		        spike_index += 1
		    if max_row_width < spike_index:
		        max_row_width = spike_index
		    line_index += 1
		return numpy.array(dataset), max_row_width



	def evaluate(self, chromosome, ga_params, fitness_func, selection_func):
		
		inout_pairs_view = []
		row_width = 0
		dataset = self.dataset_arrange(ga_params['test_cases_path'])
		chromosome['fitness'] = 0
		if fitness_func == 2:
			test_case_file = open(ga_params['test_cases_path'], 'r')
			len_dataset = len(list(test_case_file))
			dataset2, row_width = self.dataset_arrange3(len_dataset, ga_params['test_cases_path'])
			output_dataset_lengths = numpy.zeros(len_dataset, dtype = numpy.int32)
			total_dataset_lengths = 0
			for z in range(0, len_dataset):
				bitstring_length = len(dataset[z])
				single_length = int(bitstring_length / 3)
				i = numpy.arange(0, bitstring_length).reshape(bitstring_length, 1, 1)
				j = numpy.arange(0, bitstring_length, single_length)
				b = numpy.broadcast(i, j)
				inout_pairs_view.append((i + j))
				
				output_dataset_lengths[z] = len(dataset[z]) - inout_pairs_view[z][0][0][2]
				total_dataset_lengths += output_dataset_lengths[z]

				maxSteps = 3 * output_dataset_lengths[z]  

				output_dataset = dataset[z][inout_pairs_view[z][0][0][2]:inout_pairs_view[z][single_length - 1][0][2]] 	
				output_dataset = [int(x) for x in list(output_dataset)]
				input_length = (bitstring_length - single_length)/single_length
				chromosome['system'].in_spiketrain = []
				for index in range(len(chromosome['system'].inputs)):
					
					chromosome['system'].in_spiketrain.append({
	                    'index': chromosome['system'].inputs[index],
	                    'input': [int(x) for x in list(dataset[z][inout_pairs_view[z][0][0][index]:inout_pairs_view[z][single_length - 1][0][index]])]
	                })

				chromosome['system'].out_spiketrain = []
				config = deepcopy(chromosome['system'].configuration_init)

				chromosome['out_pairs'].append((chromosome['system'].main((config, chromosome['system'].ruleStatus), maxSteps), output_dataset))
			
			line_index = 0
			output_rssnp_lengths = numpy.zeros(len(list(chromosome['out_pairs'])), dtype = numpy.int32)
			max_spike_size = 20
			output_rssnp_numpy = numpy.zeros(shape=(int(len_dataset/max_numpy_arraylen) + 1, max_numpy_arraylen, max_spike_size), dtype=numpy.int32)
			total_output_lengths = 0
			n = None

			len_output = len(list(chromosome['out_pairs']))
			col_width = 0
			for m in list(chromosome['out_pairs']):
			
				n = numpy.asarray(m[0], dtype=numpy.int32)

				output_rssnp_lengths[line_index] = len(n)
				total_output_lengths += output_rssnp_lengths[line_index]

				n = based_init(n, abs(max_spike_size - len(m[0])))

				
				spike_for_out = 0
				for index_value in range(max_spike_size):
					output_rssnp_numpy[int(line_index/max_numpy_arraylen)][int(line_index%max_numpy_arraylen)][index_value] = n[index_value]
					spike_for_out += 1
				if spike_for_out > col_width:
					col_width = spike_for_out

				line_index += 1


			chromosome['fitness'] = int(self.assign_fitness(dataset2, output_rssnp_numpy, fitness_func, len_dataset, row_width, col_width, output_dataset_lengths, output_rssnp_lengths)) #+ delete_point_total
			
		else:

			len_dataset = len(dataset)	
			for z in range(0, len_dataset):
				bitstring_length = len(dataset[z])
				single_length = int(bitstring_length / 3)
				i = numpy.arange(0, bitstring_length).reshape(bitstring_length, 1, 1)
				j = numpy.arange(0, bitstring_length, single_length)
				b = numpy.broadcast(i, j) 
				inout_pairs_view.append((i + j))
				
				dataset_len = len(dataset[z]) - inout_pairs_view[z][0][0][2]
				maxSteps = 3*dataset_len  
				
				output_dataset = dataset[z][inout_pairs_view[z][0][0][2]:inout_pairs_view[z][single_length - 1][0][2]] 	
				output_dataset = [int(x) for x in list(output_dataset)]

				input_length = (bitstring_length - single_length)/single_length
				chromosome['system'].in_spiketrain = []
				for index in range(len(chromosome['system'].inputs)):
					chromosome['system'].in_spiketrain.append({
	                    'index': chromosome['system'].inputs[index],
	                    'input': [int(x) for x in list(dataset[z][inout_pairs_view[z][0][0][index]:inout_pairs_view[z][single_length - 1][0][index]])]
	                })

				chromosome['system'].out_spiketrain = []
				config = deepcopy(chromosome['system'].configuration_init)

				chromosome['out_pairs'].append((chromosome['system'].main((config, chromosome['system'].ruleStatus), maxSteps), output_dataset))
				
				
				minlen = min(len(chromosome['out_pairs'][z][1]), len(chromosome['out_pairs'][z][0]))

				
				chromosome['fitness'] += int(self.assign_fitness(chromosome['out_pairs'][z][1], chromosome['out_pairs'][z][0], fitness_func, len_dataset)/minlen*100)

			chromosome['fitness'] = int(chromosome['fitness']/len_dataset)
			

	def use_population(self, count, last_gen_chromosomes):
        # Flush population and insert the original RSSNP

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


	def simulate(self, system, size, function, generations, mutation_rate, path_name, run_index, selection_func, start_from_gen):
		filename = path_name
		ga_params = conf_load(filename)
		    
		start = 0
		timer_write_run(run_index)
		print("Continuing using the run,generation number ", ga_params['generation_index_continue'])
		run_gen_array = ga_params['generation_index_continue'].split(',')
		run_start = int(run_gen_array[0])
		generation_start = int(run_gen_array[1])
		self.use_population(ga_params['runs'][run_start]['population_size'], ga_params['runs'][run_start]['generations'][generation_start]['rssnp_chromosomes'] )
		
		# to record only the best of the added runs use this, instead of below
		#whole_run_best_fitness = 0
		whole_run_best_fitness = ga_params['max_fitness_in_runs']

		for generation in range(start, start + generations + ga_params['gens_pending']):
			print("run index is " + str(run_index) + " gen index is " + str(generation))
			current_gen = ga_params["runs"][run_index]["generations"][generation]
     

			print("Evaluating Gen "+str(generation)+"...")

			# Calculate fitness of each element
			chrom_index = 0
			max_fitness = 0
			chromosome_indexes = []
			for chrom in self.pop:


				fitness_func = ga_params['runs'][run_index]['fitness_function']
				selection_func = ga_params['runs'][run_index]['selection_func']

				chrom['out_pairs'] = []
				self.evaluate(chrom, ga_params, fitness_func, selection_func)

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
	
				ga_params_chrom = ga_params['runs'][run_index]['generations'][generation]['rssnp_chromosomes'][chrom_index]
				class_to_yaml(ga_params_chrom, chrom)
				chrom_index += 1

			current_gen['best_fitness_result'] = max_fitness
			if current_gen['best_fitness_result'] > whole_run_best_fitness:
			    whole_run_best_fitness = max_fitness

			print("best fitness got among generations is " + str(current_gen['best_fitness_result']))
			current_gen['best_chromosome_indexes'] = chromosome_indexes
			print("best chromosome indexes are  " + str(current_gen['best_chromosome_indexes']))

			conf_save(filename, ga_params)
			# Sort population acc. to fitness level
			self.pop = sorted(self.pop, key=lambda k: k['fitness'], reverse=True)
			    
			# Crossover and mutation
			print("Crossover:",generation)
			self.crossover(mutation_rate, selection_func)

		ga_params['runs'][run_index]['max_fitness_in_run'] = whole_run_best_fitness
		print("best fitness got among runs " + str(whole_run_best_fitness))
		conf_save(filename, ga_params)
		return whole_run_best_fitness