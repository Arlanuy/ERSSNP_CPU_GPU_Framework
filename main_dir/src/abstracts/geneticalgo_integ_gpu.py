from src.abstracts.rssnp import *
from src.abstracts.gpu_fitness import *

import yaml, numpy

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


class SNPGeneticAlgoGPU:	

	pop = None

	def assign_fitness(self, output_dataset, output_spike_train, function):
	    result = 1
	    print("output_dataset ", output_dataset, " ost ", output_spike_train)
	    if function == 0:
	    	result = GPUlcs(output_dataset, output_spike_train)        
	    	#pass
	    elif function == 1:
	        result = GPULCSubStr(output_dataset, output_spike_train)
	    	#pass
	    elif function == 2:
	    	result = GPUeditDistDP(output_dataset, output_spike_train)
	    	#pass

	    # print(result)
	    return result

	def crossover(self, mutation_rate, selection_func):
		pass

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

	def evaluate(self, chromosome, ga_params, function):
		inout_pairs_view = []
		dataset = self.dataset_arrange(ga_params['test_cases_path'])
		
		for z in range(0, len(dataset)):
			bitstring_length = len(dataset[z])
			single_length = int(bitstring_length / 3)
			i = numpy.arange(0, bitstring_length).reshape(bitstring_length, 1, 1)
			j = numpy.arange(0, bitstring_length, single_length)
			b = numpy.broadcast(i, j)
			inout_pairs_view.append((i + j))
			#print(inout_pairs_view)
			print("datasub input 1 ", dataset[z][inout_pairs_view[z][0][0][0]:inout_pairs_view[z][single_length - 1][0][0]])
			print("datasub input 2 ", dataset[z][inout_pairs_view[z][0][0][1]:inout_pairs_view[z][single_length - 1][0][1]])
			print("minuend ", len(dataset[z]), " subtrahend ", inout_pairs_view[z][0][0][-1], "datasub output", dataset[z][inout_pairs_view[z][0][0][-1]:inout_pairs_view[z][single_length - 1][0][-1]])
			maxSteps = 3*(len(dataset[z]) - inout_pairs_view[z][0][0][-1])  
			print("maxsteps ", maxSteps)
			output_dataset = dataset[z][inout_pairs_view[z][0][0][-1]:inout_pairs_view[z][single_length - 1][0][-1]] 	
			output_dataset = [int(x) for x in list(output_dataset)]
			input_length = (bitstring_length - single_length)/single_length
			for index in range(int(input_length)):
				chromosome['system'].in_spiketrain.append({
                    'index': chromosome['system'].inputs[index],
                    'input': [int(x) for x in list(dataset[z][inout_pairs_view[z][0][0][index]:inout_pairs_view[z][single_length - 1][0][index]])]
                })
				#print("chrom in spike ", chromosome['system'].in_spiketrain)
			chromosome['system'].out_spiketrain = []
			config = deepcopy(chromosome['system'].configuration_init)
			chromosome['out_pairs'].append((chromosome['system'].main((config, chromosome['system'].ruleStatus), maxSteps), output_dataset))
			chromosome['fitness'] += int(self.assign_fitness(output_dataset, chromosome['out_pairs'][z][0], function)/len(chromosome['out_pairs'][z][0])*100)
		chromosome['fitness'] = int(chromosome['fitness']/len(dataset))

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


	def simulate(self, system, size, function, generations, mutation_rate, path_name, run_index, selection_func, start_from_gen):
		filename = path_name
		ga_params = conf_load(filename)
		    
		start = 0

		print("Continuing using the run,generation number ", ga_params['generation_index_continue'])
		run_gen_array = ga_params['generation_index_continue'].split(',')
		run_start = int(run_gen_array[0])
		generation_start = int(run_gen_array[1])
		print(" run-gen ", run_start, generation_start)
		self.use_population(ga_params['runs'][run_start]['population_size'], ga_params['runs'][run_start]['generations'][generation_start]['rssnp_chromosomes'] )
		
		whole_run_best_fitness = 0

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

				function = ga_params['runs'][run_index]['selection_func']
				
				#print("len of inout", self.inout_pairs)
				self.evaluate(chrom, ga_params, function)


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
		print("whole run fitness is " + str(whole_run_best_fitness))
		conf_save(filename, ga_params)
		print("went here")

		


		return whole_run_best_fitness