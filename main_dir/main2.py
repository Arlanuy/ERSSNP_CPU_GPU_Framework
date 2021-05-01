#from src.experimenter_uncommented import gaframework, gaframework_gpu
import itertools
import numpy as np
from copy import deepcopy
from greenery.lego import parse
from random import randint
import yaml, os, numpy
import numpy as np
max_numpy_arraylen = 32
import pycuda.autoinit
import pycuda.driver as drv
import math
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray as pg
from pycuda import gpuarray

def based_init(a,N):
	b = numpy.zeros((a.shape[0] + N))
	b[0:a.shape[0]] = a
	return b

def class_to_yaml(ga_params_rssnp, rssnp):
    ga_params_rssnp['neurons'] = rssnp['system'].n
    ga_params_rssnp['synapses'] = rssnp['system'].l 
    ga_params_rssnp['rules'] = rssnp['system'].rule
    ga_params_rssnp['init_config'] = rssnp['system'].configuration_init
    ga_params_rssnp['rule_status'] = rssnp['system'].ruleStatus
    ga_params_rssnp['input_neurons']  = rssnp['system'].inputs
    ga_params_rssnp['output_neuron'] = rssnp['system'].outputs

def gaframework(rssnp_string, path_to_io_spike_trains, loadfile_name, start_new = True, start_from_gen = False):
    ga = SNPGeneticAlgo()
    gaeval = SNPGeneticAlgoEval()
    ga_params = conf_load(loadfile_name)

    rssnp = assign_rssnp(rssnp_string)
    ga.inout_pairs = spike_train_parser(path_to_io_spike_trains,rssnp.inputs)

    execute_experiment(rssnp, ga, gaeval, ga_params, loadfile_name, start_new, start_from_gen)

def gaframework_gpu(loadfile_name):
    ga = SNPGeneticAlgoGPU()
    gaeval = SNPGeneticAlgoEval ()
    print("entering to 1", loadfile_name)
    ga_params = conf_load(loadfile_name)
    #ga.inout_pairs = spike_train_parser(path_to_io_spike_trains,rssnp.inputs)
    execute_experiment_gpu(ga, gaeval, ga_params, loadfile_name)

def set_bounds(ga_eval, runs):
    # Ask user for the upper bounds of desired rssnp
    print("Max number of...")
    neurons = int(input("neurons: "))
    spikes = int(input("initial spikes: "))
    synapses = int(input("synapses: "))
    rules = int(input("rules: "))
    i = int(input("i (in regex): "))
    j = int(input("j (in regex): "))
    consumed = int(input("consumable spikes: "))
    produced = int(input("producable spikes: "))

    input_neurons = input("\nInput neurons (comma-separated): ")
    input_neurons = [int(x) for x in input_neurons.split(",")]

    output_neuron = int(input("\nOutput neuron: "))

    fixed = input("Fixed? (Y or N): ")
    fixed = True if fixed == 'Y' else False

    
    print("Creating RSSNP...")
 

    exit_flag = False
    while not exit_flag:
        # Make that rssnp based on the user's specifications
        rssnp_dict = create_rssnp_dict(neurons, spikes, synapses, rules, i, j, consumed, produced, input_neurons, output_neuron, fixed)
        rssnp = assign_rssnp(rssnp_dict)
        exit_flag = rssnp.isValid()

    return rssnp_dict

def set_values(ga_eval, runs):
    print("Exact number of...")
    neurons = int(input("neurons: "))
    spikes = int(input("initial spikes: "))
    synapses = int(input("synapses: "))
    rules = int(input("rules: "))
    init_config = int(input("shall be the constant init_config: "))
    init_config_list = [init_config for i in range(neurons)]
    rule_status = int(input("shall be the constant rule_status: "))
    rule_status_list = [rule_status for i in range(rules)]
    print("Write the rules (comma-separated) in the order of source neuron, sink neuron, grammar multiplicity(a+, a*), consumed spikes, produced spikes, delay") 
    rule_in_csv = []
    rule_mat = []
    rule_content = 7 
    for rule in range(rules):
        rule_vec = []
        rule_in_csv = input("Enter a rule: ")
        grammar = []
        for index in range(0, rule_content):
            element = int(rule_in_csv[2 * index])
            if index == 2:
                grammar.append(element)

            elif index == 3:
                grammar.append(element)
                rule_vec.append(grammar)
            else:
                rule_vec.append(element)
        rule_mat.append(rule_vec)
    #print(rule_mat)

    input_neurons = input("\nInput neurons (comma-separated): ")
    input_neurons = [int(x) for x in input_neurons.split(",")]

    output_neuron = int(input("\nOutput neuron: "))

    # Make that rssnp based on the user's specifications
    rssnp_dict = {'neurons': neurons, 'synapses': synapses, 'rules': rule_mat, 'init_config': init_config_list, 'rule_status': rule_status_list, 'input_neurons': input_neurons, 'output_neuron': output_neuron}
    print(rssnp_dict)
    rssnp = assign_rssnp(rssnp_dict)

    if rssnp.isValid():
        return rssnp_dict
    else:
        return None

# List of sample RSSNPs

# AND
and_rssnp_minimal = {
    'neurons': 4,
    'synapses': 3,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 3, (2, 0), 2, 1, 0],
        [2, 3, (1, 0), 1, 0, 0], 
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(4)],
    'input_neurons': [0, 1],
    'output_neuron': 3
}

and_rssnp_adversarial = {
    'neurons': 7,
    'synapses': 10,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [0, 3, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [1, 4, (1, 0), 1, 1, 0],
        [2, 6, (1, 2), 1, 0, 0],
        [2, 6, (2, 2), 2, 1, 0],
        [3, 0, (1, 0), 1, 1, 0],
        [3, 2, (1, 0), 1, 1, 0],
        [4, 2, (1, 0), 1, 1, 0],
        [4, 5, (1, 0), 1, 1, 0],
        [5, 1, (1, 0), 1, 1, 0],
    ],
    'init_config': [0, 0, 0, 0, 0, 0, 0],
    'rule_status': [-1 for x in range(11)],
    'input_neurons': [0, 1],
    'output_neuron': 6
}

and_rssnp_extra_rules = {
    'neurons': 4,
    'synapses': 3,
    'rules': [
        [0, 2, (2, 0), 2, 1, 0],
        [0, 2, (1, 0), 1, 1, 0],
        [1, 2, (2, 0), 2, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 3, (2, 0), 2, 1, 0],
        [2, 3, (1, 0), 1, 0, 0], 
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(6)],
    'input_neurons': [0, 1],
    'output_neuron': 3
}

# OR
or_rssnp_minimal = {
    'neurons': 4,
    'synapses': 3,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 3, (1, 0), 1, 1, 0],
        [2, 3, (2, 0), 2, 1, 0],
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(4)],
    'input_neurons': [0, 1],
    'output_neuron': 3
}

or_rssnp_adversarial = {
    'neurons': 7,
    'synapses': 11,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [0, 3, (1, 1), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [1, 5, (1, 1), 1, 1, 0],
        [3, 4, (1, 1), 1, 1, 0],
        [3, 5, (1, 1), 1, 1, 0],
        [4, 2, (1, 1), 1, 1, 0],
        [4, 3, (1, 1), 1, 1, 0],
        [5, 4, (1, 1), 1, 1, 0],
        [5, 1, (1, 1), 1, 1, 0],
        [2, 6, (1, 0), 1, 1, 0],
        [2, 6, (2, 0), 2, 1, 0],
    ],
    'init_config': [0, 0, 0, 0, 0, 0, 0],
    'rule_status': [-1 for x in range(12)],
    'input_neurons': [0, 1],
    'output_neuron': 6
}

or_rssnp_extra_rules = {
    'neurons': 4,
    'synapses': 3,
    'rules': [
        [0, 2, (2, 0), 2, 1, 0],
        [0, 2, (1, 0), 1, 1, 0],
        [1, 2, (2, 0), 2, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 3, (2, 0), 2, 1, 0],
        [2, 3, (1, 0), 1, 1, 0], 
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(6)],
    'input_neurons': [0, 1],
    'output_neuron': 3
}

# NOT
not_rssnp_minimal = {
    'neurons': 4,
    'synapses': 4,
    'rules': [
        [0, 3, (1, 0), 1, 1, 0],
        [0, 3, (2, 0), 2, 0, 0],
        [1, 0, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 1, (1, 0), 1, 1, 0],
    ],
    'init_config': [0, 1, 1, 0],
    'rule_status': [-1 for x in range(5)],
    'input_neurons': [0],
    'output_neuron': 3
}

not_rssnp_adversarial = {
    'neurons': 6,
    'synapses': 10,
    'rules': [
        [0, 1, (1, 0), 1, 1, 0],
        [0, 3, (1, 0), 1, 1, 0],
        [0, 3, (2, 0), 2, 0, 0],
        [0, 4, (1, 0), 1, 1, 0],
        [1, 0, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 1, (1, 0), 1, 1, 0],
        [4, 2, (1, 0), 1, 1, 0],
        [4, 5, (1, 0), 1, 1, 0],
        [5, 2, (1, 0), 1, 1, 0],
        [5, 0, (1, 0), 1, 1, 0],
    ],
    'init_config': [0, 1, 1, 0, 0, 0],
    'rule_status': [-1 for x in range(11)],
    'input_neurons': [0],
    'output_neuron': 3
}

not_rssnp_extra_rules = {
    'neurons': 4,
    'synapses': 6,
    'rules': [
        [0, 1, (1, 0), 1, 1, 0],
        [0, 3, (1, 0), 1, 1, 0],
        [0, 3, (2, 0), 2, 0, 0],
        [1, 0, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 0, (1, 0), 1, 1, 0],
        [2, 1, (1, 0), 1, 1, 0],
    ],
    'init_config': [0, 1, 1, 0],
    'rule_status': [-1 for x in range(7)],
    'input_neurons': [0],
    'output_neuron': 3
}

# ADD
add_rssnp_minimal = {
    'neurons': 4,
    'synapses': 3,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [2, 3, (1, 0), 1, 1, 0],
        [2, 3, (2, 0), 1, 0, 0],
        [2, 3, (3, 0), 2, 1, 0],
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(5)],
    'input_neurons': [0, 1],
    'output_neuron': 3
}

add_rssnp_adversarial = {
    'neurons': 7,
    'synapses': 13,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [0, 3, (1, 0), 1, 1, 0],
        [0, 4, (1, 0), 1, 1, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [1, 5, (1, 0), 1, 1, 0],
        [2, 3, (1, 0), 1, 1, 0],
        [2, 3, (2, 0), 1, 0, 0],
        [2, 3, (3, 0), 2, 1, 0],
        [4, 1, (1, 0), 1, 1, 0],
        [4, 2, (1, 0), 1, 1, 0],
        [4, 6, (1, 0), 1, 1, 0],
        [5, 1, (1, 0), 1, 1, 0],
        [5, 2, (1, 0), 1, 1, 0],
        [6, 1, (1, 0), 1, 1, 0],
        [6, 2, (1, 0), 1, 1, 0],
    ],
    'init_config': [0, 0, 0, 0, 0, 0, 0],
    'rule_status': [-1 for x in range(15)],
    'input_neurons': [0, 1],
    'output_neuron': 3
}

add_rssnp_extra_rules = {
    'neurons': 4,
    'synapses': 3,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [0, 2, (2, 0), 2, 2, 0],
        [1, 2, (1, 0), 1, 1, 0],
        [1, 2, (2, 0), 2, 2, 0],
        [2, 3, (1, 0), 1, 1, 0],
        [2, 3, (2, 0), 1, 0, 0],
        [2, 3, (3, 0), 2, 1, 0],
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(8)],
    'input_neurons': [0, 1],
    'output_neuron': 3
}

# SUB
sub_rssnp_minimal = {
    'neurons': 11,
    'synapses': 14,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [0, 3, (1, 0), 1, 1, 0],
        [0, 4, (1, 0), 1, 1, 0],
        [1, 5, (1, 0), 1, 1, 0],
        [2, 9, (1, 0), 1, 1, 0],
        [3, 9, (1, 0), 1, 1, 0],
        [4, 9, (1, 0), 1, 1, 0],
        [5, 9, (1, 0), 1, 1, 0],
        [6, 7, (1, 0), 1, 1, 0],
        [6, 8, (1, 0), 1, 1, 0],
        [7, 8, (1, 0), 1, 1, 0],
        [7, 9, (1, 0), 1, 1, 0],
        [8, 7, (1, 0), 1, 1, 0],
        [9, 10, (1, 0), 1, 0, 0],
        [9, 10, (2, 0), 1, 1, 0],
        [9, 10, (3, 0), 2, 0, 0],
        [9, 10, (4, 0), 4, 1, 0],
        [9, 10, (5, 0), 5, 0, 0],
        [9, 10, (6, 0), 5, 1, 0],
    ],
    'init_config': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'rule_status': [-1 for x in range(19)],
    'input_neurons': [0, 1],
    'output_neuron': 10
}

sub_rssnp_adversarial = {
    'neurons': 14,
    'synapses': 26,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [0, 3, (1, 0), 1, 1, 0],
        [0, 4, (1, 0), 1, 1, 0],
        [1, 5, (1, 0), 1, 1, 0],
        [1, 11, (1, 0), 1, 1, 0],
        [2, 7, (1, 0), 1, 1, 0],
        [2, 9, (1, 0), 1, 1, 0],
        [3, 9, (1, 0), 1, 1, 0],
        [4, 9, (1, 0), 1, 1, 0],
        [5, 9, (1, 0), 1, 1, 0],
        [5, 9, (2, 0), 2, 0, 0],
        [6, 7, (1, 0), 1, 1, 0],
        [6, 8, (1, 0), 1, 1, 0],
        [6, 12, (1, 0), 1, 1, 0],
        [7, 5, (2, 0), 2, 1, 0],
        [7, 8, (1, 0), 1, 1, 0],
        [7, 9, (1, 0), 1, 1, 0],
        [7, 13, (2, 0), 2, 1, 0],
        [8, 7, (1, 0), 1, 1, 0],
        [8, 12, (1, 0), 1, 1, 0],
        [9, 10, (1, 0), 1, 0, 0],
        [9, 10, (2, 0), 1, 1, 0],
        [9, 10, (3, 0), 2, 0, 0],
        [9, 10, (4, 0), 4, 1, 0],
        [9, 10, (5, 0), 5, 0, 0],
        [9, 10, (6, 0), 5, 1, 0],
        [11, 9, (1, 0), 1, 1, 0],
        [12, 2, (1, 0), 1, 1, 0],
        [12, 3, (1, 0), 1, 1, 0],
        [12, 4, (1, 0), 1, 1, 0],
        [13, 0, (1, 0), 1, 1, 0],
        [13, 1, (1, 0), 1, 1, 0],

    ],
    'init_config': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'rule_status': [-1 for x in range(32)],
    'input_neurons': [0, 1],
    'output_neuron': 10
}

sub_rssnp_extra_rules = {
    'neurons': 11,
    'synapses': 18,
    'rules': [
        [0, 2, (1, 0), 1, 1, 0],
        [0, 3, (1, 0), 1, 1, 0],
        [0, 4, (1, 0), 1, 1, 0],
        [1, 3, (1, 0), 1, 1, 0],
        [1, 5, (1, 0), 1, 1, 0],
        [2, 9, (1, 0), 1, 1, 0],
        [3, 9, (2, 0), 2, 2, 0],
        [3, 9, (1, 0), 1, 1, 0],
        [4, 5, (1, 0), 1, 1, 0],
        [4, 9, (1, 0), 1, 1, 0],
        [5, 9, (1, 0), 1, 1, 0],
        [5, 9, (2, 0), 2, 2, 0],
        [5, 9, (3, 0), 2, 1, 0],
        [6, 7, (1, 0), 1, 1, 0],
        [6, 8, (1, 0), 1, 1, 0],
        [7, 5, (1, 0), 1, 1, 0],
        [7, 8, (1, 0), 1, 1, 0],
        [7, 9, (1, 0), 1, 1, 0],
        [8, 6, (1, 0), 1, 1, 0],
        [8, 7, (1, 0), 1, 1, 0],
        [9, 10, (1, 0), 1, 0, 0],
        [9, 10, (2, 0), 1, 1, 0],
        [9, 10, (3, 0), 2, 0, 0],
        [9, 10, (4, 0), 4, 1, 0],
        [9, 10, (5, 0), 5, 0, 0],
        [9, 10, (6, 0), 5, 1, 0],
        [9, 10, (7, 0), 7, 1, 0],
        [9, 10, (8, 0), 8, 0, 0],
    ],
    'init_config': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'rule_status': [-1 for x in range(28)],
    'input_neurons': [0, 1],
    'output_neuron': 10
}

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

def create_empty_yaml(runs, generations, population_size, savefile_name):
	empty_dict = {}
	conf_save(savefile_name, empty_dict)
	ga_params = conf_load(savefile_name)
	ga_params['run_total'] = runs
	ga_params['gen_total'] = generations
	ga_params['runs'] = {}
	for i in range(runs):
		ga_params['runs'][i] = {}
		ga_params['runs'][i]['generations'] = {}
		for j in range(generations):
			ga_params['runs'][i]['generations'][j] = {}
			ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'] = {}
			for k in range(population_size):
				ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'][k] = {} 
	conf_save(savefile_name, ga_params)
	return ga_params

def continue_create_empty_yaml(savefile_name):
	ga_params = conf_load(savefile_name)
	run_start = ga_params['run_total']

	for i in range(run_start, ga_params['run_total'] + ga_params['runs_pending']):
		ga_params['runs'][i] = {}
		ga_params['runs'][i]['generations'] = {}
		for j in range(ga_params['gen_total'] + ga_params['gens_pending']):
			ga_params['runs'][i]['generations'][j] = {}
			ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'] = {}
			for k in range(ga_params['populations_pending']  + ga_params['runs'][0]['population_size']):
				ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'][k] = {} 
	conf_save(savefile_name, ga_params)
	return ga_params


def prompt_make_newsavefile(ga_params, loadfile_name, load_directory):
	newfile_choice = input("Would you like to use another savefile or use the existing savefile (Y/N): ")

	newfile_choice = True if newfile_choice == 'Y' else False

	newloadfile_name = None

	if newfile_choice == True:
		newloadfile_name = os.path.join(load_directory, input("What is the name of the new savefile: "))
		conf_save(newloadfile_name, ga_params)
		#print(ga_params)

	else:
		conf_save(loadfile_name, ga_params)
		newloadfile_name = loadfile_name
	return newloadfile_name

def execute_experiment_gpu(ga, gaeval, stats, loadfile_name):
    gaeval.no_of_gen = stats['gen_total']
    gaeval.no_of_run = stats['run_total']

    gaeval.opt_fitness = 50
    gaeval.max_fitness = 100
    gaeval.list_of_runs = []
    print("Now executing experiment involving", loadfile_name)
    stats_run = stats['runs'][gaeval.no_of_run]
    start_new = False
    start_from_gen = True
    rssnp = None
    gaeval.run(ga, rssnp, stats_run['population_size'], stats_run['fitness_function'],  gaeval.no_of_run, gaeval.no_of_gen, stats_run['mutation_rate'], loadfile_name, stats_run['selection_func'], start_new, start_from_gen)

def assign_rssnp(rssnp):
    # initialize the RSSNP system
    system = RSSNPSystem()

    system.n = rssnp['neurons']    # number of neurons
    system.l = rssnp['synapses']   # number of synapses
    system.m = len(rssnp['rules'])    # number of rules

    system.rule = rssnp['rules']

    system.configuration_init = rssnp['init_config']    # starting configuration (number of spikes per neuron)
    system.ruleStatus = rssnp['rule_status']            # initial status of each rule (set to -1)
    system.inputs = rssnp['input_neurons']
    system.outputs = rssnp['output_neuron']

    return system
 
class RSSNPSystem:
    """

    Definition to represent an RSSNP System
    with m rules, n neruons, and l synapses.

    Attributes
    ----------
    configuration: numpy int array
        The Configuration Vector CF = [cf_1, ..., c_i, ..., cf_m]
        where
            c_i is the amount of spikes in neuron i at time t

    rule: numpy tuple array rl_j = (x_j, y_j, E_j, c_j, p_j, d_j)
        The current Rule Representation RL = [rl_1, ..., rl_j, ..., rl_m]
        where
            x_j is the source neuron
            y_j is the destination neuron
            E_j is the regular expression
            c_j is the amount of spikes to be consumed
            p_j is the amount of spikes to be produced
            d_j is the delay

    ruleStatus: numpy int array
        The Rule Status Vector RS = [rs_1, ..., rs_j, ..., rs_m]
        where
            rs_j =  t'_j,   if rl_j is active and will send spikes at t + t'_j;
                    -1,     otherwise

    systemState: numpy tuple pair
        The System State SS = (configuration, ruleStatus)
        where
            configuration is the Configuration Vector
            ruleStatus is the Rule Status Vector

    synapseStatus: numpy int array
        The Synapse Status Vector SY = [sy_1, ..., sy_h, ..., sy_l]
        where
            sy_h =  1,  if syn_h is open at time t
                    0,  otherwise

    applicability: numpy int array
        The Applicability Vector AP = [ap_1, ..., ap_j, ..., ap_m]
        where
            ap_j =  1, if rl_j is applicable
                    0, otherwise

    activation: numpy int array
        The Activation Vector AC = [ac_1, ..., ac_j, ..., ac_m]
        where
            ac_j =  1, if rl_j will be applied
                    0, otherwise

    loss: numpy int array
        The Loss Vector LS = [ls_1, ..., ls_i, ..., ls_n]
        where
            ls_i =  c_j, amount of spikes to be consumed for neuron i
                    0, otherwise
    """

    l = 0
    m = 0  # Number of Rules
    n = 0  # Number of Neurons
    rule = []  # Rules in the System
    configuration = []
    configuration_init = []
    ruleStatus = []
    synapseDict = {}  # Dictionary of Synapse
    inputs = []
    outputs = None  # neuron which contains the output of the system (environment) / must not fire rules
    in_spiketrain = []  # input spike trains (Example: {'index': 0, 'input': '010101'})
    out_spiketrain = []

    def __repr__(self):
        return str(self.rule) + " | " + str(self.configuration_init)

    def initializeSynapseDict(self):
        """

            Initializes the dictionary to be used in synMap

        """

        ctr = 0
        for j in range(0, self.m):
            rule = self.rule[j]
            # Adds a new item in the dict if pair doesn't exist
            if (rule[0], rule[1]) not in self.synapseDict:
                self.synapseDict[(rule[0], rule[1])] = ctr
                ctr += 1

    def synMap(self, x_j, y_j):
        """

            Maps the pair (x_j, y_j) to its respective synapse

            Returns
            -------
            int
                The synapse number containing the pair

        """

        return self.synapseDict[(x_j, y_j)]

    def synapseStat(self, ruleStatus):
        """

            Computes for the status of the synapses

            Returns
            -------
            numpy vector
                The Synapse Status Vector at time t

        """

        # Initialized the synapseStatus vector
        synapseStatus = np.full(self.l, 1, dtype=int)
        for j in range(0, self.m):
            rule = self.rule[j]
            # Maps the pair (x_j, y_j) to its respective synapse
            h = self.synMap(rule[0], rule[1])
            # Change the status of a synapse if it contains an active rule
            if ruleStatus[j] >= 0:
                synapseStatus[h] = 0

        return synapseStatus

    def ruleApplicability(self, configuration, synapseStatus):
        """

            Determines which rules are applicable

            Returns
            -------
            numpy vector
                The Applicability Vector at time t

        """
        #print("config",configuration)
        # Initializes the applicability vector
        applicability = np.zeros(self.m, dtype=int)
        for j in range(0, self.m):
            rule = self.rule[j]
            # Maps the pair (x_j, y_j) to its respective synapse
            h = self.synMap(rule[0], rule[1])
            # Checks if the rule can be applied
            if synapseStatus[h] == 1 and configuration[rule[0]] >= rule[3]:
                if rule[2][1] == 0 and configuration[rule[0]] == rule[2][0]:
                    applicability[j] = 1
                elif rule[2][1] != 0 and (
                        configuration[rule[0]] - rule[2][0]) % rule[2][1] == 0:
                    applicability[j] = 1

        return applicability

    def activationVectors(self, applicability):
        """

            Determines which rules will be activated per time step

            Returns
            -------
            list
                List containing the Activation Vectors at time t

        """
        #print("app =",applicability)
        activationMatrix = self.synapseRestriction(applicability)
        finalActivationMatrix = self.costRestriction(activationMatrix)

        return finalActivationMatrix

    def synapseRestriction(self, applicability):
        """

            Determines which rules will be activated considering the synapse restriction

            Returns
            -------
            list
                List containing the temporary Activation Vectors at time t

        """

        SA = []
        for h in range(self.l):
            SA.append([])

        for j in range(0, self.m):
            if applicability[j] == 1:
                rule = self.rule[j]
                h = self.synMap(rule[0], rule[1])
                SA[h].append(j + 1)

        for h in range(0, self.l):
            if len(SA[h]) == 0:
                SA[h].append(0)

        SC = []
        for elem in itertools.product(*SA):
            SC.append(elem)

        activationMatrix = []
        for elem in SC:
            activation = [0] * self.m
            for h in range(0, self.l):
                j = elem[h]
                if j != 0:
                    activation[j - 1] = 1

            activationMatrix.append(activation)
            #print("AM = ",activationMatrix)
        return activationMatrix

    def costRestriction(self, activationMatrix):
        """

            Determines which rules will be activated considering all restrictions

            Returns
            -------
            list
                List containing the final Activation Vectors at time t

        """

        finalActivationMatrix = []

        for activation in activationMatrix:
            DC = []

            for i in range(0, self.n):
                DC.append([])

            for j in range(0, self.m):
                if activation[j] == 1:
                    DC[self.rule[j][0]].append(self.rule[j][3])

            for i in range(0, self.n):
                if len(DC[i]) == 0:
                    DC[i].append(0)
            #print("DC = ",DC)
            CC = []
            for elem in itertools.product(*DC):
                CC.append(elem)
            #print("CC = ", CC)
            finalActivation = activation

            for elem in CC:
                #print("elem ",elem)
                #print("rule[3] elem[rule[0]]")
                for j in range(0, self.m):
                    rule = self.rule[j]
                    #print("j=",j)
                    #print(rule[3],"         ",elem[rule[0]])
                    if rule[3] > elem[rule[0]]:
                        finalActivation[j] = 0

            finalActivationMatrix.append(finalActivation)
            #print("FAM = ",finalActivationMatrix)
        return finalActivationMatrix

    def applyRules(self, activationMatrix, ruleStatus):
        """


        """
        lossMatrix = []
        ruleStatusMatrix = []

        for activation in activationMatrix:
            ls = [0 for i in range(0, self.n)]
            rs = [0 for i in range(0, self.m)]

            for j in range(0, self.m):
                rs[j] = ruleStatus[j]
                if activation[j] == 1:
                    rule = self.rule[j]
                    ls[rule[0]] = rule[3]
                    rs[j] = rule[5]
            lossMatrix.append(ls)
            ruleStatusMatrix.append(rs)

        return ruleStatusMatrix, lossMatrix

    def nextState(self, configuration, ruleStatusMatrix, lossMatrix):
        configurationMatrix = []
        nextRuleStatusMatrix = []
        for row in range(0, len(ruleStatusMatrix)):
            cf = []
            for i in range(0, self.n):
                cf.append(configuration[i] - lossMatrix[row][i])

            ruleStatus = ruleStatusMatrix[row]
            for j in range(0, self.m):
                if ruleStatusMatrix[row][j] == 0:
                    y_j = self.rule[j][1]
                    p_j = self.rule[j][4]
                    cf[y_j] += p_j
                if ruleStatusMatrix[row][j] >= 0:
                    ruleStatusMatrix[row][j] -= 1

            configurationMatrix.append(cf)
            nextRuleStatusMatrix.append(ruleStatus)
        return configurationMatrix, nextRuleStatusMatrix

    def main(self, initSystemState, maxSteps):
        unexploredStates = [initSystemState]
        exploredStates = []
        step = 0

        # Get length of longest input spike train
        max_input_bits = 0
        for input_neuron in self.in_spiketrain:
            if max_input_bits < len(input_neuron['input']):
                max_input_bits = len(input_neuron['input'])
        #print("max input bits is ", max_input_bits)
        self.initializeSynapseDict()
        
        from_input_state = False
        while unexploredStates != [] and step <= 3*maxSteps:
            systemState = unexploredStates.pop(0)

            self.configuration = systemState[0]
            self.ruleStatus = systemState[1]

            synapseStatus = self.synapseStat(self.ruleStatus)
            applicability = self.ruleApplicability(self.configuration, synapseStatus)
            activation = self.activationVectors(applicability)
            ruleStatusMatrix, lossMatrix = self.applyRules(activation, self.ruleStatus)
            configurationMatrix, nextRuleStatusMatrix = self.nextState(self.configuration, ruleStatusMatrix, lossMatrix)

            if from_input_state:
                from_input_state = False
            else:
                exploredStates.append(systemState)

            for row in range(0, len(ruleStatusMatrix)):
                nextSystemState = (configurationMatrix[row],nextRuleStatusMatrix[row])
                self.out_spiketrain.append(nextSystemState[0][self.outputs])
                nextSystemState[0][self.outputs] = 0
                if step <= max_input_bits:
                    # Add input spike train to neurons
                    for input_neuron in self.in_spiketrain:
                        nextSystemState[0][input_neuron['index']] += input_neuron['input'][step] if step < len(input_neuron['input']) else 0
                    unexploredStates.append(nextSystemState)
                    from_input_state = True
                elif not (nextSystemState in (unexploredStates + exploredStates)):
                    unexploredStates.append(nextSystemState)
            step += 1
           
        return self.out_spiketrain[0:max_input_bits]
        #return self.out_spiketrain

    def randomize(self, mutation_rate=2):
        """
        
            Randomly decrease the resources of the system

            Returns
            -------
            RSSNP System
                An RSSNP system with mutated variables

        """
        deleted_rules = []
        for i in range(0, self.m):
            # Do not change anything about input neurons
            if self.rule[i][0] in self.inputs or self.rule[i][1] in self.inputs:
                continue

            # Do not change the output neuron
            if self.rule[i][0] == self.outputs or self.rule[i][1] == self.outputs:
                continue

            # Small chance to delete rule
            if randint(0, mutation_rate-1) == 0:
                deleted_rules.append(i)
                continue

            # Change connected synapses
            if randint(0, mutation_rate-1) == 0:
                # Cannot change input and output neurons connection
                self.rule[i][randint(0, 1)] = randint(0, self.n - 1)
                while self.rule[i][0] == self.outputs:  # Cannot have the environment fire spikes
                    self.rule[i][randint(0, 1)] = randint(0, self.n - 1)

                # cannot have synapse pointing to the same neuron
                while self.rule[i][0] == self.rule[i][1]:
                    self.rule[i][randint(0, 1)] = randint(0, self.n - 1)

            # Modify regular expression
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][2] = (randint(1, self.rule[i][2][0]), self.rule[i][2][1])
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][2] = (self.rule[i][2][0], randint(0, self.rule[i][2][1]))

            # Consumed spikes must be within regular expression
            if self.rule[i][2][0] < self.rule[i][3] and self.rule[i][2][1] == 0:
                self.rule[i][3] = self.rule[i][2][0]

            # Change consumed spikes
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][3] = randint(1, self.rule[i][3])

            # Change produced spikes
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][4] = randint(0, self.rule[i][4])

            # Cannot produce more spikes than consumed
            if self.rule[i][4] > self.rule[i][3]:
                self.rule[i][4] = self.rule[i][3]

        delete_counter = 0
        for index in deleted_rules:
            self.rule.pop(index - delete_counter)
            self.m -= 1
            delete_counter += 1
            
        # Change initial configuration
        for i in range(0, self.n):
            if randint(0,mutation_rate-1) == 0:
                self.configuration_init[i] = randint(0, self.configuration_init[i])

        return self

    def intersectionCheck(self, i, j):
        # Same synapse
        if self.rule[i][1] == self.rule[j][1]:
            return False
        # Different synapse, but same source neuron with varying consumed spikes
        elif self.rule[i][3] != self.rule[j][3]:
            return False
        else:
            return True

    def checkDeterministic(self):
        # delete duplicate rules because for some reason the code below doesnt fix this problem
        deleted_rules = []
        for i in range(0, self.m - 1):
            for j in range(i + 1, self.m):
                if self.rule[i] == self.rule[j] and not j in deleted_rules:    # identical rules
                    deleted_rules.append(j)
        delete_ctr = 0
        for index in deleted_rules:
            self.rule.pop(index - delete_ctr)
            self.m -= 1
            delete_ctr += 1

        for i in range(0, self.m - 1):
            for j in range(i + 1, self.m):
                # disregard if from different source neurons
                if self.rule[i][0] != self.rule[j][0]: continue
                
                # value will intersect if no kleene star
                if self.rule[i][2][0] == self.rule[j][2][0]:
                    if not self.intersectionCheck(i,j):
                        return False
                # intersection exists
                if str(parse("a{" + str(self.rule[i][2][0]) + "}(a{" + str(self.rule[i][2][1]) + "})*") & parse("a{" + str(self.rule[j][2][0]) + "}(a{" + str(self.rule[j][2][1]) + "})*")) != '[]':                   
                    if not self.intersectionCheck(i, j):
                        return False

                # e_i = self.rule[i][2]
                # e_j = self.rule[j][2]

                # if e_i[1] == 0 and e_j[1] == 0:
                #     if e_i[0] == e_j[0]:
                #         if not self.intersectionCheck(i, j):
                #             return False
                # elif e_i[1] == 0:
                #     # Intersection
                #     if (e_i[0] - e_j[0]) / e_j[1] % 1 == 0:
                #         if not self.intersectionCheck(i, j):
                #             return False
                # elif e_j[1] == 0:
                #     # Intersection
                #     if (e_j[0] - e_i[0]) / e_i[1] % 1 == 0:
                #         if not self.intersectionCheck(i, j):
                #             return False
                # else:
                #     try:
                #         c1 = (e_j[0] - e_i[0]) % e_i[1] * modinv(e_j[1], e_i[1])
                #         c2 = (e_i[1] * c1 - (e_j[0] - e_i[0])) / e_j[1]

                #         # If regular expression of both rules have an intersection
                #         if c1 % 1 == 0 and c2 % 1 == 0:
                #             if not self.intersectionCheck(i, j):
                #                 return False
                #     except Exception:  # Modular inverse does not exist
                #         # value will intersect if j = 0
                #         if self.rule[i][2][0] == self.rule[j][2][0]:
                #             return False
                #         # intersection exists
                #         if str(parse("a{" + str(self.rule[i][2][0]) + "}(a{" + str(self.rule[i][2][1]) + "})*") & parse("a{" + str(self.rule[j][2][0]) + "}(a{" + str(self.rule[j][2][1]) + "})*")) != '[]':
                #             return False
                #         return True
        return True

    def checkPath(self, unexploredEdges, edges):
        exploredEdges = []

        while unexploredEdges != []:
            curr_edge = unexploredEdges.pop(0)
            
            if curr_edge[1] == self.outputs:
                return True
            
            for index, edge in enumerate(edges):
                if edge[0] == curr_edge[1]:
                    unexploredEdges.append(edge)
                    edges.pop(index)
            
            exploredEdges.append(curr_edge)
            
        return False

    def checkPathInToOut(self):
        for num in self.inputs:
            edges = deepcopy(self.rule)
            unexploredEdges = []
            
            # Add all edges connected to the input node
            for index, edge in enumerate(edges):
                if edge[0] == num:
                    unexploredEdges.append(edge)
                    edges.pop(index)
            
            if not self.checkPath(unexploredEdges, edges):
                return False
        
        return True

    def isValid(self):
        if self.checkPathInToOut() and self.checkDeterministic():
            return True
        return False

class SNPGeneticAlgoGPU:	

	pop = None

	def assign_fitness(self, output_dataset, output_spike_train, function, len_dataset = 0, max_row_width = 0, max_col_width = 0, output_dataset_lengths = None, output_rssnp_lengths = None):
	    result = 0
	    #print("output_dataset ", output_dataset, " ost ", output_spike_train)
	    
	    if function == 0:
	    	result = GPUlcs(output_dataset, output_spike_train, len_dataset)        
	    	#pass
	    elif function == 1:
	        result = GPULCSubStr(output_dataset, output_spike_train, len_dataset)
	    	#pass
	    elif function == 2:
	    	result = GPUeditDistDP(output_dataset, output_spike_train, max_row_width, max_col_width, len_dataset, output_dataset_lengths, output_rssnp_lengths)
	    	#pass

	    # print(result)
	    return result



	def selection(self, selection_func):
		parents = []
		if selection_func == 0:
		    # Get top 50%
		    parents = self.pop[:int(len(self.pop)/2)]
		elif selection_func == 1:
		    # Get random 50%
		    total_fitness = 0
		    for chrom in self.pop:
		        total_fitness += chrom['fitness']

		    if total_fitness != 0:
		        i = 0
		        while len(parents) != int(len(self.pop)/2):
		            if random.randint(0,total_fitness) <= self.pop[-i]['fitness'] and not (self.pop[-i] in parents):    # chance to become parent is fitness/total fitness
		                parents.insert(0,self.pop[-i])
		            i = (i + 1) % len(self.pop)
		    else:
		        parents = self.pop[:int(len(self.pop)/4)]
		    print("chose this selection 1")
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
		    print("chose this selection 2")
		print("parents returned by selection are ", parents)
		return parents



	def crossover(self, mutation_rate, selection_func):
		population_size = len(self.pop)
		# Get only parents
		parents = self.selection(selection_func)
		# delete half of the population
		self.pop = self.pop[:(int(len(self.pop)/2))]

		print("parents are ", parents)
		#rand_rule = random.randint(0,total_fitness)		
		#mutate_rand = random.randint()
		#rand_changed_to = random.randint()
		#pass
		i = 0
		while True:
			cross_counter = 0
			print("majestic length of parents is ", len(parents))
			size_tuple = 4

			crossover_indexes = crossover_gpu_defined(len(parents), parents, self.pop)
			#crossover_indexes = get_param(crossover_indexes, population_size)
			
			num_crosses = len(parents)

			print("num crosses is ", num_crosses)#, " random parent 1 is ", random_rule_parents[0], " while 2 is ", random_rule_parents[1])

			while True:
				parent1 = deepcopy(parents[i % len(parents)])  # best parent
				parent2 = deepcopy(parents[(i + 1) % len(parents)]) # 2nd best

				rule1 = int(crossover_indexes[i % len(parents)]) 


				rule2 = int(crossover_indexes[(i + 1) % len(parents) + len(parents)])
				print("rule 1 is ", rule1, " while rule 2 is ", rule2)
				 # Choose random rule to swap
				
				backup1 = deepcopy(parent1)
				backup2 = deepcopy(parent2)
				# Swap rules
				parent1['system'].rule[rule1], parent2['system'].rule[rule2] = parent2['system'].rule[rule2], parent1['system'].rule[rule1]
				# Mutate
				parent1['system'].randomize(mutation_rate)
				parent2['system'].randomize(mutation_rate)

				if parent1['system'].isValid() and parent2['system'].isValid():
					parent1['system'].out_spiketrain = []
					parent2['system'].out_spiketrain = []
					self.pop.extend([parent1,parent2])
					print("passed first")
					if len(self.pop) > population_size:
						while len(self.pop) > population_size:
							self.pop = self.pop[:-1]
						return
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

			print("comparing", len(self.pop), population_size)
			if len(self.pop) == population_size and not i == 0:
				return

			i += 1

		

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
		    print("numbers is ", numbers)
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
		max_col_width = 0
		input = open(filename, 'r')
		for line in input:
		    numbers = line.strip('\n').split(',')
		    #print("numbers is ", numbers)
		    spike_index = 0
		    for number in numbers[-1]:
		        dataset[int(line_index/max_numpy_arraylen)][int(line_index%max_numpy_arraylen)][spike_index] = int(number)
		        spike_index += 1
		    #print("numbers array is ", dataset[int(line_index/max_numpy_arraylen)])
		    if max_row_width < spike_index:
		        max_row_width = spike_index
		    if max_col_width < spike_index:
		        max_col_width = spike_index
		    line_index += 1
		return numpy.array(dataset), max_row_width, max_col_width



	def evaluate(self, chromosome, ga_params, fitness_func, selection_func):
		inout_pairs_view = []
		row_width = 0
		col_width = 0
		dataset = self.dataset_arrange(ga_params['test_cases_path'])
		if fitness_func == 2:
			test_case_file = open(ga_params['test_cases_path'], 'r')
			len_dataset = len(list(test_case_file))
			dataset2, row_width, col_width = self.dataset_arrange3(len_dataset, ga_params['test_cases_path'])
			output_dataset_lengths = numpy.zeros(len_dataset)
			for z in range(0, len_dataset):
				bitstring_length = len(dataset[z])
				single_length = int(bitstring_length / 3)
				i = numpy.arange(0, bitstring_length).reshape(bitstring_length, 1, 1)
				j = numpy.arange(0, bitstring_length, single_length)
				b = numpy.broadcast(i, j)
				inout_pairs_view.append((i + j))
				#print(inout_pairs_view)
				#print("datasub input 1 ", dataset[z][inout_pairs_view[z][0][0][0]:inout_pairs_view[z][single_length - 1][0][0]])
				#print("datasub input 2 ", dataset[z][inout_pairs_view[z][0][0][1]:inout_pairs_view[z][single_length - 1][0][1]])
				#print("minuend ", len(dataset[z]), " subtrahend ", inout_pairs_view[z][0][0][-1], "datasub output", dataset[z][inout_pairs_view[z][0][0][-1]:inout_pairs_view[z][single_length - 1][0][-1]])
				output_dataset_lengths[z] = len(dataset[z]) - inout_pairs_view[z][0][0][2]
				maxSteps = 3 * output_dataset_lengths[z]  
				#print("maxsteps ", maxSteps)
				output_dataset = dataset[z][inout_pairs_view[z][0][0][2]:inout_pairs_view[z][single_length - 1][0][2]] 	
				output_dataset = [int(x) for x in list(output_dataset)]
				input_length = (bitstring_length - single_length)/single_length
				chromosome['system'].in_spiketrain = []
				for index in range(len(chromosome['system'].inputs)):
					
					chromosome['system'].in_spiketrain.append({
	                    'index': chromosome['system'].inputs[index],
	                    'input': [int(x) for x in list(dataset[z][inout_pairs_view[z][0][0][index]:inout_pairs_view[z][single_length - 1][0][index]])]
	                })
					print("added at index ", index, " is ", chromosome['system'].in_spiketrain[index])
					#print("chrom in spike ", chromosome['system'].in_spiketrain)
				chromosome['system'].out_spiketrain = []
				config = deepcopy(chromosome['system'].configuration_init)
				#print("config is ", config)
				chromosome['out_pairs'].append((chromosome['system'].main((config, chromosome['system'].ruleStatus), maxSteps), output_dataset))
			
			line_index = 0
			output_rssnp_lengths = numpy.zeros(len(list(chromosome['out_pairs'])))
			max_spike_size = 20
			output_rssnp_numpy = numpy.zeros(shape=(int(len_dataset/max_numpy_arraylen) + 1, max_numpy_arraylen, max_spike_size), dtype=numpy.int32)
			
			n = None
			print("the length of out_pairs is ", len(list(chromosome['out_pairs'])))
			for m in list(chromosome['out_pairs']):
			
				n = numpy.asarray(m[0], dtype=numpy.int32)
				#numpy.lib.pad(n, ((0,0),(0,max_spike_size - len(m[0]))), 'constant', constant_values=(0))
				#print("inputs are ", chromosome['system'].in_spiketrain[line_index])
				print("line index is ", line_index, "n is ", n, "n shape is ", n.shape, " compared to ", max_spike_size - len(m[0]))
				#numpy.concatenate((n,np.zeros((n.shape[0], max_spike_size - len(m[0])))), axis=0)
				#numpy.hstack([n,np.zeros([n.shape[0], max_spike_size - len(m[0])])])
				output_rssnp_lengths[line_index] = len(n)
				n = based_init(n, max_spike_size - len(m[0]))
				print("numpy n is ", n)
				for index_value in range(max_spike_size):
					output_rssnp_numpy[int(line_index/max_numpy_arraylen)][int(line_index%max_numpy_arraylen)][index_value] = n[index_value]

				# if line_index == 0:
				# 	output_rssnp_numpy = n 
				# else:
				# 	print("shapes of orn and n respectively are ", output_rssnp_numpy.shape, n.shape)
				# 	numpy.stack((output_rssnp_numpy, n), axis=1)
				line_index += 1
			print("orn is ", output_rssnp_numpy)


			#print("chromosome out pairs is ", chromosome['out_pairs'])
			print("EXITED with ", output_rssnp_lengths, " and ", output_dataset_lengths)
			chromosome['fitness'] = int(self.assign_fitness(dataset2, output_rssnp_numpy, fitness_func, len_dataset, row_width, col_width, output_dataset_lengths, output_rssnp_lengths)/len(dataset))
		
		else:
			len_dataset = len(dataset)	
			for z in range(0, len_dataset):
				bitstring_length = len(dataset[z])
				single_length = int(bitstring_length / 3)
				i = numpy.arange(0, bitstring_length).reshape(bitstring_length, 1, 1)
				j = numpy.arange(0, bitstring_length, single_length)
				b = numpy.broadcast(i, j) 
				inout_pairs_view.append((i + j))
				#print("appended ", inout_pairs_view[z])
				#print("datasub input 1 ", dataset[z][inout_pairs_view[z][0][0][0]:inout_pairs_view[z][single_length - 1][0][0]])
				#print("datasub input 2 ", dataset[z][inout_pairs_view[z][0][0][1]:inout_pairs_view[z][single_length - 1][0][1]])
				#print("minuend ", len(dataset[z]), " subtrahend ", inout_pairs_view[z][0][0][-1], "datasub output", dataset[z][inout_pairs_view[z][0][0][-1]:inout_pairs_view[z][single_length - 1][0][-1]])
				dataset_len = len(dataset[z]) - inout_pairs_view[z][0][0][2]
				maxSteps = 3*dataset_len  
				#print("maxsteps ", maxSteps)
				#print("inout pairs view is ", inout_pairs_view[z])
				#print("index 1: ", inout_pairs_view[z][0][0][2], " index 2: ", inout_pairs_view[z][single_length - 1][0][2])
				output_dataset = dataset[z][inout_pairs_view[z][0][0][2]:inout_pairs_view[z][single_length - 1][0][2]] 	
				output_dataset = [int(x) for x in list(output_dataset)]
				#print("output dataset is ", output_dataset , " at index ", z)
				input_length = (bitstring_length - single_length)/single_length
				chromosome['system'].in_spiketrain = []
				for index in range(len(chromosome['system'].inputs)):
					chromosome['system'].in_spiketrain.append({
	                    'index': chromosome['system'].inputs[index],
	                    'input': [int(x) for x in list(dataset[z][inout_pairs_view[z][0][0][index]:inout_pairs_view[z][single_length - 1][0][index]])]
	                })
					#print("chrom in spike ", chromosome['system'].in_spiketrain)
				chromosome['system'].out_spiketrain = []
				config = deepcopy(chromosome['system'].configuration_init)
				#print("config is ", config)
				chromosome['out_pairs'].append((chromosome['system'].main((config, chromosome['system'].ruleStatus), maxSteps), output_dataset))
				value = self.assign_fitness(chromosome['out_pairs'][z][1], chromosome['out_pairs'][z][0], fitness_func, len_dataset)
				#print("dataset len is ", dataset_len, " while dividend is ", value, " with result of ", value/dataset_len)
				chromosome['fitness'] += (value/dataset_len) * 100
				#print("fitness now is ", chromosome['fitness'], " at z: ", z)
			chromosome['fitness'] = int(chromosome['fitness']/len_dataset)

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

				fitness_func = ga_params['runs'][run_index]['fitness_function']
				selection_func = ga_params['runs'][run_index]['selection_func']
				#print("len of inout", self.inout_pairs)
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

class SNPGeneticAlgoEval:
    opt_fitness = 0
    max_fitness  = 0
    no_of_gen = 0
    no_of_run = 0
    # list_of_runs = []    
    # file_name = None
    # ga_params = None
    #path name is now loadfile_name
    def run(self, genetic_algo, system, size, function, runs, generations, mutation_rate, path_name, selection_func, start_new = True, start_from_gen = False):
        max_fitness_in_run = 0
        self.ga_params = conf_load(path_name)
        self.no_of_run = self.ga_params['run_total'] + self.ga_params['runs_pending']
        self.no_of_gen = self.ga_params['gen_total'] + self.ga_params['gens_pending']
        self.file_name = path_name
        #print("ga params in run is " + str(ga_params))
        if start_new == True:
            print("Starting anew")
            start = 0
            middle = self.ga_params['run_total']
        else:
            start = self.ga_params['run_total']
            middle = 0
        for run in range(start, start + middle + self.ga_params['runs_pending']):
            print("run run baby # " + str(run))
            # Perform a single run of the GA framework
            run_fitness = genetic_algo.simulate(system, size, function, generations, mutation_rate, path_name, run, selection_func, start_from_gen)            
            if  run_fitness > max_fitness_in_run:
                max_fitness_in_run = run_fitness
                if max_fitness_in_run >= self.ga_params['goal_fitness']:
                    break

        print("max fit is " + str(max_fitness_in_run))
        self.max_fitness = max_fitness_in_run
        self.ga_params = conf_load(path_name)
        self.ga_params['max_fitness_in_runs'] = max_fitness_in_run
        conf_save(path_name, self.ga_params)
 
        print("Likelihood of Evolution Leap: ", self.likelihoodOfEvolLeap())#, file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        print("Likelihood of Optimality: ", self.likelihoodOfOptimality())#, file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        print("Average Fitness Value: ", self.avgFitnessValue())#, file=open(path_name + "/Run" + str(run) + "/run.txt","a"))

    def likelihoodOfOptimality(self):
        no_of_opt_run = 0

        # Iterate through all runs
        for run in range(self.no_of_run):
            # Get highest fitness for that run
            #fitness = int(re.search('\'fitness\': (.*), \'out', run[-1]).group(1))
            fitness = self.ga_params['runs'][run]['max_fitness_in_run']
            # Increase number of optimal run if highest fitness is greater than the optimal fitness
            if fitness >= self.opt_fitness: no_of_opt_run += 1
        
        # Get average optimal runs
        return no_of_opt_run / self.no_of_run

    def avgFitnessValue(self):
        fitness_sum = 0
        # Iterate through all runs
        run_index = 0
        for run in range(self.no_of_run):
            # Get highest fitness for that run
            #fitness = int(re.search('\'fitness\': (.*), \'out', run[-1]).group(1))
            fitness = self.ga_params['runs'][run]['max_fitness_in_run']
            run_index += 1
            # Add highest fitness to the sum of fitness for all runs
            fitness_sum += fitness
        
        # Get average fitness 
        return fitness_sum / (self.no_of_run * self.max_fitness)

    def likelihoodOfEvolLeap(self):
        evol_leap = 0

        # Iterate through all runs
        for run in range(self.no_of_run):
            prev_fitness = 0
            leaps = 0
            # Iterate through every generation of a run
            for gen in range(self.no_of_gen):
                # Get fitness of the generation
                #fitness = int(re.search('\'fitness\': (.*), \'out', gen).group(1))
                fitness = self.ga_params['runs'][run]['max_fitness_in_run']
                # If the fitness is greater than the previous gen's fitness, increase number of leap for that run
                if fitness > prev_fitness: leaps += 1
                prev_fitness = fitness
            
            # Get summation of the average no of leaps of all runs
            evol_leap += (leaps / self.no_of_gen)
            print("evol leap is " + str(evol_leap))
        print('no of run is ' + str(self.no_of_run))
        # Return summatoon of the average no of leaps of all runs
        return evol_leap / self.no_of_run

def gpuarray_copy(array: gpuarray.GPUArray):
    array_copy = array.copy()
    array_copy.strides = array.strides
    array_copy.flags.f_contiguous = array.flags.f_contiguous
    array_copy.flags.c_contiguous = array.flags.c_contiguous
    array_copy.flags.forc = array.flags.forc

def GPUlcs(output_dataset, output_spike_train, len_dataset):
    mod = SourceModule("""
    #include <stdlib.h>
    __device__ int max1(int a,int b){
        if(a>b){
            return a;
        }
        else{
            return b;
        }
    }

    __global__ void lc_subsequence(int *X,int *Y,int *res, int *LCSuff,int row_width,int col_width){
       int j = blockIdx.x * blockDim.x + threadIdx.x;
       if (j < col_width){
            for (int i = 1; i < row_width; i++) {
                if (i == 0 || j == 0){
                    LCSuff[i*col_width+j] = 0;
                    __syncthreads();
                }
                else if (X[i-1] == Y[j-1]) {  
                    LCSuff[i*col_width+j] = LCSuff[((i-1)*col_width)+j-1] + 1;
                    __syncthreads();
                    //printf("compare %d %d\\n",res[0],LCSuff[i*col_width+j]);
                    //printf("res %d %d %d\\n",res[0],i,j);
                    
                } 
                else{
                    LCSuff[i*col_width+j] = max1(LCSuff[(i-1)*col_width+j],LCSuff[i*col_width+j-1]);
                    __syncthreads();
                }
                
            }

        } 
    }


    """)
    LCSQ = mod.get_function("lc_subsequence")
    
    #a = numpy.array(test,dtype=numpy.int32) #row the width
    #b = numpy.array([0,0,0,1],dtype=numpy.int32) #col
    a = numpy.array(output_spike_train,dtype=numpy.int32)
    b = numpy.array(output_dataset,dtype=numpy.int32)
    res = numpy.array([0],dtype=numpy.int32)
    LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)


    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
    res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    drv.memcpy_htod(res_gpu, res)

    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)

    LCSQ(a_gpu,b_gpu,res_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(thread_num,1,1),grid=(thread_num,1,1))
    drv.memcpy_dtoh(res, res_gpu)
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    #print(LCSuff)
    #print(LCSuff[a[1].size][b.size])
    return LCSuff[a.size][b.size]
    #print("res lcs", res)

    #print (lcs(a[1], b)) 

    #return res[0]

def GPULCSubStr2(output_dataset, output_spike_train, len_dataset):
    mod = SourceModule("""
    __global__ void lc_substring(int *X,int *Y,int *res, int *LCSuff,int row_width,int col_width){
       int j = blockIdx.x * blockDim.x + threadIdx.x;
    }
    """)
    LCS = mod.get_function("lc_substring")
    a_gpu = None
    b_gpu = None
    res_gpu = None
    LCSuff_gpu = None
    thread_num = 32
    LCS(a_gpu,b_gpu,res_gpu,LCSuff_gpu, 10, 10, block=(thread_num,1,1),grid=(thread_num,1,1))  

def GPULCSubStr(output_dataset, output_spike_train, len_dataset): 

    mod = SourceModule("""
    #include <stdlib.h>
    __device__ int max1(int a,int b){
        if(a>b){
            return a;
        }
        else{
            return b;
        }
    }
    __global__ void lc_substring(int *X,int *Y,int *res, int *LCSuff,int row_width,int col_width){
       int j = blockIdx.x * blockDim.x + threadIdx.x;
       if (j < col_width){
            for (int i = 0; i < row_width; i++) {
                if (X[i-1] == Y[j-1] && !(i == 0 || j == 0)) {  
                    LCSuff[i*col_width+j] = LCSuff[(i-1)*col_width+j-1] + 1;
                    __syncthreads();
                    //printf("compare %d %d\\n",res[0],LCSuff[i*col_width+j]);
                    res[0] = max1(res[0], LCSuff[i*col_width+j]);
                    __syncthreads();
                    //printf("res %d %d %d\\n",res[0],i,j);
                    
                } 
                else{
                    LCSuff[i*col_width+j] = 0;
                    __syncthreads();
                }
                
            }

        } 
    }


    """)
    LCS = mod.get_function("lc_substring")
    #no = [1,1,1,0,1,0,1]
    #a = numpy.array(no,dtype=numpy.int32) #row the width
    #b = numpy.array([0,0,0,0,0,1],dtype=numpy.int32) #col
    a = numpy.array(output_spike_train,dtype=numpy.int32)
    b = numpy.array(output_dataset,dtype=numpy.int32)
    res = numpy.array([0],dtype=numpy.int32)
    LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)

    #print(LCSuff[0])
    

    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
    res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    drv.memcpy_htod(res_gpu, res)

    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)

    LCS(a_gpu,b_gpu,res_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(thread_num,1,1),grid=(thread_num,1,1))
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    drv.memcpy_dtoh(res, res_gpu)


    print ("input 1 ", a)
    print ("input 2 ", b)

    #print(LCSuff)
    
    #print("cpu res substr ", LCS(a, b, len(a), len(b)))
    print("res substr", res)
    return res[0] 


def GPUeditDistDP(output_dataset, output_spike_train, max_row_width, max_col_width, len_dataset, output_dataset_lengths, output_rssnp_lengths):
    mod = SourceModule("""
    #include <stdlib.h>
    __device__ int min1(int a,int b){

        if(a<b){
            return a;
        }
        else{
            return b;
        }
    }

    __global__ void edit_distDP(int* result_mat_gpu, int *dataset_gpu, int *output_gpu, int row_width, int col_width, int len_dataset, int *LCSuff, int *output_dataset_lengths, int *output_rssnp_lengths){
       const int z = threadIdx.x + blockDim.x * blockIdx.x;
       if (z < len_dataset) {
           //int max_val = 0;
           //printf("on thread %d i constrained by %d j constrained by %d", z, output_rssnp_lengths[z], output_dataset_lengths[z]);
           //printf("with content %d", result_mat_gpu[z]);
           //printf("row width is %d col width is %d", row_width, col_width);
           int j_constraint = 8;//output_dataset_lengths[z];
           int i_constraint = 9;//output_rssnp_lengths[z];
           int* max_val = 0;
           for (int j = 0; j < j_constraint; j++) {
                
                for (int i = 0; i <  i_constraint; i++){
                    //printf("computed value is %d", (z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + i*row_width + i));
                    int* LCSuff_base = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + i*row_width + i)];
                    __syncthreads();
                    if (i == 0){
                        *LCSuff_base = j;
                        //printf("A %d B %d LC %d\\n", i,j,*LCSuff_base);          
                        __syncthreads();
                    }
                    else if (j == 0){
                        *LCSuff_base = i;
                        //printf("A %d B %d LC %d\\n", i,j,*LCSuff_base);
                        __syncthreads();
                    }

                    else{
                        int delt = 1;
                        if (dataset_gpu[z * row_width + (i-1)] == output_gpu[z * col_width + (j-1)]) {  
                            delt = 0;
                        }
                        int* LCSuff_col_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + i*row_width + i)];
                        int* LCSuff_row_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + (i-1)*row_width + i-1)];
                        int* LCSuff_both_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + (i-1)*row_width + i-1)];
                        //printf(" gpu %d %d %d with %d and %d compared\\n", * LCSuff_col_decrem, *LCSuff_row_decrem, * LCSuff_both_decrem, dataset_gpu[z * row_width + (i-1)], output_gpu[z * col_width + (j-1)]);
                        *LCSuff_base = min1(min1(*LCSuff_col_decrem + 1, *LCSuff_row_decrem), *LCSuff_both_decrem + delt);
                        max_val = LCSuff_base;
                        __syncthreads();
                    }
                       
                }
            }
            if (max_val != 0) {
                //result_mat_gpu[z] = LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j_constraint*len_dataset*row_width + i_constraint*row_width + i_constraint)];
                result_mat_gpu[z] = *max_val;
            }
        } 
    }


    """)
    ED = mod.get_function("edit_distDP")
    print("legendary shape is ", output_spike_train.shape, " with content ", output_dataset)
    a = numpy.ndarray(shape= output_dataset.shape, buffer =  output_dataset, dtype=numpy.int32) 
    b = numpy.ndarray(shape= output_spike_train.shape, buffer =  output_spike_train, dtype=numpy.int32) 
    #a = numpy.array(output_dataset, dtype=numpy.int32)
    #b = numpy.array(output_spike_train, dtype=numpy.int32)
    print("a strides and at index 0", a.strides, a[0].strides)
    print("b strides", b.strides)

    print("a size is ", a.size, " while b size is ", b.size)


    LCSuff = numpy.zeros((len_dataset, a.size+1,b.size+1),dtype=numpy.int32)
    print("LCSuff shape is ", LCSuff.shape)
    row_width = numpy.int32(max_row_width)
    col_width = numpy.int32(max_col_width)
    result_mat = numpy.zeros((len_dataset),dtype=numpy.int32)
    #LCSuff = LCSuff.flatten()

        #print("LCSuff line is ", LCSuff[z][0])
    #print("LCSuff orig is ", LCSuff)

    #c = numpy.ndarray(shape= output_dataset_lengths.shape, buffer =  output_dataset_lengths, dtype=numpy.int32) 
    #d = numpy.ndarray(shape= output_rssnp_lengths.shape, buffer =  output_rssnp_lengths, dtype=numpy.int32) 
    c = pg.get(output_dataset_lengths)
    d = pg.get(output_rssnp_lengths)

    #print("c and d are magically ", c, " and ",  d, " with type ", type(c))
    #inout_pairs_view_gpu = drv.mem_alloc(inout_pairs_view.size * inout_pairs_view.dtype.itemsize)
    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    result_mat_gpu = drv.mem_alloc(result_mat.size * result_mat.dtype.itemsize)
    print("LCSuff size is ", LCSuff.size)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
  
    #func(drv.In(a), drv.InOut(e), np.int32(N), block=block_size, grid=grid_size)
    #c_gpu = drv.mem_alloc(c.size * c.dtype.itemsize)
    #d_gpu = drv.mem_alloc(d.size * d.dtype.itemsize)
    c_gpu = drv.mem_alloc(c.nbytes)
    d_gpu = drv.mem_alloc(d.nbytes)
    #c_gpu = gpuarray.to_gpu(output_dataset_lengths)
    #d_gpu = gpuarray.to_gpu(output_rssnp_lengths)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    drv.memcpy_htod(c_gpu, c)
    drv.memcpy_htod(d_gpu, d)
    #print("FINALLY entered with values ", c, " and ", d)
    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)

    ED(result_mat_gpu, a_gpu,b_gpu, numpy.int32(row_width), numpy.int32(col_width), numpy.int32(len_dataset), LCSuff_gpu, c_gpu, d_gpu, block=(thread_num,1,1),grid=(thread_num,1,1))

  
    drv.memcpy_dtoh(d, d_gpu)
    drv.memcpy_dtoh(c, c_gpu)
    drv.memcpy_dtoh(result_mat, result_mat_gpu)
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    print("result mat is ", result_mat)
    print("FINALLY 2 entered with values ", c, " and ", d)
   
    sum = 0
    for index in range(len(result_mat)):
        sum += (result_mat[index]/output_dataset_lengths[index]) * 100
    return sum

def GPUeditDistDP2(output_dataset, output_spike_train):
    mod = SourceModule("""
    #include <stdlib.h>
    __device__ int min1(int a,int b){

        if(a<b){
            return a;
        }
        else{
            return b;
        }
    }

    __global__ void edit_distDP(int j,int *X,int *Y, int *LCSuff,int row_width,int col_width){
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < row_width){
                if (i == 0){
                    LCSuff[i*col_width+j] = j;
                    __syncthreads();
                }
                else if (j == 0){

                    LCSuff[i*col_width+j] = i;
                    //printf("A %d B %d LC %d\\n", i,j,LCSuff[i*col_width+j]);
                    __syncthreads();
                }
                else if (X[i-1] == Y[j-1]) {  
                    LCSuff[i*col_width+j] = LCSuff[((i-1)*col_width)+j-1];
                    //printf("true");
                    __syncthreads();
                    //printf("compare %d %d\\n",res[0],LCSuff[i*col_width+j]);
                    //printf("res %d %d %d\\n",res[0],i,j);
                    
                } 
                else{
                    //printf("%d %d %d\\n",LCSuff[((i-1)*col_width)+j],LCSuff[i*col_width+j-1],LCSuff[((i-1)*col_width)+j-1]);
                    LCSuff[i*col_width+j] = 1+min1(min1(LCSuff[((i-1)*col_width)+j],LCSuff[i*col_width+j-1]),LCSuff[((i-1)*col_width)+j-1]);
                    __syncthreads();
                }

        } 
    }


    """)
    ED = mod.get_function("edit_distDP")

    #a = numpy.array([1,1,1,1,1],dtype=numpy.int32) #row the width
    #b = numpy.array([0,0,0,0,0],dtype=numpy.int32) #col
    a = numpy.array(output_spike_train,dtype=numpy.int32)
    b = numpy.array(output_dataset,dtype=numpy.int32)
    res = numpy.array([0],dtype=numpy.int32)
    LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)


    for i in range(b.size+1):
        LCSuff[0][i]=i
    for i in range(a.size+1):
        LCSuff[i][0]=i
    # print(LCSuff)

    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
    res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    drv.memcpy_htod(res_gpu, res)

    for j in range(b.size+1):
        print("at index ", j)
        ED(numpy.int32(j),a_gpu,b_gpu,LCSuff_gpu,numpy.int32(a.size+1),numpy.int32(b.size+1) , block=(10,10,1),grid=(1,1,1))


    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    drv.memcpy_dtoh(res, res_gpu)
    #print(LCSuff)
    print(LCSuff[a.size][b.size])
    #print(res)
    #print("res editdist", res)
    #print("CPU", editDistDP(a, b, len(a), len(b))) 

    return (LCSuff[a.size][b.size])

def getrandom(parents):
    n = len(parents)
    np_list = np.zeros(n, dtype=np.int32) 
    #np_list = numpy.random.randn(n,2).astype('int32')
    for z in range(n):
        maxi = len(parents[z]['system'].rule)
        np_list[z] = maxi
    #a_gpu = gpuarray.to_gpu(np_list)
    a_gpu = cuda.mem_alloc(np_list.size * np_list.dtype.itemsize)
    cuda.memcpy_htod(a_gpu, np_list)
    return a_gpu

def getrandom2(parents):
    n = len(parents)
    np_list = [[0 for x in range(n)] for y in range(2)] 
    #np_list = numpy.random.randn(n,2).astype('int32')
    for z in range(n):
        mini = 0
        maxi = len(parents[z].rule)
        np_list[z][0] = numpy.random.randint(mini, maxi)
        np_list[z][1] = numpy.random.randint(mini, maxi)
    a_gpu = gpuarray.to_gpu(np_list)
    return a_gpu


mod_cross = SourceModule("""
    #include <stdio.h>
    #include <curand.h>
    #include <curand_kernel.h>
    #include <math.h>
    #include <assert.h>
    #define MIN 2
    #define MAX 7
    #define ITER 10000000
 
    //1d grid of 2d blocks of size 4 x 4 blocks in a grid and 20 threads in both x and y dimension 
    __global__ void get_every_poss_of_ruleswap2(int* res, int* random_rules) {
        int tidx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        int size_tuple = 4;

        if ((blockIdx.x != blockIdx.y) && (threadIdx.x < random_rules[blockIdx.x]) && (threadIdx.y < random_rules[blockDim.x + blockIdx.y])) {
            printf("dimensions are %d %d %d %d", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
            res[tidx * size_tuple] = blockIdx.x;
            res[tidx * size_tuple + 1] = blockIdx.y;
            res[tidx * size_tuple + 2] = threadIdx.x;
            res[tidx * size_tuple + 3] = threadIdx.y;
            printf("random rules are %d and %d", random_rules[res[tidx * size_tuple + 2]],  random_rules[res[tidx * size_tuple + 3]]);
            //printf("passed here with parents %d %d", res[tidx * size_tuple], res[tidx * size_tuple + 1]);            
            //printf("passed here with rules %d %d", res[tidx * size_tuple + 2], res[tidx * size_tuple + 3]);
     
        }
       

    }

    //1d grid of 2d blocks of size 4 x 4 blocks in a grid and 20 threads in both x and y dimension  int tidz = threadIdx.z + blockIdx.z * blockDim.z;
    __global__ void get_every_poss_of_ruleswap(int* random_rules_limit, int* random_gpu) {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;;
        int tidy = threadIdx.y + blockIdx.y * blockDim.y;
        
        //int id = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


        //printf("gpu: x = %d, y = %d", tidx, tidy);

        if (blockIdx.x != blockIdx.y) {
            //printf("block: x = %d, block y = %d", blockIdx.x, blockIdx.y); 
            if (threadIdx.x < random_rules_limit[blockIdx.x]) {
                random_gpu[blockIdx.x] = threadIdx.x;
            }

            if (threadIdx.y < random_rules_limit[blockIdx.y]) {
                random_gpu[blockDim.x + blockIdx.y] = threadIdx.y;
            }

        }
       

    }


    """)

def crossover_gpu_defined(parents_size, parents, prev_offspring):
    max_rules = 20
    print("num parents is ", parents_size, " while max rules is ", max_rules)
    random_init_list = np.zeros(parents_size*2,dtype=np.int32)
    #random_gpu = gpuarray.to_gpu(random_init_list) 
    random_rule_parents_limit = getrandom(parents)
    #res = np.zeros((parents_size * parents_size, max_rules * max_rules),dtype=np.int32)
    random_gpu = cuda.mem_alloc(random_init_list.size * random_init_list.dtype.itemsize)
    cuda.memcpy_htod(random_gpu, random_init_list)
    cross = mod_cross.get_function("get_every_poss_of_ruleswap")
    cross(random_rule_parents_limit, random_gpu, block=(max_rules, max_rules,1), grid=(parents_size, parents_size,1))
    cuda.memcpy_dtoh(random_init_list, random_gpu)
    #cuda.memcpy_dtoh(res, res_gpu)
    
    print("res in crossover is ", random_init_list)
    return random_init_list

def program_main():
	print("Menu\n Enter a number of your choice: (1) Create a new evolutionary process w/ autosave\n (2) Load an evolutionary process from a configuration file: ")
	menu_choice = int(input("What would you like to do in this program: "))
	home = os.getcwd()
	if  menu_choice == 1:
		print("Given the choices of logic gates (1) AND, (2) OR, (3) NOT, (4) ADD, (5) SUB")
		answer = int(input("What kind of system would you like to evolve: "))
		print("Given the choices of version of the systems to evolve (1) minimal, (2) adversarial, (3) extra rules, (4) random, (5) user-defined")
		type_answer = int(input("What kind of system would you like to evolve: "))
		save_directory = os.path.join(home, "load_directory")
		#save_directory = input("Which directory would you like to save (.yaml) the evolutionary process: ")
		#if not os.path.exists(save_directory):
		  #  os.makedirs(save_directory)
		savefile_name = os.path.join(save_directory, input("What will be the name of this savefile: ") + ".yaml")
		runs = int(input("How many runs would you like to do (min of 1): "))
		generations = int(input("How many generation would you like to do (min of 1): "))
		population_size = int(input("How many RSSNP should be in the population? "))
		mutation_rate = int((input("How likely should it mutate in percentage? "))) * 100
		print("Of the Parent Selection methods:\n 0. **Top 50% of the population**\n1. **25% of the population based on fitness**\n2. **Top 25% + 25% of the population based on fitness**")
		selection_func = int(input("Which would you use?"))
		print("Of the Fitness Selection methods:\n 0. Longest Common Subsequence\n1. Longest Common Substring\n2. Edit Distance Method")
		fitness_func = int(input("Which would you use?"))

		ga_params = create_empty_yaml(runs, generations, population_size, savefile_name)
		ga_params['runs_pending'] = 0
		ga_params['gens_pending'] = 0
		ga_params['populations_pending'] = 0
		ga_params['goal_fitness'] = 101
		for run in range(runs):
			ga_params['runs'][run]['max_fitness_in_run'] = 0
			ga_params['runs'][run]['population_size'] = population_size
			ga_params['runs'][run]['mutation_rate'] = mutation_rate
			ga_params['runs'][run]['selection_func'] = selection_func
			ga_params['runs'][run]['fitness_function'] = fitness_func



		#building the test_cases_path
		and_path = os.path.join(home, "test cases", "and_test_cases.txt")
		or_path = os.path.join(home, "test cases", "or_test_cases.txt")
		not_path = os.path.join(home, "test cases", "not_test_cases.txt")
		add_path = os.path.join(home, "test cases", "add_test_cases.txt")
		sub_path = os.path.join(home, "test cases", "sub_test_cases.txt")


		#Default or made or random system to initially evolve
		#for the different kind of evolutionary process (logic gates), we evolve
		system = None
		test_cases_path = None
		ga_params['test_cases_path'] = None

		if answer == 1:
			if type_answer == 1:
				system = and_rssnp_minimal
			if type_answer == 2:
				system = and_rssnp_adversarial
			if type_answer == 3:
				system = and_rssnp_extra_rules
			test_cases_path = and_path
			ga_params['test_cases_path'] = and_path

		if answer == 2:
			if type_answer == 1:
				system = or_rssnp_minimal
			if type_answer == 2:
				system = or_rssnp_adversarial
			if type_answer == 3:
				system = or_rssnp_extra_rules
			test_cases_path = or_path
			ga_params['test_cases_path'] = or_path

		if answer == 3:
			if type_answer == 1:
				system = not_rssnp_minimal
			if type_answer == 2:
				system = not_rssnp_adversarial
			if type_answer == 3:
				system = not_rssnp_extra_rules
			test_cases_path = not_path
			ga_params['test_cases_path'] = not_path

		if answer == 4:
			if type_answer == 1:
				system = add_rssnp_minimal
			if type_answer == 2:
				system = add_rssnp_adversarial
			if type_answer == 3:
				system = add_rssnp_extra_rules
			test_cases_path = add_path
			ga_params['test_cases_path'] = add_path

		if answer == 5:
			if type_answer == 1:
				system = sub_rssnp_minimal
			if type_answer == 2:
				system = sub_rssnp_adversarial
			if type_answer == 3:
				system = sub_rssnp_extra_rules
			test_cases_path = sub_path
			ga_params['test_cases_path'] = sub_path

		if type_answer == 4:
			system = set_bounds(ga_params, runs)

		if type_answer == 5:
			system = set_values(ga_params, runs)
			if system == None:
				print("Rssnp given is invalid")
				exit()

		conf_save(savefile_name, ga_params)
		if system == None or test_cases_path == None:
			print("Illegal system chosen or wrong direction for evolution")
		else:
			gaframework(system, test_cases_path, savefile_name)



	if int(menu_choice) == 2:
		load_directory = os.path.join(home, "load_directory")
		#load_directory = input("Which load directory would you like to load (.yaml) an evolutionary process from: ")
		loadfile_name = os.path.join(load_directory, input("What is the name of this loadfile (append a yaml extension) : "))
		ga_params = conf_load(loadfile_name)
		print("Load function starting")

		execution_choice = int(input("Where do you want your experiment to be executed (1) CPU or (2) GPU: "))
		start_from_a_gen = False
		ga_params['goal_fitness'] = 101
		if execution_choice == 1:

			#get the information from console
			print("Menu: Would you either increase (1) runs/generations/population_size or (2) a goal fitness of any chromosome in the evolutionary process: or (3) just maintain the current number of generation/population_size but extend runs further using an initial population from any generation number ")
			print("Note that for this program the GPU can also do option 3 but not the others ")
			sub_choice = int(input())
			if sub_choice == 1:
				add_runs = int(input("How many runs would you like to add (minimum of 1 and maximum of 100): "))
				add_gens = int(input("How many generations would you like to add (minimum of 0 and maximum of 100): "))
				add_populations = int(input("How many individuals would you like to add to the population (minimum of 0): "))
				ga_params['runs_pending'] = add_runs
				ga_params['gens_pending'] = add_gens
				ga_params['populations_pending'] = add_populations
				
			elif sub_choice == 2:
				print("Notice that the evolutionary process wont stop till a chromosome reached the desired goal fitness\n")
				print("Or till you exit the program itself (make sure its after a run in a GA) and the autosave will record only up to the point of termination\n")
				add_goal_fitness = int(input("What is the desired goal fitness: (minimum of 50 and maximum of 100): "))
				#101 denotes an unending condition no matter how many runs or generations are reported
				ga_params['runs_pending'] = 101
				ga_params['gens_pending'] = 101
				ga_params['populations_pending'] = 0
				ga_params['goal_fitness'] = add_goal_fitness
			elif sub_choice == 3:					
				print("Maintaining current number of generation and population_size")
				add_runs = int(input("How many runs would you like to add (minimum of 1 and maximum of 100): "))
				ga_params['gens_pending'] = 0
				ga_params['populations_pending'] = 0
				ga_params['runs_pending'] = add_runs
				ga_params['generation_index_continue'] = input("Which run and generation would you like to use as the starting parents of the succeeding runs (separate by comma)? ")
				start_from_a_gen = True

			
			newloadfile_name = prompt_make_newsavefile(ga_params, loadfile_name, load_directory)
			continue_create_empty_yaml(newloadfile_name)
			ga_params = conf_load(newloadfile_name)
			run_total = ga_params['run_total'] + ga_params['runs_pending']
			for run in range(ga_params['run_total'], run_total):
				ga_params['runs'][run]['max_fitness_in_run'] = 0
				ga_params['runs'][run]['population_size'] = ga_params['runs'][0]['population_size'] + ga_params['populations_pending']
				ga_params['runs'][run]['mutation_rate'] = ga_params['runs'][0]['mutation_rate']
				ga_params['runs'][run]['selection_func'] = ga_params['runs'][0]['selection_func']
				ga_params['runs'][run]['fitness_function'] = ga_params['runs'][0]['fitness_function']
			conf_save(newloadfile_name, ga_params)
			
			start_new = False

			print("Running GA in CPU: ")
			#The system that will be parent0 is the first rssnp chromosome in the ga_params
			#change the value of this into a system in RSSNP_list in order to change it
			#for subchoice 3, rssnp string is used only in order to acquire the input neuron indexes for spike_train_parser function
			rssnp_string = ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0]
			
			print("rssnp string is " + str(rssnp_string))
			gaframework(rssnp_string, ga_params['test_cases_path'] , newloadfile_name, start_new, start_from_a_gen)
			ga_params = conf_load(newloadfile_name)
			ga_params['run_total'] += ga_params['runs_pending']
			ga_params['runs_pending'] = 0
			ga_params['gen_total'] += ga_params['gens_pending']
			ga_params['gens_pending'] = 0
			ga_params['populations_pending'] = 0

			conf_save(newloadfile_name, ga_params)
		
		else:
			print("Running GA in GPU: ")
			
			print("Maintaining current number of generation and population_size")
			add_runs = int(input("How many runs would you like to add (minimum of 1 and maximum of 100): "))	
			ga_params['gens_pending'] = 0
			ga_params['populations_pending'] = 0
			ga_params['runs_pending'] = add_runs
			ga_params['generation_index_continue'] = input("Which run and generation would you like to use as the starting parents of the succeeding runs (separate by comma)? ")
			newloadfile_name =  prompt_make_newsavefile(ga_params, loadfile_name, load_directory)
			print("rewriting to 1", newloadfile_name)
			continue_create_empty_yaml(newloadfile_name)
			ga_params = conf_load(newloadfile_name)
			run_total = ga_params['run_total'] + ga_params['runs_pending']
			for run in range(ga_params['run_total'], run_total):
				ga_params['runs'][run]['max_fitness_in_run'] = 0
				ga_params['runs'][run]['population_size'] = ga_params['runs'][0]['population_size'] + ga_params['populations_pending']
				ga_params['runs'][run]['mutation_rate'] = ga_params['runs'][0]['mutation_rate']
				ga_params['runs'][run]['selection_func'] = ga_params['runs'][0]['selection_func']
				ga_params['runs'][run]['fitness_function'] = ga_params['runs'][0]['fitness_function']
			print("rewriting to 2", newloadfile_name)
			conf_save(newloadfile_name, ga_params)
			#rssnp_string = ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0]
			gaframework_gpu(newloadfile_name)
			ga_params = conf_load(newloadfile_name)
			ga_params['run_total'] += ga_params['runs_pending']
			ga_params['runs_pending'] = 0
			ga_params['gen_total'] += ga_params['gens_pending']
			ga_params['gens_pending'] = 0
			ga_params['populations_pending'] = 0

			print("rewriting to 3", newloadfile_name)
			conf_save(newloadfile_name, ga_params)




print("lol")
program_main()
