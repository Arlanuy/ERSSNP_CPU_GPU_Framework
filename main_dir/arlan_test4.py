from src.experimenter import gaframework
import os
sub_rssnp_extra_rules = {
    'neurons': 6,
    'synapses': 7,
    'rules': [
      [0, 2, (0, 1), 1, 1, 0]
      [0, 3, (0, 1), 1, 1, 0]
      [1, 2, (0, 1), 1, 1, 0]
      [1, 3, (0, 1), 1, 1, 0] 
	  [2, 4, (2, 0), 2, 1, 0]
	  [2, 5, (2, 0), 2, 1, 0]
	  [2, 4, (1, 0), 1, 0, 0]
	  [2, 5, (1, 0), 1, 0, 0]
	  [3, 5, (2, 0), 2, 0, 0 ]
	  [3, 5, (1, 0), 1, 1, 0 ]
]
    ],
    'init_config': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'rule_status': [-1 for x in range(10)],
    'input_neurons': [0, 1],
    'output_neuron': [4, 5]
}
ga_params = {
    'population_size': 12,
    'mutation_rate': 100,
    'fitness_function': 0,
    'generations': 50,
    'runs': 1,
    'selection_func': 1
}
home = os.getcwd()
path = os.path.join(home, "test_cases", "sub_test_cases.txt")
print(path)
path_to_inputoutput_spike_trains = path

gaframework(sub_rssnp_extra_rules, path_to_inputoutput_spike_trains, ga_params)
8


