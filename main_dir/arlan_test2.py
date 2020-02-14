from src.experimenter import gaframework
import os
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
