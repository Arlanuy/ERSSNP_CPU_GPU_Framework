from src.experimenter import gaframework

initial_rssnp = {
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

ga_params = {
    'population_size': 12,
    'mutation_rate': 100,
    'fitness_function': 1,
    'generations': 5,
    'runs': 1,
    'selection_func': 1
}

path_to_inputoutput_spike_trains = 'test cases/and_test_cases.txt'

gaframework(initial_rssnp, path_to_inputoutput_spike_trains, ga_params)