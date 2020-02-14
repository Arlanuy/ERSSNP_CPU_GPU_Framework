from src.abstracts.rssnp import assign_rssnp
from src.abstracts.grapher import draw

# ADD
add_rssnp_minimal = {
    'neurons': 4,
    'synapses': 4,
    'rules': [
        [0, 1, (2, 0), 1, 1, 0],
        [0, 2, (2, 0), 2, 1, 0],
        [1, 3, (1, 0), 1, 1, 0],
        [2, 3, (1, 0), 1, 1, 0], 
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(5)],
    'input_neurons': [0],
    'output_neuron': 3
}

rssnp = assign_rssnp(add_rssnp_minimal)
draw(rssnp, "my_add_rssnp")