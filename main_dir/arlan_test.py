from src.abstracts.rssnp import assign_rssnp
from src.abstracts.grapher import draw

and_rssnp = assign_rssnp({
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
    'output_neurons': 3
})

and_rssnp.in_spiketrain = [
    {
        'index': 0,
        'input': [0,0,1,0,0,0,1,1,1]
    },
    {
        'index': 1,
        'input': [1,0,1,1,0,0,0,1,1]
    },
    
]

system_state = (and_rssnp.configuration_init, and_rssnp.ruleStatus) # the initial configuration
maxSteps = 9 # the max number of steps before halting the simulation (this is just a safety net)
print(and_rssnp.main(system_state, maxSteps))

'''
 0. The index of the source neuron
    1. The index of the destination neuron
    2. The regular expression that activates the rule a^i(a^j)*
    3. The number of spikes consumed
    4. The number of spikes produced
    5. The delay of firing the spikes to other neurons
'''