from src.abstracts.rssnp import assign_rssnp
from src.abstracts.grapher import draw

# BITONIC
bitonic_rssnp_minimal = assign_rssnp({
    'neurons': 28,
    'synapses': 42,
    'rules': [
      [0, 4, (0, 1), 1, 1, 0], 
      [0, 5, (0, 1), 1, 1, 0], 
      [1, 4, (0, 1), 1, 1, 0], 
      [1, 5, (0, 1), 1, 1, 0],
      [2, 6, (0, 1), 1, 1, 0], 
      [2, 7, (0, 1), 1, 1, 0], 
      [3, 6, (0, 1), 1, 1, 0], 
      [3, 7, (0, 1), 1, 1, 0],  
      [4, 8, (2, 0), 2, 1, 0],
      [4, 9, (2, 0), 2, 1, 0],
      [4, 8, (1, 0), 1, 0, 0],
      [4, 9, (1, 0), 1, 0, 0],
      [5, 9, (2, 0), 2, 0, 0 ],
      [5, 9, (1, 0), 1, 1, 0 ],
      [6, 10, (2, 0), 2, 1, 0],
      [6, 11, (2, 0), 2, 1, 0],
      [6, 10, (1, 0), 1, 0, 0],
      [6, 11, (1, 0), 1, 0, 0],
      [7, 10, (2, 0), 2, 0, 0 ],
      [7, 10, (1, 0), 1, 1, 0 ],
      [8, 12, (0, 1), 1, 1, 0], 
      [8, 13, (0, 1), 1, 1, 0], 
      [9, 12, (0, 1), 1, 1, 0], 
      [9, 13, (0, 1), 1, 1, 0],
      [10, 14, (0, 1), 1, 1, 0], 
      [10, 15, (0, 1), 1, 1, 0],
      [11, 14, (0, 1), 1, 1, 0], 
      [11, 15, (0, 1), 1, 1, 0],  
      [12, 16, (2, 0), 2, 1, 0],
      [12, 17, (2, 0), 2, 1, 0],
      [12, 16, (1, 0), 1, 0, 0],
      [12, 17, (1, 0), 1, 0, 0],
      [13, 17, (2, 0), 2, 0, 0],
      [13, 17, (1, 0), 1, 1, 0],
      [14, 18, (2, 0), 2, 1, 0],
      [14, 19, (2, 0), 2, 1, 0],
      [14, 18, (1, 0), 1, 0, 0],
      [14, 19, (1, 0), 1, 0, 0],
      [15, 19, (2, 0), 2, 0, 0],
      [15, 19, (1, 0), 1, 1, 0],
      [16, 20, (0, 1), 1, 1, 0], 
      [16, 21, (0, 1), 1, 1, 0], 
      [18, 20, (0, 1), 1, 1, 0], 
      [18, 21, (0, 1), 1, 1, 0],
      [17, 22, (0, 1), 1, 1, 0], 
      [17, 23, (0, 1), 1, 1, 0], 
      [19, 22, (0, 1), 1, 1, 0], 
      [19, 23, (0, 1), 1, 1, 0],  
      [20, 24, (2, 0), 2, 1, 0],
      [20, 26, (2, 0), 2, 1, 0],
      [20, 24, (1, 0), 1, 0, 0],
      [20, 26, (1, 0), 1, 0, 0],
      [21, 26, (2, 0), 2, 0, 0],
      [21, 26, (1, 0), 1, 1, 0],
      [22, 25, (2, 0), 2, 1, 0],
      [22, 27, (2, 0), 2, 1, 0],
      [22, 25, (1, 0), 1, 0, 0],
      [22, 27, (1, 0), 1, 0, 0],
      [23, 27, (2, 0), 2, 0, 0],
      [23, 27, (1, 0), 1, 1, 0]
    ],
    'init_config': [0 for x in range(28)],
    'rule_status': [-1 for x in range(60)],
    'input_neurons': [0, 1, 2, 3],
    'output_neurons': [24, 25, 26, 27]
})

bitonic_rssnp_minimal.in_spiketrain = [
    {
        'index': 0,
        'input': [1, 1, 1, 1]
    },
    {
        'index': 1,
        'input': [1, 1, 1]
    },

    {
        'index': 2,
        'input': [1, 1]
    },
    {
        'index': 3,
        'input': [1]
    }
    
]

bitonic_rssnp_minimal.out_spiketrain = [
    {
        'index': 24,
        'output': []
    },
    {
        'index': 25,
        'output': []
    },

    {
        'index': 26,
        'output': []
    },
    {
        'index': 27,
        'output': []
    }
    
]

system_state = (bitonic_rssnp_minimal.configuration_init, bitonic_rssnp_minimal.ruleStatus) # the initial configuration
maxSteps = 9 # the max number of steps before halting the simulation (this is just a safety net)
print(bitonic_rssnp_minimal.main(system_state, maxSteps))

'''
 0. The index of the source neuron
    1. The index of the destination neuron
    2. The regular expression that activates the rule a^i(a^j)*
    3. The number of spikes consumed
    4. The number of spikes produced
    5. The delay of firing the spikes to other neurons
'''