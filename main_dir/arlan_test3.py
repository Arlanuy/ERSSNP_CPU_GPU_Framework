from src.abstracts.rssnp import assign_rssnp
and_rssnp = assign_rssnp({
    'neurons': 4,
    'synapses': 4,
    'rules': [
        [0, 1, (2, 0), 1, 1, 0],
        [0, 2, (2, 0), 2, 1, 0],
        [1, 3, (1, 0), 1, 1, 0],
        [2, 3, (1, 0), 1, 1, 0], 
    ],
    'init_config': [0, 0, 0, 0],
    'rule_status': [-1 for x in range(4)],
    'input_neurons': [0],
    'output_neuron': 3
})

# The input spike train that will be fed to the input neurons
and_rssnp.in_spiketrain = [
    {
        'index': 0,
        'input': [2]
    },
    
]

system_state = (and_rssnp.configuration_init, and_rssnp.ruleStatus) # the initial configuration
maxSteps = 90 # the max number of steps before halting the simulation (this is just a safety net)
print(and_rssnp.main(system_state, maxSteps))