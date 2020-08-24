from .grapher import draw
from .rssnp import assign_rssnp
import random

def create_rssnp_dict(neurons, init_spikes, synapses, rules, i, j, consumed, produced, inputs=[], output=None, fixed=False):
    """
    Creates an RSSNP with the given upper bounds
    Setting fixed=True will make the largest possible RSSNP

    IMPORTANT!!!
    Using this method is very unreliable right now, it will probably create an invalid RSSNP
    Need an algo to fix this
    """

    # Check if inputs can be accommodated by bound of neurons
    for neuron in inputs:
        if neuron >= neurons:
            print("Input neurons cannot be supported!")
            exit()
    
    # Check if output can be accommodated by bound of neurons
    if output >= neurons:
        print("Output neuron cannot be supported!")
        exit()

    # Set the bounds
    if fixed:
        desired_rssnp = {
            'neurons': neurons,
            'synapses': synapses,
            'rules': [],
            'rule_status': [-1 for x in range(rules)],
        }
    else:
        desired_rssnp = {
            'neurons': random.randint(1, neurons-len(inputs)) + len(inputs),
            'synapses': random.randint(1, synapses),
            'rules': [],
            'rule_status': [-1 for x in range(random.randint(1, rules),)]
        }

    # Set initial configuration
    desired_rssnp['init_config'] = [random.randint(0, init_spikes) for x in range(desired_rssnp['neurons'])]

    synapse_list = []
    # Create rules
    for _ in range(0, len(desired_rssnp['rule_status'])):
        # Select two neurons
        source = random.randint(0, desired_rssnp['neurons']-1)
        destination = random.randint(0, desired_rssnp['neurons']-1)
        while source == destination:    # can't have loops
            source = random.randint(0, desired_rssnp['neurons']-1)
            destination = random.randint(0, desired_rssnp['neurons']-1)

        if not (source,destination) in synapse_list and len(synapse_list) < desired_rssnp['synapses']:    # Record this synapse
            synapse_list.append((source,destination))
        elif not (source,destination) in synapse_list:             # Max synapses reached; select a random synapse from the list
            source, destination = synapse_list[random.randint(0, len(synapse_list)-1)]

        # Adding rule here
        desired_rssnp['rules'].append(
            [source, destination, (random.randint(1, i), random.randint(0, j)), random.randint(1, consumed), random.randint(0, produced), 0]
        )
    
    # Set input and output neurons
    desired_rssnp['input_neurons'] = inputs
    desired_rssnp['output_neuron'] = output

    return desired_rssnp

def create_rssnp(neurons, init_spikes, synapses, rules, i, j, consumed, produced, inputs=[], output=None, fixed=False):
    exit_flag = False
    print("Creating RSSNP...")
    while not exit_flag:
        # Make that rssnp based on the user's specifications
        rssnp_dict = create_rssnp_dict(neurons, init_spikes, synapses, rules, i, j, consumed, produced, inputs=inputs, output=output, fixed=fixed)
        rssnp = assign_rssnp(rssnp_dict)

        exit_flag = rssnp.isValid()
    
    print("Success!")
    return rssnp


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


    



