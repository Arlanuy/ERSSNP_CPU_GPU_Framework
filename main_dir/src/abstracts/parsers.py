def spike_train_parser(filename, input_list):
    """
    Inserts a text file with comma-separated bitstrings into a dictionary with the following format:

        {
            inputs: [
                {
                    'index': (int),
                    'input': [ (list of 0 or 1)]
                },
            ],
            'output' : [(list of 0 or 1)]
        }
    """

    try:
        file = open(filename, 'r')
    except(Exception):
        print("File not found!")
        exit()
    try:
        inout_pairs = []
        for line in file:
            bitstrings = line.split(',')
            pair = dict()
            pair['inputs'] = []

            # start counter for changing the input neuron to be considered
            input_index = 0

            # for each bitstring except the last, add it to the dictionary
            for bitstring in bitstrings[:-1]:
                pair['inputs'].append({
                    'index': input_list[input_index],
                    'input': [int(x) for x in list(bitstring)]
                })
                input_index += 1
            
            # then add the output
            pair['output'] = [int(x) for x in list(bitstrings[-1])[:-1]] if (bitstrings[-1][-1] == '\n') else [int(x) for x in list(bitstrings[-1])]

            # then add the pair to the list of pairs
            inout_pairs.append(pair)
        #print("inout pairs is " + str(inout_pairs))

        file.close()
        return inout_pairs
    except(Exception):
        print("Invalid format. Cannot parse.")
        exit()