from src.abstracts.grapher import draw_from_string

path = input("Enter path to desired run (Example: results/Experiment1/Func/Run0): ")

try:
    with open(path+"/run.txt") as file:
        gen = input("Select generation: ")
        buffer = file.readlines()
        line_index = buffer.index("Generation " + gen + "\n")
        interesting_string = buffer[line_index + 1]
        start = interesting_string.index(': ') + 2
        end = interesting_string.index(", '")
        actual_string = interesting_string[start:end]
        index = input("Input neuron indexes: ")
        input_neurons = [int(x) for x in index.split(",")]
        output_neuron = int(input("Output neuron index: "))
        filename = input("Enter a filename: ")
        draw_from_string(actual_string, input_neurons, output_neuron, filename, view=True)
except:
    print("Error")