from graphviz import Digraph
import ast

def draw(system, file_name, view=False):
    '''
    Draws a directed graph using the system and outputs a pdf
    '''
    graph = Digraph(file_name, file_name + ".gv", format='png')

    # Draw neurons
    for i in range(0, system.n):
        if i in system.inputs:
            graph.attr('node', shape='egg')
        elif i == system.outputs:
            graph.attr('node', shape='doublecircle')
        else:
            graph.attr('node', shape='circle')
        graph.node(str(i), "n"+str(i)+"\n"+str(system.configuration_init[i]))

    # Connect neurons
    synapses = dict()
    for i in range(0, system.m):
        if not (system.rule[i][0],system.rule[i][1]) in synapses:
            synapses[(system.rule[i][0],system.rule[i][1])] = ""
        synapses[(system.rule[i][0],system.rule[i][1])] += str(system.rule[i][2]) + '/' + str(system.rule[i][3]) + '->' + str(system.rule[i][4]) + "\n"
    
    for source, destination in synapses:
        graph.edge(
            str(source),
            str(destination),
            label=synapses[(source, destination)]
        )

    graph.render(view=view)


def draw_from_string(string, inputs, outputs, file_name, view=False):
    '''
    Draws a directed graph using a string depicting a system
    Cannot determine input and output neurons
    '''
    # Separate the rule and configuration
    rule, configuration = string.split(' | ')
    rule = ast.literal_eval(rule)
    configuration = ast.literal_eval(configuration)

    graph = Digraph(file_name, file_name + ".gv", format='png')
    # Draw neurons
    for i in range(0, len(configuration)):
        if i in inputs:
            graph.attr('node', shape='egg')
        elif i == outputs:
            graph.attr('node', shape='doublecircle')
        else:
            graph.attr('node', shape='circle')
        graph.node(str(i), "n"+str(i)+"\n"+str(configuration[i]))

    # Connect neurons
    synapses = dict()
    for i in range(0, len(rule)):
        if not (rule[i][0],rule[i][1]) in synapses:
            synapses[(rule[i][0],rule[i][1])] = ""
        synapses[(rule[i][0],rule[i][1])] += str(rule[i][2]) + '/' + str(rule[i][3]) + '->' + str(rule[i][4]) + "\n"
    
    for source, destination in synapses:
        graph.edge(
            str(source),
            str(destination),
            label=synapses[(source, destination)]
        )

    graph.render(view=view)