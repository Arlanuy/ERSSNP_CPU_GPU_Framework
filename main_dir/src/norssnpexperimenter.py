# This is the main script for running experiments - use CAREFULLY!!

from abstracts.geneticalgorithm import SNPGeneticAlgo
from abstracts.GAevaluator import SNPGeneticAlgoEval
from abstracts.parsers import spike_train_parser
from abstracts.norssnp import create_rssnp

import os

def execute_experiment(results_dir, rssnp, ga, gaeval, stats, experiment_name):
    gaeval.no_of_gen = stats['generations']
    gaeval.no_of_run = stats['runs']
    gaeval.opt_fitness = 50
    gaeval.max_fitness = 100
    gaeval.list_of_runs = []
    print("Now executing", experiment_name)
    gaeval.run(ga, rssnp, stats['population_size'], stats['fitness_function'], gaeval.no_of_gen, stats['mutation_rate'], dir, experiment_name, stats['selection_func'])

results_directory = input("Where would you like to put the results: ")

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

print("Starting...")
print("Setting up RSSNPs...")

# Create the rssnps
and_rssnp = create_rssnp(10, 2, 20, 20, 3, 3, 2, 2, inputs=[0, 1], output=2, fixed=True)
or_rssnp = create_rssnp(10, 2, 20, 20, 3, 3, 2, 2, inputs=[0, 1], output=2, fixed=True)
not_rssnp = create_rssnp(10, 2, 20, 20, 3, 3, 2, 2, inputs=[0], output=1, fixed=True)
add_rssnp = create_rssnp(10, 2, 20, 20, 3, 3, 2, 2, inputs=[0, 1], output=2, fixed=True)
# sub_rssnp = create_rssnp(20, 2, 40, 40, 6, 6, 3, 3, inputs=[0, 1], output=2, fixed=True)

print("Setting up GA frameworks...")
# Assign genetic algorithm framework
and_ga = SNPGeneticAlgo()
or_ga = SNPGeneticAlgo()
not_ga = SNPGeneticAlgo()
add_ga = SNPGeneticAlgo()
# sub_ga = SNPGeneticAlgo()

print("Prepping up GA evaluators...")
# Assign evaluators
and_gaeval = SNPGeneticAlgoEval()
or_gaeval = SNPGeneticAlgoEval()
not_gaeval = SNPGeneticAlgoEval()
add_gaeval = SNPGeneticAlgoEval()
# sub_gaeval = SNPGeneticAlgoEval()

print("Setting up spike trains for simulations...")
# Set spike trains to each GA framework
and_ga.inout_pairs = spike_train_parser('test cases/and_test_cases.txt',and_rssnp.inputs)
or_ga.inout_pairs = spike_train_parser('test cases/or_test_cases.txt',or_rssnp.inputs)
not_ga.inout_pairs = spike_train_parser('test cases/not_test_cases.txt',not_rssnp.inputs)
add_ga.inout_pairs = spike_train_parser('test cases/add_test_cases.txt',add_rssnp.inputs)
# sub_ga.inout_pairs = spike_train_parser('test cases/sub_test_cases.txt',sub_rssnp.inputs)

experiment_list = [
    
    {'population_size': 12, 'mutation_rate': 4, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 0},
    {'population_size': 12, 'mutation_rate': 4, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 0},

    {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
    {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},

    {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 2},
    {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 2},
]

print("Starting experiments...")
ctr = 1 # Experiment counter
# Run stats for each GA framework
for stats in experiment_list:
    dir = results_directory + "/Experiment" + str(ctr)
    ctr += 1
    if not os.path.exists(dir):
        os.makedirs(dir)
    execute_experiment(dir, and_rssnp, and_ga, and_gaeval, stats, "And")
    execute_experiment(dir, or_rssnp, or_ga, or_gaeval, stats, "Or")
    execute_experiment(dir, not_rssnp, not_ga, not_gaeval, stats, "Not")
    execute_experiment(dir, add_rssnp, add_ga, add_gaeval, stats, "Add")
    # execute_experiment(dir, sub_rssnp, sub_ga, sub_gaeval, stats, "Sub")

