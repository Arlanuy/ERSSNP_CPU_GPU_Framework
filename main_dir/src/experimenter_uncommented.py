# This is the main script for running experiments - use CAREFULLY!!
#for CPU
from src.abstracts.geneticalgo_integ import SNPGeneticAlgo
from src.abstracts.GAevaluator_integ import SNPGeneticAlgoEval, conf_load
from src.abstracts.grapher import draw
from src.abstracts.parsers import spike_train_parser
from src.abstracts.rssnp import assign_rssnp

#for GPU
from src.abstracts.geneticalgo_integ_gpu import SNPGeneticAlgoGPU


#from RSSNP_list import *    # contains list of sample rssnps

import os

# stats = {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 5, 'runs': 1, 'selection_func': 1},
def gaframework(rssnp_string, path_to_io_spike_trains, loadfile_name, start_new = True, start_from_gen = False):
    ga = SNPGeneticAlgo()
    gaeval = SNPGeneticAlgoEval()
    ga_params = conf_load(loadfile_name)

    rssnp = assign_rssnp(rssnp_string)
    ga.inout_pairs = spike_train_parser(path_to_io_spike_trains,rssnp.inputs)

    execute_experiment(rssnp, ga, gaeval, ga_params, loadfile_name, start_new, start_from_gen)

def gaframework_gpu(loadfile_name):
    ga = SNPGeneticAlgoGPU()
    gaeval = SNPGeneticAlgoEval ()
    print("entering to 1", loadfile_name)
    ga_params = conf_load(loadfile_name)
    #ga.inout_pairs = spike_train_parser(path_to_io_spike_trains,rssnp.inputs)
    execute_experiment_gpu(ga, gaeval, ga_params, loadfile_name)

def execute_experiment(rssnp, ga, gaeval, stats, loadfile_name, start_new = True, start_from_gen = False):
    
    gaeval.no_of_gen = stats['gen_total']
    gaeval.no_of_run = stats['run_total']

    gaeval.opt_fitness = 50
    gaeval.max_fitness = 100
    gaeval.list_of_runs = []
    print("Now executing experiment involving", loadfile_name)
    if start_new == True:
        stats_run = stats['runs'][0]
    else:
        stats_run = stats['runs'][stats['run_total']]
    gaeval.run(ga, rssnp, stats_run['population_size'], stats_run['fitness_function'],  gaeval.no_of_run, gaeval.no_of_gen, stats_run['mutation_rate'], loadfile_name, stats_run['selection_func'], start_new, start_from_gen)

def execute_experiment_gpu(ga, gaeval, stats, loadfile_name):
    gaeval.no_of_gen = stats['gen_total']
    gaeval.no_of_run = stats['run_total']

    gaeval.opt_fitness = 50
    gaeval.max_fitness = 100
    gaeval.list_of_runs = []
    print("Now executing experiment involving", loadfile_name)
    stats_run = stats['runs'][gaeval.no_of_run]
    start_new = False
    start_from_gen = True
    rssnp = None
    gaeval.run(ga, rssnp, stats_run['population_size'], stats_run['fitness_function'],  gaeval.no_of_run, gaeval.no_of_gen, stats_run['mutation_rate'], loadfile_name, stats_run['selection_func'], start_new, start_from_gen)

def setup_sample():

    print("Setting up RSSNPs...")
    # Create the rssnps
    and1_rssnp = assign_rssnp(and_rssnp_minimal)
    and2_rssnp = assign_rssnp(and_rssnp_adversarial)
    and3_rssnp = assign_rssnp(and_rssnp_extra_rules)
    or1_rssnp = assign_rssnp(or_rssnp_minimal)
    or2_rssnp = assign_rssnp(or_rssnp_adversarial)
    or3_rssnp = assign_rssnp(or_rssnp_extra_rules)
    not1_rssnp = assign_rssnp(not_rssnp_minimal)
    not2_rssnp = assign_rssnp(not_rssnp_adversarial)
    not3_rssnp = assign_rssnp(not_rssnp_extra_rules)
    add1_rssnp = assign_rssnp(add_rssnp_minimal)
    add2_rssnp = assign_rssnp(add_rssnp_adversarial)
    add3_rssnp = assign_rssnp(add_rssnp_extra_rules)
    sub1_rssnp = assign_rssnp(sub_rssnp_minimal)
    sub2_rssnp = assign_rssnp(sub_rssnp_adversarial)
    sub3_rssnp = assign_rssnp(sub_rssnp_extra_rules)

    # draw(not2_rssnp, "test", view=True)

    print("Setting up GA frameworks...")
    # Assign genetic algorithm framework
    and1_ga = SNPGeneticAlgo()
    and2_ga = SNPGeneticAlgo()
    and3_ga = SNPGeneticAlgo()
    or1_ga = SNPGeneticAlgo()
    or2_ga = SNPGeneticAlgo()
    or3_ga = SNPGeneticAlgo()
    not1_ga = SNPGeneticAlgo()
    not2_ga = SNPGeneticAlgo()
    not3_ga = SNPGeneticAlgo()
    add1_ga = SNPGeneticAlgo()
    add2_ga = SNPGeneticAlgo()
    add3_ga = SNPGeneticAlgo()
    sub1_ga = SNPGeneticAlgo()
    sub2_ga = SNPGeneticAlgo()
    sub3_ga = SNPGeneticAlgo()

    print("Prepping up GA evaluators...")
    # Assign evaluators
    and1_gaeval = SNPGeneticAlgoEval()
    and2_gaeval = SNPGeneticAlgoEval()
    and3_gaeval = SNPGeneticAlgoEval()
    or1_gaeval = SNPGeneticAlgoEval()
    or2_gaeval = SNPGeneticAlgoEval()
    or3_gaeval = SNPGeneticAlgoEval()
    not1_gaeval = SNPGeneticAlgoEval()
    not2_gaeval = SNPGeneticAlgoEval()
    not3_gaeval = SNPGeneticAlgoEval()
    add1_gaeval = SNPGeneticAlgoEval()
    add2_gaeval = SNPGeneticAlgoEval()
    add3_gaeval = SNPGeneticAlgoEval()
    sub1_gaeval = SNPGeneticAlgoEval()
    sub2_gaeval = SNPGeneticAlgoEval()
    sub3_gaeval = SNPGeneticAlgoEval()

    print("Setting up spike trains for simulations...")
    # Set spike trains to each GA framework
    and1_ga.inout_pairs = spike_train_parser('test cases/and_test_cases.txt',and1_rssnp.inputs)
    and2_ga.inout_pairs = spike_train_parser('test cases/and_test_cases.txt',and2_rssnp.inputs)
    and3_ga.inout_pairs = spike_train_parser('test cases/and_test_cases.txt',and3_rssnp.inputs)
    or1_ga.inout_pairs = spike_train_parser('test cases/or_test_cases.txt',or1_rssnp.inputs)
    or2_ga.inout_pairs = spike_train_parser('test cases/or_test_cases.txt',or2_rssnp.inputs)
    or3_ga.inout_pairs = spike_train_parser('test cases/or_test_cases.txt',or3_rssnp.inputs)
    not1_ga.inout_pairs = spike_train_parser('test cases/not_test_cases.txt',not1_rssnp.inputs)
    not2_ga.inout_pairs = spike_train_parser('test cases/not_test_cases.txt',not2_rssnp.inputs)
    not3_ga.inout_pairs = spike_train_parser('test cases/not_test_cases.txt',not3_rssnp.inputs)
    add1_ga.inout_pairs = spike_train_parser('test cases/add_test_cases.txt',add1_rssnp.inputs)
    add2_ga.inout_pairs = spike_train_parser('test cases/add_test_cases.txt',add2_rssnp.inputs)
    add3_ga.inout_pairs = spike_train_parser('test cases/add_test_cases.txt',add3_rssnp.inputs)
    sub1_ga.inout_pairs = spike_train_parser('test cases/sub_test_cases.txt',sub1_rssnp.inputs)
    sub2_ga.inout_pairs = spike_train_parser('test cases/sub_test_cases.txt',sub2_rssnp.inputs)
    sub3_ga.inout_pairs = spike_train_parser('test cases/sub_test_cases.txt',sub3_rssnp.inputs)

    experiment_list = [
        # test experiment
        {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 5, 'runs': 1, 'selection_func': 1},
        # base line
        {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 1},
        {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 1},
         # increase generations
        {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
        {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # double mutation rate
        {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 1},
        {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 1},
        # double population
        {'population_size': 24, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 1},
        {'population_size': 24, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 1},
        # double population and mutation rate
        {'population_size': 24, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 1},
        {'population_size': 24, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 1},
        # increase generations and mutation rate
        {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
        {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # double population and generations
        {'population_size': 24, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
        {'population_size': 24, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # double population, mutation rate and increase generations
        {'population_size': 24, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
        {'population_size': 24, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},
        
        # # base line
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 0},
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 0},
        # # increase generations
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 0},
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 0},
        # # double mutation rate
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 0},
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 0},
        # # increase generations and mutation rate
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 0},
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 0},

        # # base line
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 1},
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 1},
        # # increase generations
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # # double mutation rate
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 1},
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 1},
        # # increase generations and mutation rate
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},

        # # base line
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 2},
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 2},
        # # increase generations
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 2},
        # {'population_size': 12, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 2},
        # # double mutation rate
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 2},
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 2},
        # # increase generations and mutation rate
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 2},
        # {'population_size': 12, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 2},

        # # base line
        # {'population_size': 12, 'mutation_rate': 10, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 0},
        # {'population_size': 12, 'mutation_rate': 10, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 0},
        # # increase generations
        # {'population_size': 12, 'mutation_rate': 4, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 0},
        # {'population_size': 12, 'mutation_rate': 4, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 0},


        ## End of Experiments

        
        # # double population
        # {'population_size': 24, 'mutation_rate': 100, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 2},
        # {'population_size': 24, 'mutation_rate': 100, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 2},
        # # double population and mutation rate
        # {'population_size': 24, 'mutation_rate': 50, 'fitness_function': 0, 'generations': 20, 'runs': 5, 'selection_func': 2},
        # {'population_size': 24, 'mutation_rate': 50, 'fitness_function': 1, 'generations': 20, 'runs': 5, 'selection_func': 2},
        
        # # double population and generations
        # {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 1},
        # # double population, mutation rate and increase generations
        # {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 0, 'generations': 50, 'runs': 5, 'selection_func': 2},
        # {'population_size': 24, 'mutation_rate': 10, 'fitness_function': 1, 'generations': 50, 'runs': 5, 'selection_func': 2},
    ]

    print("Starting experiments...")
    ctr = 1 # Experiment counter
    # Run stats for each GA framework
    for stats in experiment_list:
        dir = results_directory + "/Experiment" + str(ctr)
        ctr += 1
        if not os.path.exists(dir):
            os.makedirs(dir)
        # execute_experiment(dir, and1_rssnp, and1_ga, and1_gaeval, stats, "And1")
        # execute_experiment(dir, and2_rssnp, and2_ga, and2_gaeval, stats, "And2")
        # execute_experiment(dir, and3_rssnp, and3_ga, and3_gaeval, stats, "And3")
        # execute_experiment(dir, or1_rssnp, or1_ga, or1_gaeval, stats, "Or1")
        # execute_experiment(dir, or2_rssnp, or2_ga, or2_gaeval, stats, "Or2")
        # execute_experiment(dir, or3_rssnp, or3_ga, or3_gaeval, stats, "Or3")
        # execute_experiment(dir, not1_rssnp, not1_ga, not1_gaeval, stats, "Not1")
        # execute_experiment(dir, not2_rssnp, not2_ga, not2_gaeval, stats, "Not2")
        # execute_experiment(dir, not3_rssnp, not3_ga, not3_gaeval, stats, "Not3")
        #execute_experiment(dir, add1_rssnp, add1_ga, add1_gaeval, stats, "Add1")
        # execute_experiment(dir, add2_rssnp, add2_ga, add2_gaeval, stats, "Add2")
        # execute_experiment(dir, add3_rssnp, add3_ga, add3_gaeval, stats, "Add3")
        # execute_experiment(dir, sub1_rssnp, sub1_ga, sub1_gaeval, stats, "Sub1")
        # execute_experiment(dir, sub2_rssnp, sub2_ga, sub2_gaeval, stats, "Sub2")
        # execute_experiment(dir, sub3_rssnp, sub3_ga, sub3_gaeval, stats, "Sub3")


