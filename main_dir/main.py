from src.experimenter_uncommented import gaframework, gaframework_gpu
from src.abstracts.norssnp_integ import set_bounds, set_values
from src.RSSNP_list import *
import yaml, os
import src.experimenter_uncommented
import src.abstracts.norssnp_integ 
import src.RSSNP_list

def conf_load(filename):
    with open(filename, 'r') as stream:
        try:
            ga_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return ga_params
def conf_save(filename, ga_params):
    with open(filename, 'w+') as out:
        doc = yaml.safe_dump(ga_params, out)

def create_empty_yaml(runs, generations, population_size, savefile_name):
	empty_dict = {}
	conf_save(savefile_name, empty_dict)
	ga_params = conf_load(savefile_name)
	ga_params['run_total'] = runs
	ga_params['gen_total'] = generations
	ga_params['runs'] = {}
	for i in range(runs):
		ga_params['runs'][i] = {}
		ga_params['runs'][i]['generations'] = {}
		for j in range(generations):
			ga_params['runs'][i]['generations'][j] = {}
			ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'] = {}
			for k in range(population_size):
				ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'][k] = {} 
	conf_save(savefile_name, ga_params)
	return ga_params

def continue_create_empty_yaml(savefile_name):
	ga_params = conf_load(savefile_name)
	run_start = ga_params['run_total']

	for i in range(run_start, ga_params['run_total'] + ga_params['runs_pending']):
		ga_params['runs'][i] = {}
		ga_params['runs'][i]['generations'] = {}
		for j in range(ga_params['gen_total'] + ga_params['gens_pending']):
			ga_params['runs'][i]['generations'][j] = {}
			ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'] = {}
			for k in range(ga_params['populations_pending']  + ga_params['runs'][0]['population_size']):
				ga_params['runs'][i]['generations'][j]['rssnp_chromosomes'][k] = {} 
	conf_save(savefile_name, ga_params)
	return ga_params


def prompt_make_newsavefile(ga_params, loadfile_name, load_directory):
	newfile_choice = input("Would you like to use another savefile (Y) or use the existing savefile (N) (Y/N): ")

	newfile_choice = True if newfile_choice == 'Y' else False

	newloadfile_name = None

	if newfile_choice == True:
		newloadfile_name = os.path.join(load_directory, input("What is the name of the new savefile: "))
		conf_save(newloadfile_name, ga_params)
		#print(ga_params)

	else:
		conf_save(loadfile_name, ga_params)
		newloadfile_name = loadfile_name
	return newloadfile_name

def program_main():
	print("Menu\n Enter a number of your choice: (1) Create a new evolutionary process w/ autosave\n (2) Load an evolutionary process from a configuration file: ")
	menu_choice = int(input("What would you like to do in this program: "))
	home = os.getcwd()
	if  menu_choice == 1:
		print("Given the choices of logic gates (1) AND, (2) OR, (3) NOT, (4) ADD, (5) SUB")
		answer = int(input("What kind of system would you like to evolve: "))
		print("Given the choices of version of the systems to evolve (1) minimal, (2) adversarial, (3) extra rules, (4) user-defined")
		type_answer = int(input("What kind of system would you like to evolve: "))
		save_directory = os.path.join(home, "load_directory")
		#save_directory = input("Which directory would you like to save (.yaml) the evolutionary process: ")
		#if not os.path.exists(save_directory):
		  #  os.makedirs(save_directory)
		savefile_name = os.path.join(save_directory, input("What will be the name of this savefile: ") + ".yaml")
		runs = int(input("How many runs would you like to do (min of 1): "))
		generations = int(input("How many generation would you like to do (min of 1): "))
		population_size = int(input("How many RSSNP should be in the population? "))
		mutation_rate = int((input("How likely should it mutate in percentage? "))) * 100
		print("Of the Parent Selection methods:\n 0. **Top 50% of the population**\n1. **50% of the population based on random**\n2. **Top 25% + 25% of the population based on fitness**")
		selection_func = int(input("Which would you use?"))
		print("Of the Fitness Selection methods:\n 0. Longest Common Subsequence\n1. Longest Common Substring\n2. Edit Distance Method")
		fitness_func = int(input("Which would you use?"))

		ga_params = create_empty_yaml(runs, generations, population_size, savefile_name)
		ga_params['runs_pending'] = 0
		ga_params['gens_pending'] = 0
		ga_params['populations_pending'] = 0
		ga_params['goal_fitness'] = 101
		for run in range(runs):
			ga_params['runs'][run]['max_fitness_in_run'] = 0
			ga_params['runs'][run]['population_size'] = population_size
			ga_params['runs'][run]['mutation_rate'] = mutation_rate
			ga_params['runs'][run]['selection_func'] = selection_func
			ga_params['runs'][run]['fitness_function'] = fitness_func



		#building the test_cases_path
		and_path = os.path.join(home, "test cases", "and_test_cases.txt")
		or_path = os.path.join(home, "test cases", "or_test_cases.txt")
		not_path = os.path.join(home, "test cases", "not_test_cases.txt")
		add_path = os.path.join(home, "test cases", "add_test_cases.txt")
		sub_path = os.path.join(home, "test cases", "sub_test_cases.txt")


		#Default or made or random system to initially evolve
		#for the different kind of evolutionary process (logic gates), we evolve
		system = None
		test_cases_path = None
		ga_params['test_cases_path'] = None

		if answer == 1:
			if type_answer == 1:
				system = and_rssnp_minimal
			if type_answer == 2:
				system = and_rssnp_adversarial
			if type_answer == 3:
				system = and_rssnp_extra_rules
			test_cases_path = and_path
			ga_params['test_cases_path'] = and_path

		if answer == 2:
			if type_answer == 1:
				system = or_rssnp_minimal
			if type_answer == 2:
				system = or_rssnp_adversarial
			if type_answer == 3:
				system = or_rssnp_extra_rules
			test_cases_path = or_path
			ga_params['test_cases_path'] = or_path

		if answer == 3:
			if type_answer == 1:
				system = not_rssnp_minimal
			if type_answer == 2:
				system = not_rssnp_adversarial
			if type_answer == 3:
				system = not_rssnp_extra_rules
			test_cases_path = not_path
			ga_params['test_cases_path'] = not_path

		if answer == 4:
			if type_answer == 1:
				system = add_rssnp_minimal
			if type_answer == 2:
				system = add_rssnp_adversarial
			if type_answer == 3:
				system = add_rssnp_extra_rules
			test_cases_path = add_path
			ga_params['test_cases_path'] = add_path

		if answer == 5:
			if type_answer == 1:
				system = sub_rssnp_minimal
			if type_answer == 2:
				system = sub_rssnp_adversarial
			if type_answer == 3:
				system = sub_rssnp_extra_rules
			test_cases_path = sub_path
			ga_params['test_cases_path'] = sub_path

		if type_answer == 4:
		 	system = set_values(ga_params, runs)
		 	if system == None:
		 		print("Rssnp given is invalid")
		 		exit()

		# random RSSNP option
		#if type_answer == 5:
			#system = set_bounds(ga_params, runs)


		conf_save(savefile_name, ga_params)
		if system == None or test_cases_path == None:
			print("Illegal system chosen or wrong direction for evolution")
		else:
			gaframework(system, test_cases_path, savefile_name)



	if int(menu_choice) == 2:
		load_directory = os.path.join(home, "load_directory")
		#load_directory = input("Which load directory would you like to load (.yaml) an evolutionary process from: ")
		loadfile_name = os.path.join(load_directory, input("What is the name of this loadfile (append a yaml extension) : "))
		ga_params = conf_load(loadfile_name)
		print("Load function starting")

		execution_choice = int(input("Where do you want your experiment to be executed (1) CPU or (2) GPU: "))
		start_from_a_gen = False
		ga_params['goal_fitness'] = 101
		if execution_choice == 1:

			#get the information from console
			print("Menu: Would you either increase (1) runs/generations/population_size or (2) a goal fitness of any chromosome in the evolutionary process: or (3) just maintain the current number of generation/population_size but extend runs further using an initial population from any generation number ")
			print("Note that for this program the GPU can also do option 3 but not the others ")
			sub_choice = int(input())
			if sub_choice == 1:
				add_runs = int(input("How many runs would you like to add (minimum of 1 and maximum of 100): "))
				add_gens = int(input("How many generations would you like to add (minimum of 0 and maximum of 100): "))
				add_populations = int(input("How many individuals would you like to add to the population (minimum of 0): "))
				ga_params['runs_pending'] = add_runs
				ga_params['gens_pending'] = add_gens
				ga_params['populations_pending'] = add_populations
				
			elif sub_choice == 2:
				print("Notice that the evolutionary process wont stop till a chromosome reached the desired goal fitness\n")
				print("Or till you exit the program itself (make sure its after a run in a GA) and the autosave will record only up to the point of termination\n")
				add_goal_fitness = int(input("What is the desired goal fitness: (minimum of 50 and maximum of 100): "))
				#101 denotes an unending condition no matter how many runs or generations are reported
				ga_params['runs_pending'] = 101
				ga_params['gens_pending'] = 101
				ga_params['populations_pending'] = 0
				ga_params['goal_fitness'] = add_goal_fitness
			elif sub_choice == 3:					
				print("Maintaining current number of generation and population_size")
				add_runs = int(input("How many runs would you like to add (minimum of 1 and maximum of 100): "))
				ga_params['gens_pending'] = 0
				ga_params['populations_pending'] = 0
				ga_params['runs_pending'] = add_runs
				ga_params['generation_index_continue'] = input("Which run and generation would you like to use as the starting parents of the succeeding runs (separate by comma)? ")
				start_from_a_gen = True

			
			newloadfile_name = prompt_make_newsavefile(ga_params, loadfile_name, load_directory)
			continue_create_empty_yaml(newloadfile_name)
			ga_params = conf_load(newloadfile_name)
			run_total = ga_params['run_total'] + ga_params['runs_pending']
			for run in range(ga_params['run_total'], run_total):
				ga_params['runs'][run]['max_fitness_in_run'] = 0
				ga_params['runs'][run]['population_size'] = ga_params['runs'][0]['population_size'] + ga_params['populations_pending']
				ga_params['runs'][run]['mutation_rate'] = ga_params['runs'][0]['mutation_rate']
				ga_params['runs'][run]['selection_func'] = ga_params['runs'][0]['selection_func']
				ga_params['runs'][run]['fitness_function'] = ga_params['runs'][0]['fitness_function']
			conf_save(newloadfile_name, ga_params)
			
			start_new = False

			print("Running GA in CPU: ")
			#The system that will be parent0 is the first rssnp chromosome in the ga_params
			#change the value of this into a system in RSSNP_list in order to change it
			#for subchoice 3, rssnp string is used only in order to acquire the input neuron indexes for spike_train_parser function
			rssnp_string = ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0]
			
			print("rssnp string is " + str(rssnp_string))
			gaframework(rssnp_string, ga_params['test_cases_path'] , newloadfile_name, start_new, start_from_a_gen)
			ga_params = conf_load(newloadfile_name)
			ga_params['run_total'] += ga_params['runs_pending']
			ga_params['runs_pending'] = 0
			ga_params['gen_total'] += ga_params['gens_pending']
			ga_params['gens_pending'] = 0
			ga_params['populations_pending'] = 0

			conf_save(newloadfile_name, ga_params)
		
		else:
			print("Running GA in GPU: ")
			
			print("Maintaining current number of generation and population_size")
			add_runs = int(input("How many runs would you like to add (minimum of 1 and maximum of 100): "))	
			ga_params['gens_pending'] = 0
			ga_params['populations_pending'] = 0
			ga_params['runs_pending'] = add_runs
			ga_params['generation_index_continue'] = input("Which run and generation would you like to use as the starting parents of the succeeding runs (separate by comma)? ")
			newloadfile_name =  prompt_make_newsavefile(ga_params, loadfile_name, load_directory)
			print("rewriting to 1", newloadfile_name)
			continue_create_empty_yaml(newloadfile_name)
			ga_params = conf_load(newloadfile_name)
			run_total = ga_params['run_total'] + ga_params['runs_pending']
			for run in range(ga_params['run_total'], run_total):
				ga_params['runs'][run]['max_fitness_in_run'] = 0
				ga_params['runs'][run]['population_size'] = ga_params['runs'][0]['population_size'] + ga_params['populations_pending']
				ga_params['runs'][run]['mutation_rate'] = ga_params['runs'][0]['mutation_rate']
				ga_params['runs'][run]['selection_func'] = ga_params['runs'][0]['selection_func']
				ga_params['runs'][run]['fitness_function'] = ga_params['runs'][0]['fitness_function']
			print("rewriting to 2", newloadfile_name)
			conf_save(newloadfile_name, ga_params)
			#rssnp_string = ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0]
			gaframework_gpu(newloadfile_name)
			ga_params = conf_load(newloadfile_name)
			ga_params['run_total'] += ga_params['runs_pending']
			ga_params['runs_pending'] = 0
			ga_params['gen_total'] += ga_params['gens_pending']
			ga_params['gens_pending'] = 0
			ga_params['populations_pending'] = 0

			print("rewriting to 3", newloadfile_name)
			conf_save(newloadfile_name, ga_params)

# ga_params = {
#     'population_size': 12,
#     'mutation_rate': 100,
#     'fitness_function': 0,
#     'generations': 50,
#     'runs': 1,
#     'selection_func': 1
# }
# home = os.getcwd()
# path = os.path.join(home, "test_cases", "sub_test_cases.txt")
# print(path)
# path_to_inputoutput_spike_trains = path

# gaframework(sub_rssnp_extra_rules, path_to_inputoutput_spike_trains, ga_params)

program_main()
