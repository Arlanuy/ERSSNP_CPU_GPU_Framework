from src.experimenter_uncommented import gaframework, execute_experiment
from src.abstracts.norssnp_integ import set_bounds
from src.RSSNP_list import *
import yaml, os

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



def program_main():
	print("Menu\n Enter a number of your choice: (1) Create a new evolutionary process w/ autosave\n (2) Load an evolutionary process from a configuration file: ")
	menu_choice = int(input("What would you like to do in this program: "))
	home = os.getcwd()
	if  menu_choice == 1:
		print("Given the choices of logic gates (1) AND, (2) OR, (3) NOT, (4) ADD, (5) SUB")
		answer = int(input("What kind of system would you like to evolve: "))
		save_directory = os.path.join(home, "load_directory")
		#save_directory = input("Which directory would you like to save (.yaml) the evolutionary process: ")
		savefile_name = save_directory + '\\' + input("What will be the name of this savefile: ") + ".yaml"
		runs = int(input("How many runs would you like to do (min of 1): "))
		generations = int(input("How many generation would you like to do (min of 1): "))
		population_size = int(input("How many RSSNP should be in the population? "))
		mutation_rate = int((input("How likely should it mutate in percentage? "))) * 100
		print("Of the Parent Selection methods:\n 0. **Top 50% of the population**\n1. **25% of the population based on fitness**\n2. **Top 25% + 25% of the population based on fitness**")
		selection_func = int(input("Which would you use?"))
		print("Of the Fitness Selection methods:\n 0. Longest Common Subsequence\n1. Longest Common Substring\n2. Edit Distance Method")
		fitness_func = int(input("Which would you use?"))
		ga_params = create_empty_yaml(runs, generations, population_size, savefile_name)
		ga_params = set_bounds(ga_params, runs, population_size, mutation_rate, selection_func, fitness_func)


		
		#for the different kind of evolutionary process (logic gates), we evolve
		#AND gate
		if answer == 1:
			and_path = os.path.join(home, "test cases", "and_test_cases.txt")
			gaframework(ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0], and_path, ga_params, savefile_name)
		if answer == 2:
			or_path = os.path.join(home, "test cases", "or_test_cases.txt")
			gaframework(ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0], or_path, ga_params, savefile_name)
		if answer == 3:
			not_path = os.path.join(home, "test cases", "not_test_cases.txt")
			gaframework(ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0], not_path, ga_params, savefile_name)
		if answer == 4:
			add_path = os.path.join(home, "test cases", "add_test_cases.txt")
			gaframework(ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0], add_path, ga_params, savefile_name)
		if answer == 5:
			sub_path = os.path.join(home, "test cases", "sub_test_cases.txt")
			gaframework(ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'][0], sub_path, ga_params, savefile_name)
		conf_save(savefile_name, ga_params)
		if not os.path.exists(save_directory):
		    os.makedirs(save_directory)

	if int(menu_choice) == 2:
		load_directory = os.path.join(home, "load_directory")
		#load_directory = input("Which load directory would you like to load (.yaml) an evolutionary process from: ")
		loadfile_name = load_directory +'\\' + input("What is the name of this loadfile : ")
		ga_params = conf_load(loadfile_name)
		print(ga_params)
	#gaframework(rssnp_string, path_to_io_spike_trains, stats)

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