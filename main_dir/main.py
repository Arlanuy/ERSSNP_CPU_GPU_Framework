from src.experimenter_uncommented import gaframework, execute_experiment
from src.abstracts.norssnp_integ import set_bounds
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

def create_empty_yaml(runs, generations, population_size):
	new_file = open("temp.yaml", 'w+')
	empty_dict = {}
	conf_save("temp.yaml", empty_dict)
	ga_params = conf_load("temp.yaml")
	ga_params['runs'] = [dict() for i in range(runs)]
	#ga_params['runs'][0]  = {}
	ga_params['runs'][0]['generations'] = [dict() for i in range(generations)]
	ga_params['runs'][0]['generations'][0] = {}
	ga_params['runs'][0]['generations'][0]['rssnp_chromosomes'] = [dict() for i in range(population_size)]
	return ga_params


def program_main():
	print("Menu\n Enter a number of your choice: (1) Create a new evolutionary process w/ autosave\n (2) Load an evolutionary process from a configuration file: ")
	menu_choice = input("What would you like to do in this program: ")

	if int(menu_choice) == 1:
		print("Given the choices of logic gates (1) AND, (2) OR, (3) NOT, (4) ADD, (5) SUB")
		answer = input("What kind of system would you like to evolve: ")
		save_directory = "C:\\Users\\Arvy\\Documents\\cs 199 docs\\RSSNP_CPU_GPU_Framework\\main_dir\\load_directory"
		#save_directory = input("Which directory would you like to save (.yaml) the evolutionary process: ")
		savefile_name = save_directory + '\\' + input("What will be the name of this savefile: ")
		runs = int(input("How many runs would you like to do (min of 1): "))
		generations = int(input("How many generation would you like to do (min of 1): "))
		population_size = int(input("How many RSSNP should be in the population? "))
		mutation_rate = int((input("How likely should it mutate in percentage? "))) * 100
		print("Of the Parent Selection methods:\n 0. **Top 50% of the population**\n1. **25% of the population based on fitness**\n2. **Top 25% + 25% of the population based on fitness**")
		selection_func = input("Which would you use?")
		ga_params = create_empty_yaml(runs, generations, population_size)
		ga_params = set_bounds(ga_params, runs, population_size)
		savefile = conf_save(savefile_name + ".yaml", ga_params)

		if not os.path.exists(save_directory):
		    os.makedirs(save_directory)

	if int(menu_choice) == 2:
		load_directory = "C:\\Users\\Arvy\\Documents\\cs 199 docs\\RSSNP_CPU_GPU_Framework\\main_dir\\load_directory"
		#load_directory = input("Which load directory would you like to load (.yaml) an evolutionary process from: ")
		loadfile_name = load_directory +'\\' + input("What is the name of this loadfile : ")
		ga_params = conf_load(loadfile_name)
		print(ga_params)
	#gaframework(rssnp_string, path_to_io_spike_trains, stats)

program_main()