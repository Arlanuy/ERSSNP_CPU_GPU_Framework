import os, yaml
timer_adder_out = open(os.getcwd()+ "\\cpu_time_test", "w+")

def conf_load(filename):
    with filename as stream:
        try:
            timer_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return timer_params

num_runs = 5
time_reader_and_minimal = open(os.getcwd()+ "\\timer_directory\\cpuandminimal00outreal.yaml", "r")
time_reader_and_adversarial = open(os.getcwd()+ "\\timer_directory\\cpuandadversarial11outreal.yaml", "r")
time_reader_and_extra = open(os.getcwd()+ "\\timer_directory\\cpuandextra22outreal.yaml", "r")
time_reader_or_minimal = open(os.getcwd()+ "\\timer_directory\\cpuorminimal00outreal.yaml", "r")
time_reader_or_adversarial = None
time_reader_or_extra = open(os.getcwd()+ "\\timer_directory\\cpuorextra22outreal.yaml", "r")
time_reader_not_minimal = open(os.getcwd()+ "\\timer_directory\\cpunotminimal00outreal.yaml", "r")
time_reader_not_adversarial = open(os.getcwd()+ "\\timer_directory\\cpunotadversarial11outreal.yaml", "r")
time_reader_not_extra = open(os.getcwd()+ "\\timer_directory\\cpunotextra22outreal.yaml", "r")
time_reader_add_minimal = open(os.getcwd()+ "\\timer_directory\\cpuaddminimal00outreal.yaml", "r")
time_reader_add_adversarial = None
time_reader_add_extra = open(os.getcwd()+ "\\timer_directory\\cpuaddextra22outreal.yaml", "r")
time_reader_sub_minimal = open(os.getcwd()+ "\\timer_directory\\cpusubminimal00outreal.yaml", "r")
time_reader_sub_adversarial = open(os.getcwd()+ "\\timer_directory\\cpusubadversarial11outreal.yaml", "r")
time_reader_sub_extra = open(os.getcwd()+ "\\timer_directory\\cpusubextra22outreal.yaml", "r")


time_input_files = [time_reader_and_minimal,  time_reader_and_adversarial,  time_reader_and_extra, time_reader_or_minimal, time_reader_or_adversarial, time_reader_or_extra, time_reader_not_minimal, time_reader_not_adversarial, time_reader_not_extra, time_reader_add_minimal, time_reader_add_adversarial, time_reader_add_extra, time_reader_sub_minimal, time_reader_sub_adversarial, time_reader_sub_extra]

def atof(s, gpu_format):
	while s:
		try:
			return float(s)
		except:
			s=s[:-1]
	return 0.0

index_file = 0


for time_reader in time_input_files:
	if time_reader != None:
		selection_array = [0] * 40
		evaluate_array = [0] * 40
		crossover_array = [0] * 40
		selection_time = 0
		evaluate_time = 0
		crossover_time = 0
		avg_selection_time = 0
		avg_evaluate_time = 0
		avg_crossover_time = 0
		avg_total_time = 0

		timer_adder_out.write("CPU ")

		if int(index_file / 3) == 0:
			timer_adder_out.write("AND ")
		elif int(index_file / 3) == 1:
			timer_adder_out.write("OR ")
		elif int(index_file / 3) == 2:
			timer_adder_out.write("NOT ")
		elif int(index_file / 3) == 3:
			timer_adder_out.write("ADD ")
		elif int(index_file / 3) == 4:
			timer_adder_out.write("SUB ")

		if index_file % 3 == 0:
			timer_adder_out.write("Minimal \n")
		elif index_file % 3 == 1:
			timer_adder_out.write("Adversarial \n")
		elif index_file % 3 == 2:
			timer_adder_out.write("Extra \n") 
		loop_index = 0
		time_yaml = conf_load(time_reader)

		for run in time_yaml['run_indexes']:
			selection_array[loop_index] = time_yaml['run_indexes'][run]['Selection']
			evaluate_array[loop_index] = time_yaml['run_indexes'][run]['Evaluate']
			crossover_array[loop_index] = time_yaml['run_indexes'][run]['Crossover']
			
			loop_index += 1
			
		print("selection array is ", (selection_array))
		print("crossover array is ", (crossover_array))
		print("evaluate array is ", (evaluate_array))
		avg_selection_time = sum(selection_array)/num_runs
		avg_crossover_time = sum(crossover_array)/num_runs
		print("sum eval is ", sum(evaluate_array))
		avg_evaluate_time = sum(evaluate_array)/num_runs
		avg_total_time = avg_selection_time + avg_crossover_time + avg_evaluate_time
		print("written avg evaluate time is ", avg_evaluate_time )
		timer_adder_out.write("selection time is " + str(avg_selection_time) + " crossover is " + str(avg_crossover_time) + " evaluate is " + str(avg_evaluate_time) + "\n")
		timer_adder_out.write("Total time is " + str(avg_total_time) + "\n")
		time_reader.close()
	index_file += 1
	
	
