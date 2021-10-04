import os, yaml

def conf_load(filename):
    with filename as stream:
        try:
            timer_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return timer_params

timer_adder_out = open(os.getcwd()+ "\\time_test", "w+")



num_runs = 10

time_reader_and_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandminimal00outreal.yaml", "r")
time_reader_and_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandadversarial11outreal.yaml", "r")
time_reader_and_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandextra22outreal.yaml", "r")
time_reader_or_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpuorminimal00outreal.yaml", "r")
time_reader_or_adversarial_gpu = None
time_reader_or_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpuorextra22outreal.yaml", "r")
time_reader_not_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpunotminimal00outreal.yaml", "r")
time_reader_not_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpunotadversarial11outreal.yaml", "r")
time_reader_not_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpunotextra22outreal.yaml", "r")
time_reader_add_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpuaddminimal00outreal.yaml", "r")
time_reader_add_adversarial_gpu = None
time_reader_add_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpuaddextra22outreal.yaml", "r")
time_reader_sub_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpusubminimal00outreal.yaml", "r")
time_reader_sub_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpusubadversarial11outreal.yaml", "r")
time_reader_sub_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpusubextra22outreal.yaml", "r")

time_reader_and_minimal_gpu2 = open(os.getcwd()+ "\\timer_directory\\2gpuandminimal00outreal.yaml", "r")
time_reader_and_adversarial_gpu2 = None
time_reader_and_extra_gpu2 = None
time_reader_or_minimal_gpu2 = open(os.getcwd()+ "\\timer_directory\\2gpuorminimal00outreal.yaml", "r")
time_reader_or_adversarial_gpu2 = None
time_reader_or_extra_gpu2 = None
time_reader_not_minimal_gpu2 = open(os.getcwd()+ "\\timer_directory\\2gpunotminimal00outreal.yaml", "r")
time_reader_not_adversarial_gpu2 = None
time_reader_not_extra_gpu2 = None
time_reader_add_minimal_gpu2 = open(os.getcwd()+ "\\timer_directory\\2gpuaddminimal00outreal.yaml", "r")
time_reader_add_adversarial_gpu2 = None
time_reader_add_extra_gpu2 = None
time_reader_sub_minimal_gpu2 = open(os.getcwd()+ "\\timer_directory\\2gpusubminimal00outreal.yaml", "r")
time_reader_sub_adversarial_gpu2 = open(os.getcwd()+ "\\timer_directory\\2gpusubadversarial11outreal.yaml", "r")
time_reader_sub_extra_gpu2 = open(os.getcwd()+ "\\timer_directory\\2gpusubextra22outreal.yaml", "r")

time_reader_and_minimal_gpu3 = None
time_reader_and_adversarial_gpu3 = None
time_reader_and_extra_gpu3 = None
time_reader_or_minimal_gpu3 = None
time_reader_or_adversarial_gpu3 = None
time_reader_or_extra_gpu3 = None
time_reader_not_minimal_gpu3 = None
time_reader_not_adversarial_gpu3 = None
time_reader_not_extra_gpu3 = None
time_reader_add_minimal_gpu3 = None
time_reader_add_adversarial_gpu3 = None
time_reader_add_extra_gpu3 = None
time_reader_sub_minimal_gpu3 = open(os.getcwd()+ "\\timer_directory\\3gpusubminimal00outreal.yaml", "r")
time_reader_sub_adversarial_gpu3 = open(os.getcwd()+ "\\timer_directory\\3gpusubadversarial11outreal.yaml", "r")
time_reader_sub_extra_gpu3 = open(os.getcwd()+ "\\timer_directory\\3gpusubextra22outreal.yaml", "r")

time_reader_and_minimal_gpu4 = None
time_reader_and_adversarial_gpu4 = None
time_reader_and_extra_gpu4 = None
time_reader_or_minimal_gpu4 = None
time_reader_or_adversarial_gpu4 = None
time_reader_or_extra_gpu4 = None
time_reader_not_minimal_gpu4 = None
time_reader_not_adversarial_gpu4 = None
time_reader_not_extra_gpu4 = None
time_reader_add_minimal_gpu4 = None
time_reader_add_adversarial_gpu4 = None
time_reader_add_extra_gpu4 = None
time_reader_sub_minimal_gpu4 = open(os.getcwd()+ "\\timer_directory\\4gpusubminimal00outreal.yaml", "r")
time_reader_sub_adversarial_gpu4 = open(os.getcwd()+ "\\timer_directory\\4gpusubadversarial11outreal.yaml", "r")
time_reader_sub_extra_gpu4 = open(os.getcwd()+ "\\timer_directory\\4gpusubextra22outreal.yaml", "r")	


time_reader_and_minimal_gpu5 = None
time_reader_and_adversarial_gpu5 = None
time_reader_and_extra_gpu5 = None
time_reader_or_minimal_gpu5 = None
time_reader_or_adversarial_gpu5 = None
time_reader_or_extra_gpu5 = None
time_reader_not_minimal_gpu5 = None
time_reader_not_adversarial_gpu5 = None
time_reader_not_extra_gpu5 = None
time_reader_add_minimal_gpu5 = None
time_reader_add_adversarial_gpu5 = None
time_reader_add_extra_gpu5 = None
time_reader_sub_minimal_gpu5 = open(os.getcwd()+ "\\timer_directory\\5gpusubminimal00outreal.yaml", "r")
time_reader_sub_adversarial_gpu5 = open(os.getcwd()+ "\\timer_directory\\5gpusubadversarial11outreal.yaml", "r")
time_reader_sub_extra_gpu5 = open(os.getcwd()+ "\\timer_directory\\5gpusubextra22outreal.yaml", "r")

time_input_files = [time_reader_and_minimal_gpu, time_reader_and_adversarial_gpu, time_reader_and_extra_gpu, time_reader_or_minimal_gpu, time_reader_or_adversarial_gpu, time_reader_or_extra_gpu, time_reader_not_minimal_gpu, time_reader_not_adversarial_gpu, time_reader_not_extra_gpu, time_reader_add_minimal_gpu,  time_reader_add_adversarial_gpu, time_reader_add_extra_gpu, time_reader_sub_minimal_gpu,  time_reader_sub_adversarial_gpu, time_reader_sub_extra_gpu]
sec_time_input_files = [time_reader_and_minimal_gpu2, time_reader_and_adversarial_gpu2, time_reader_and_extra_gpu2, time_reader_or_minimal_gpu2, time_reader_or_adversarial_gpu2, time_reader_or_extra_gpu2, time_reader_not_minimal_gpu2, time_reader_not_adversarial_gpu2, time_reader_not_extra_gpu2, time_reader_add_minimal_gpu2, time_reader_add_adversarial_gpu2, time_reader_add_extra_gpu2, time_reader_sub_minimal_gpu2, time_reader_sub_adversarial_gpu2, time_reader_sub_extra_gpu2]
third_time_input_files = [time_reader_and_minimal_gpu3, time_reader_and_adversarial_gpu3, time_reader_and_extra_gpu3, time_reader_or_minimal_gpu3, time_reader_or_adversarial_gpu3, time_reader_or_extra_gpu3, time_reader_not_minimal_gpu3, time_reader_not_adversarial_gpu3, time_reader_not_extra_gpu3, time_reader_add_minimal_gpu3, time_reader_add_adversarial_gpu3, time_reader_add_extra_gpu3, time_reader_sub_minimal_gpu3, time_reader_sub_adversarial_gpu3, time_reader_sub_extra_gpu3]
fourth_time_input_files = [time_reader_and_minimal_gpu4, time_reader_and_adversarial_gpu4, time_reader_and_extra_gpu4, time_reader_or_minimal_gpu4, time_reader_or_adversarial_gpu4, time_reader_or_extra_gpu4, time_reader_not_minimal_gpu4, time_reader_not_adversarial_gpu4, time_reader_not_extra_gpu4, time_reader_add_minimal_gpu4, time_reader_add_adversarial_gpu4, time_reader_add_extra_gpu4, time_reader_sub_minimal_gpu4, time_reader_sub_adversarial_gpu4, time_reader_sub_extra_gpu4]
fifth_time_input_files = [time_reader_and_minimal_gpu5, time_reader_and_adversarial_gpu5, time_reader_and_extra_gpu5, time_reader_or_minimal_gpu5, time_reader_or_adversarial_gpu5, time_reader_or_extra_gpu5, time_reader_not_minimal_gpu5, time_reader_not_adversarial_gpu5, time_reader_not_extra_gpu5, time_reader_add_minimal_gpu5, time_reader_add_adversarial_gpu5, time_reader_add_extra_gpu5, time_reader_sub_minimal_gpu5, time_reader_sub_adversarial_gpu5, time_reader_sub_extra_gpu5]

time_input_array = [time_input_files, sec_time_input_files, third_time_input_files, fourth_time_input_files, fifth_time_input_files];
def atof(s, gpu_format):
	while s:
		try:
			return float(s)
		except:
			s=s[:-1]
	return 0.0

#for the file time_reader_and_minimal_gpu we have a complete successful 2 runs on the first file namely gpuandminimal00outreal.yaml and 3 runs for 2gpuandminimal00outreal.yaml
multi_runs_array = [[2,3], [5], [5], [2,3], [0], [5], [4,1], [5], [5], [3,2], [0], [5], [1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]];
print("mra is ", multi_runs_array)
break_run_index_file = 0
for time_input_files in time_input_array:
	index_file = 0
	for time_reader in time_input_files:
		if time_reader != None: 
			selection_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			evaluate_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			crossover_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			selection_time = 0
			evaluate_time = 0
			crossover_time = 0
			avg_selection_time = 0
			avg_evaluate_time = 0
			avg_crossover_time = 0
			avg_total_time = 0

			timer_adder_out.write("GPU ")
			gpu_format = True

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

			if index_file % 3 == 0 :
				timer_adder_out.write("Minimal \n")
			elif index_file % 3 == 1:
				timer_adder_out.write("Adversarial \n")
			elif index_file % 3 == 2:
				timer_adder_out.write("Extra \n")
			print('time reader is ', time_reader)
			
			loop_index = 0
			time_yamls = conf_load(time_reader)
			num_runs = multi_runs_array[index_file][break_run_index_file]
			for run in time_yamls['run_indexes']:
				#correction due to unfinished runs that wont be counted
				print("if is ", index_file, " and brif is ", break_run_index_file)
				if(loop_index < num_runs):
					selection_array[loop_index] = time_yamls['run_indexes'][run]['Selection']
					evaluate_array[loop_index] = time_yamls['run_indexes'][run]['Evaluate']
					crossover_array[loop_index] = time_yamls['run_indexes'][run]['Crossover']	
				
				
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

	break_run_index_file += 1
	
