import os
timer_adder_out = open(os.getcwd()+ "\\cpu_time_test", "w+")

tutorial_time = False

if tutorial_time == True:
	num_runs = 1
	timer_test = open(os.getcwd()+ "\\timer_directory\\test_cpu", "r")
	time_input_files = [timer_test]

#with alternating index
else:
	num_runs = 5
	time_reader_and_minimal = open(os.getcwd()+ "\\timer_directory\\cpuandminimal00outreal.txt", "r")
	time_reader_and_adversarial = open(os.getcwd()+ "\\timer_directory\\cpuandadversarial11outreal.txt", "r")
	time_reader_and_extra = open(os.getcwd()+ "\\timer_directory\\cpuandextra22outreal.txt", "r")
	time_reader_or_minimal = open(os.getcwd()+ "\\timer_directory\\cpuorminimal00outreal.txt", "r")
	time_reader_or_adversarial = open(os.getcwd()+ "\\timer_directory\\cpuoradversarial11outreal.txt", "r")
	time_reader_or_extra = open(os.getcwd()+ "\\timer_directory\\cpuorextra22outreal.txt", "r")
	time_reader_not_minimal = open(os.getcwd()+ "\\timer_directory\\cpunotminimal00outreal.txt", "r")
	time_reader_not_adversarial = open(os.getcwd()+ "\\timer_directory\\cpunotadversarial11outreal.txt", "r")
	time_reader_not_extra = open(os.getcwd()+ "\\timer_directory\\cpunotextra22outreal.txt", "r")
	time_reader_add_minimal = open(os.getcwd()+ "\\timer_directory\\cpuaddminimal00outreal.txt", "r")
	time_reader_add_adversarial = open(os.getcwd()+ "\\timer_directory\\cpuaddadversarial11outreal.txt", "r")
	time_reader_add_extra = open(os.getcwd()+ "\\timer_directory\\cpuaddextra22outreal.txt", "r")
	time_reader_sub_minimal = open(os.getcwd()+ "\\timer_directory\\cpusubminimal00outreal.txt", "r")
	time_reader_sub_adversarial = open(os.getcwd()+ "\\timer_directory\\cpusubadversarial11outreal.txt", "r")
	time_reader_sub_extra = open(os.getcwd()+ "\\timer_directory\\cpusubextra22outreal.txt", "r")

	time_reader_and_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandminimal00outreal.txt", "r")
	time_reader_and_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandadversarial11outreal.txt", "r")
	time_reader_and_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandextra22outreal.txt", "r")
	time_reader_or_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpuorminimal00outreal.txt", "r")
	time_reader_or_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpuoradversarial11outreal.txt", "r")
	time_reader_or_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpuorextra22outreal.txt", "r")
	time_reader_not_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpunotminimal00outreal.txt", "r")
	time_reader_not_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpunotadversarial11outreal.txt", "r")
	time_reader_not_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpunotextra22outreal.txt", "r")
	time_reader_add_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpuaddminimal00outreal.txt", "r")
	time_reader_add_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpuaddadversarial11outreal.txt", "r")
	time_reader_add_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpuaddextra22outreal.txt", "r")
	time_reader_sub_minimal_gpu = open(os.getcwd()+ "\\timer_directory\\gpusubminimal00outreal.txt", "r")
	time_reader_sub_adversarial_gpu = open(os.getcwd()+ "\\timer_directory\\gpusubadversarial11outreal.txt", "r")
	time_reader_sub_extra_gpu = open(os.getcwd()+ "\\timer_directory\\gpusubextra22outreal.txt", "r")
	
	time_input_files = [time_reader_and_minimal, time_reader_and_minimal_gpu, time_reader_and_adversarial, time_reader_and_adversarial_gpu, time_reader_and_extra, time_reader_and_extra_gpu, time_reader_or_minimal, time_reader_or_minimal_gpu, time_reader_or_adversarial, time_reader_or_adversarial_gpu, time_reader_or_extra, time_reader_or_extra_gpu, time_reader_not_minimal, time_reader_not_minimal_gpu, time_reader_not_adversarial, time_reader_not_adversarial_gpu, time_reader_not_extra, time_reader_not_extra_gpu, time_reader_add_minimal, time_reader_add_minimal_gpu, time_reader_add_adversarial, time_reader_add_adversarial_gpu, time_reader_add_extra, time_reader_add_extra_gpu, time_reader_sub_minimal, time_reader_sub_minimal_gpu, time_reader_sub_adversarial, time_reader_sub_adversarial_gpu, time_reader_sub_extra, time_reader_sub_extra_gpu]

def atof(s, gpu_format):
	while s:
		try:
			return float(s)
		except:
			s=s[:-1]
	return 0.0

index_file = 0


for time_reader in time_input_files:
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
	gpu_format = False

	if index_file % 2 == 0:
		timer_adder_out.write("CPU ")
		
	else:
		timer_adder_out.write("GPU ")
		gpu_format = True

	if int(index_file / 6) == 0:
		timer_adder_out.write("AND ")
	elif int(index_file / 6) == 1:
		timer_adder_out.write("OR ")
	elif int(index_file / 6) == 2:
		timer_adder_out.write("NOT ")
	elif int(index_file / 6) == 3:
		timer_adder_out.write("ADD ")
	elif int(index_file / 6) == 4:
		timer_adder_out.write("SUB ")

	if index_file % 6 == 0 or index_file % 6 == 1:
		timer_adder_out.write("Minimal \n")
	elif index_file % 6 == 2 or index_file % 6 == 3:
		timer_adder_out.write("Adversarial \n")
	elif index_file % 6 == 4 or index_file % 6 == 5:
		timer_adder_out.write("Extra \n") 
	loop_index = 0
	index_reader = -1
	for line in time_reader:
		array = line.split(' ')
		topass = None
		if array[0] == "Run":
			index_reader = int(array[3])
		else:
			if gpu_format == True:
				topass = array[7]
				#print("topass at 7 is ", topass)
			else:
				topass = array[4]
				#print("topass at 4 is ", topass)
			if array[0] == "Selection":
				selection_time += atof(topass, gpu_format)
				selection_array[index_reader] += selection_time
			elif array[0] == "Crossover":
				crossover_time += atof(topass, gpu_format)
				crossover_array[index_reader] += crossover_time
			elif array[0] == "Evaluate":
				evaluate_time += atof(topass, gpu_format)
				evaluate_array[index_reader] += evaluate_time
		
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
	index_file += 1
	time_reader.close()
	
