import yaml, os
yaml_averager_out = open(os.getcwd()+ "\\yaml_test.txt", "w+")

tutorial_time = False

if tutorial_time == True:
	numruns = 1
	yaml_test = open(os.getcwd()+ "\\load_directory\\test.yaml", "r")
	yaml_input_files = [yaml_test]

else:
	numruns = 5
	yaml_reader_and_minimal = open(os.getcwd()+ "\\load_directory\\cpuandminimal00.yaml", "r")
	yaml_reader_and_adversarial = open(os.getcwd()+ "\\load_directory\\cpuandadversarial11.yaml", "r")
	yaml_reader_and_extra = open(os.getcwd()+ "\\load_directory\\cpuandextra22.yaml", "r")
	yaml_reader_or_minimal = open(os.getcwd()+ "\\load_directory\\cpuorminimal00.yaml", "r")
	yaml_reader_or_adversarial = open(os.getcwd()+ "\\load_directory\\cpuoradversarial11.yaml", "r")
	yaml_reader_or_extra = open(os.getcwd()+ "\\load_directory\\cpuorextra22.yaml", "r")
	yaml_reader_not_minimal = open(os.getcwd()+ "\\load_directory\\cpunotminimal00.yaml", "r")
	yaml_reader_not_adversarial = open(os.getcwd()+ "\\load_directory\\cpunotadversarial11.yaml", "r")
	yaml_reader_not_extra = open(os.getcwd()+ "\\load_directory\\cpunotextra22.yaml", "r")
	yaml_reader_add_minimal = open(os.getcwd()+ "\\load_directory\\cpuaddminimal00.yaml", "r")
	yaml_reader_add_adversarial = open(os.getcwd()+ "\\load_directory\\cpuaddadversarial11.yaml", "r")
	yaml_reader_add_extra = open(os.getcwd()+ "\\load_directory\\cpuaddextra22.yaml", "r")
	yaml_reader_sub_minimal = open(os.getcwd()+ "\\load_directory\\cpusubminimal00.yaml", "r")
	yaml_reader_sub_adversarial = open(os.getcwd()+ "\\load_directory\\cpusubadversarial11.yaml", "r")
	yaml_reader_sub_extra = open(os.getcwd()+ "\\load_directory\\cpusubextra22.yaml", "r")

	yaml_reader_and_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpuandminimal00.yaml", "r")
	yaml_reader_and_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpuandadversarial11.yaml", "r")
	yaml_reader_and_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpuandextra22.yaml", "r")
	yaml_reader_or_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpuorminimal00.yaml", "r")
	yaml_reader_or_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpuoradversarial11.yaml", "r")
	yaml_reader_or_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpuorextra22.yaml", "r")
	yaml_reader_not_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpunotminimal00.yaml", "r")
	yaml_reader_not_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpunotadversarial11.yaml", "r")
	yaml_reader_not_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpunotextra22.yaml", "r")
	yaml_reader_add_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpuaddminimal00.yaml", "r")
	yaml_reader_add_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpuaddadversarial11.yaml", "r")
	yaml_reader_add_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpuaddextra22.yaml", "r")
	yaml_reader_sub_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpusubminimal00.yaml", "r")
	yaml_reader_sub_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpusubadversarial11.yaml", "r")
	yaml_reader_sub_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpusubextra22.yaml", "r")
	
	yaml_input_files = [yaml_reader_and_minimal, yaml_reader_and_minimal_gpu, yaml_reader_and_adversarial, yaml_reader_and_adversarial_gpu, yaml_reader_and_extra, yaml_reader_and_extra_gpu, yaml_reader_or_minimal, yaml_reader_or_minimal_gpu, yaml_reader_or_adversarial, yaml_reader_or_adversarial_gpu, yaml_reader_or_extra, yaml_reader_or_extra_gpu, yaml_reader_not_minimal, yaml_reader_not_minimal_gpu, yaml_reader_not_adversarial, yaml_reader_not_adversarial_gpu, yaml_reader_not_extra, yaml_reader_not_extra_gpu, yaml_reader_add_minimal, yaml_reader_add_minimal_gpu, yaml_reader_add_adversarial, yaml_reader_add_adversarial_gpu, yaml_reader_add_extra, yaml_reader_add_extra_gpu, yaml_reader_sub_minimal, yaml_reader_sub_minimal_gpu, yaml_reader_sub_adversarial, yaml_reader_sub_adversarial_gpu, yaml_reader_sub_extra, yaml_reader_sub_extra_gpu]


index_file = 0
for yaml_reader in yaml_input_files:
	gpu_format = False
	if index_file % 2 == 0:
		yaml_averager_out.write("CPU ")
		
	else:
		yaml_averager_out.write("GPU ")
		gpu_format = True

	if int(index_file / 6) == 0:
		yaml_averager_out.write("AND ")
	elif int(index_file / 6) == 1:
		yaml_averager_out.write("OR ")
	elif int(index_file / 6) == 2:
		yaml_averager_out.write("NOT ")
	elif int(index_file / 6) == 3:
		yaml_averager_out.write("ADD ")
	elif int(index_file / 6) == 4:
		yaml_averager_out.write("SUB ")

	if index_file % 6 == 0 or index_file % 6 == 1:
		yaml_averager_out.write("Minimal \n")
	elif index_file % 6 == 2 or index_file % 6 == 3:
		yaml_averager_out.write("Adversarial \n")
	elif index_file % 6 == 4 or index_file % 6 == 5:
		yaml_averager_out.write("Extra \n") 
	max_fitness = 0
	with yaml_reader as f:
		data = yaml.safe_load(f)
		
		avg_len_rules = 0
		avg_fitness = 0
		for run in data['runs']:
			if gpu_format == True:
				if run >= 5:
					total_len_rules = 0
					total_fitness = 0
					for gen in data['runs'][run]['generations']:
						fitness_array = [0, 0, 0, 0, 0, 0, 0, 0]
						rules_array = [0, 0, 0, 0, 0, 0, 0, 0]
						#print("gen is ", gen)
						for chrom_index in data['runs'][run]['generations'][gen]['rssnp_chromosomes']:
							chrom =  data['runs'][run]['generations'][gen]['rssnp_chromosomes'][chrom_index]
							fitness_array[chrom_index] =chrom['chrom_fitness']
							if max_fitness < fitness_array[chrom_index]:
								max_fitness = fitness_array[chrom_index]
							rules_array[chrom_index] = len(chrom['rules'])
							#print("chrom is ", chrom)

						total_len_rules += sum(rules_array)/len(data['runs'][run]['generations'][gen]['rssnp_chromosomes'])
						total_fitness += sum(fitness_array)/len(data['runs'][run]['generations'][gen]['rssnp_chromosomes'])

			else:
				total_len_rules = 0
				total_fitness = 0
				for gen in data['runs'][run]['generations']:
					fitness_array = [0, 0, 0, 0, 0, 0, 0, 0]
					rules_array = [0, 0, 0, 0, 0, 0, 0, 0]
					#print("gen is ", gen)
					for chrom_index in data['runs'][run]['generations'][gen]['rssnp_chromosomes']:
						chrom =  data['runs'][run]['generations'][gen]['rssnp_chromosomes'][chrom_index]
						fitness_array[chrom_index] =chrom['chrom_fitness']
						if max_fitness < fitness_array[chrom_index]:
								max_fitness = fitness_array[chrom_index]
						rules_array[chrom_index] = len(chrom['rules'])
						#print("chrom is ", chrom)

					total_len_rules += sum(rules_array)/len(data['runs'][run]['generations'][gen]['rssnp_chromosomes'])
					total_fitness += sum(fitness_array)/len(data['runs'][run]['generations'][gen]['rssnp_chromosomes'])

			if gpu_format == True:
				if run >= 5:
					numgens = len(data['runs'][run]['generations'])
					avg_len_rules += total_len_rules/numgens
					avg_fitness += total_fitness/numgens	
			else:
				numgens = len(data['runs'][run]['generations'])
				avg_len_rules += total_len_rules/numgens
				avg_fitness += total_fitness/numgens


		avg_len_rules = avg_len_rules/numruns
		avg_fitness = avg_fitness/numruns
		yaml_averager_out.write("\n Highest fitness among runs and generations: " + str(max_fitness))
		yaml_averager_out.write("\nAvg len of rules: " + str(avg_len_rules))
		yaml_averager_out.write("\nAvg fitness: " + str(avg_fitness) + "\n")





		#print(data)
	index_file += 1