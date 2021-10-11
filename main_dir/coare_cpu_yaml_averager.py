import yaml, os
yaml_averager_out = open(os.getcwd()+ "\\cpu_data_test.txt", "w+")

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
	yaml_reader_or_adversarial = None
	yaml_reader_or_extra = open(os.getcwd()+ "\\load_directory\\cpuorextra22.yaml", "r")
	yaml_reader_not_minimal = open(os.getcwd()+ "\\load_directory\\cpunotminimal00.yaml", "r")
	yaml_reader_not_adversarial = open(os.getcwd()+ "\\load_directory\\cpunotadversarial11.yaml", "r")
	yaml_reader_not_extra = open(os.getcwd()+ "\\load_directory\\cpunotextra22.yaml", "r")
	yaml_reader_add_minimal = open(os.getcwd()+ "\\load_directory\\cpuaddminimal00.yaml", "r")
	yaml_reader_add_adversarial = None
	yaml_reader_add_extra = open(os.getcwd()+ "\\load_directory\\cpuaddextra22.yaml", "r")
	yaml_reader_sub_minimal = open(os.getcwd()+ "\\load_directory\\cpusubminimal00.yaml", "r")
	yaml_reader_sub_adversarial = open(os.getcwd()+ "\\load_directory\\cpusubadversarial11.yaml", "r")
	yaml_reader_sub_extra = open(os.getcwd()+ "\\load_directory\\cpusubextra22.yaml", "r")

	
	yaml_input_files = [yaml_reader_and_minimal, yaml_reader_and_adversarial, yaml_reader_and_extra, yaml_reader_or_minimal, yaml_reader_or_adversarial, yaml_reader_or_extra, yaml_reader_not_minimal,  yaml_reader_not_adversarial, yaml_reader_not_extra, yaml_reader_add_minimal, yaml_reader_add_adversarial, yaml_reader_add_extra,  yaml_reader_sub_minimal, yaml_reader_sub_adversarial, yaml_reader_sub_extra]


index_file = 0
for yaml_reader in yaml_input_files:
	if yaml_reader != None:
		#print("file is ", yaml_reader);

		yaml_averager_out.write("GPU ")

		if int(index_file / 3) == 0:
			yaml_averager_out.write("AND ")
		elif int(index_file / 3) == 1:
			yaml_averager_out.write("OR ")
		elif int(index_file / 3) == 2:
			yaml_averager_out.write("NOT ")
		elif int(index_file / 3) == 3:
			yaml_averager_out.write("ADD ")
		elif int(index_file / 3) == 4:
			yaml_averager_out.write("SUB ")

		if index_file % 3 == 0:
			yaml_averager_out.write("Minimal \n")
		elif index_file % 3 == 1:
			yaml_averager_out.write("Adversarial \n")
		elif index_file % 3 == 2:
			yaml_averager_out.write("Extra \n") 
		max_fitness = 0
		with yaml_reader as f:
			data = yaml.safe_load(f)
			
			avg_len_rules = 0
			avg_fitness = 0
			for run in data['runs']:
	
				total_len_rules = 0
				total_fitness = 0
				for gen in data['runs'][run]['generations']:
					fitness_array = [0] * 40
					rules_array = [0] * 40
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