import yaml, os

def conf_load(filename):
    with filename as stream:
        try:
            timer_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return timer_params

yaml_averager_out = open(os.getcwd()+ "\\data_test.txt", "w+")

yaml_reader_and_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpuandminimal00.yaml", "r")
yaml_reader_and_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpuandadversarial11.yaml", "r")
yaml_reader_and_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpuandextra22.yaml", "r")
yaml_reader_or_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpuorminimal00.yaml", "r")
yaml_reader_or_adversarial_gpu = None
yaml_reader_or_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpuorextra22.yaml", "r")
yaml_reader_not_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpunotminimal00.yaml", "r")
yaml_reader_not_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpunotadversarial11.yaml", "r")
yaml_reader_not_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpunotextra22.yaml", "r")
yaml_reader_add_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpuaddminimal00.yaml", "r")
yaml_reader_add_adversarial_gpu = None
yaml_reader_add_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpuaddextra22.yaml", "r")
yaml_reader_sub_minimal_gpu = open(os.getcwd()+ "\\load_directory\\gpusubminimal00.yaml", "r")
yaml_reader_sub_adversarial_gpu = open(os.getcwd()+ "\\load_directory\\gpusubadversarial11.yaml", "r")
yaml_reader_sub_extra_gpu = open(os.getcwd()+ "\\load_directory\\gpusubextra22.yaml", "r")

yaml_reader_and_minimal_gpu2 = open(os.getcwd()+ "\\load_directory\\2gpuandminimal00.yaml", "r")
yaml_reader_and_adversarial_gpu2 = None
yaml_reader_and_extra_gpu2 = None
yaml_reader_or_minimal_gpu2 = open(os.getcwd()+ "\\load_directory\\2gpuorminimal00.yaml", "r")
yaml_reader_or_adversarial_gpu2 = None
yaml_reader_or_extra_gpu2 = None
yaml_reader_not_minimal_gpu2 = open(os.getcwd()+ "\\load_directory\\2gpunotminimal00.yaml", "r")
yaml_reader_not_adversarial_gpu2 = None
yaml_reader_not_extra_gpu2 = None
yaml_reader_add_minimal_gpu2 = open(os.getcwd()+ "\\load_directory\\2gpuaddminimal00.yaml", "r")
yaml_reader_add_adversarial_gpu2 = None
yaml_reader_add_extra_gpu2 = None
yaml_reader_sub_minimal_gpu2 = open(os.getcwd()+ "\\load_directory\\2gpusubminimal00.yaml", "r")
yaml_reader_sub_adversarial_gpu2 = open(os.getcwd()+ "\\load_directory\\2gpusubadversarial11.yaml", "r")
yaml_reader_sub_extra_gpu2 = open(os.getcwd()+ "\\load_directory\\2gpusubextra22.yaml", "r")

yaml_reader_and_minimal_gpu3 = None
yaml_reader_and_adversarial_gpu3 = None
yaml_reader_and_extra_gpu3 = None
yaml_reader_or_minimal_gpu3 = None
yaml_reader_or_adversarial_gpu3 = None
yaml_reader_or_extra_gpu3 = None
yaml_reader_not_minimal_gpu3 = None
yaml_reader_not_adversarial_gpu3 = None
yaml_reader_not_extra_gpu3 = None
yaml_reader_add_minimal_gpu3 = None
yaml_reader_add_adversarial_gpu3 = None
yaml_reader_add_extra_gpu3 = None
yaml_reader_sub_minimal_gpu3 = open(os.getcwd()+ "\\load_directory\\3gpusubminimal00.yaml", "r")
yaml_reader_sub_adversarial_gpu3 = open(os.getcwd()+ "\\load_directory\\3gpusubadversarial11.yaml", "r")
yaml_reader_sub_extra_gpu3 = open(os.getcwd()+ "\\load_directory\\3gpusubextra22.yaml", "r")

yaml_reader_and_minimal_gpu4 = None
yaml_reader_and_adversarial_gpu4 = None
yaml_reader_and_extra_gpu4 = None
yaml_reader_or_minimal_gpu4 = None
yaml_reader_or_adversarial_gpu4 = None
yaml_reader_or_extra_gpu4 = None
yaml_reader_not_minimal_gpu4 = None
yaml_reader_not_adversarial_gpu4 = None
yaml_reader_not_extra_gpu4 = None
yaml_reader_add_minimal_gpu4 = None
yaml_reader_add_adversarial_gpu4 = None
yaml_reader_add_extra_gpu4 = None
yaml_reader_sub_minimal_gpu4 = open(os.getcwd()+ "\\load_directory\\4gpusubminimal00.yaml", "r")
yaml_reader_sub_adversarial_gpu4 = open(os.getcwd()+ "\\load_directory\\4gpusubadversarial11.yaml", "r")
yaml_reader_sub_extra_gpu4 = open(os.getcwd()+ "\\load_directory\\4gpusubextra22.yaml", "r")	


yaml_reader_and_minimal_gpu5 = None
yaml_reader_and_adversarial_gpu5 = None
yaml_reader_and_extra_gpu5 = None
yaml_reader_or_minimal_gpu5 = None
yaml_reader_or_adversarial_gpu5 = None
yaml_reader_or_extra_gpu5 = None
yaml_reader_not_minimal_gpu5 = None
yaml_reader_not_adversarial_gpu5 = None
yaml_reader_not_extra_gpu5 = None
yaml_reader_add_minimal_gpu5 = None
yaml_reader_add_adversarial_gpu5 = None
yaml_reader_add_extra_gpu5 = None
yaml_reader_sub_minimal_gpu5 = open(os.getcwd()+ "\\load_directory\\5gpusubminimal00.yaml", "r")
yaml_reader_sub_adversarial_gpu5 = open(os.getcwd()+ "\\load_directory\\5gpusubadversarial11.yaml", "r")
yaml_reader_sub_extra_gpu5 = open(os.getcwd()+ "\\load_directory\\5gpusubextra22.yaml", "r")	

existing_run = 5


yaml_input_files = [yaml_reader_and_minimal_gpu, yaml_reader_and_adversarial_gpu, yaml_reader_and_extra_gpu, yaml_reader_or_minimal_gpu, yaml_reader_or_adversarial_gpu, yaml_reader_or_extra_gpu, yaml_reader_not_minimal_gpu, yaml_reader_not_adversarial_gpu, yaml_reader_not_extra_gpu, yaml_reader_add_minimal_gpu, yaml_reader_add_adversarial_gpu, yaml_reader_add_extra_gpu, yaml_reader_sub_minimal_gpu, yaml_reader_sub_adversarial_gpu, yaml_reader_sub_extra_gpu]
sec_yaml_input_files = [yaml_reader_and_minimal_gpu2, yaml_reader_and_adversarial_gpu2, yaml_reader_and_extra_gpu2, yaml_reader_or_minimal_gpu2, yaml_reader_or_adversarial_gpu2, yaml_reader_or_extra_gpu2, yaml_reader_not_minimal_gpu2, yaml_reader_not_adversarial_gpu2, yaml_reader_not_extra_gpu2, yaml_reader_add_minimal_gpu2, yaml_reader_add_adversarial_gpu2, yaml_reader_add_extra_gpu2, yaml_reader_sub_minimal_gpu2, yaml_reader_sub_adversarial_gpu2, yaml_reader_sub_extra_gpu2]
third_yaml_input_files = [yaml_reader_and_minimal_gpu3, yaml_reader_and_adversarial_gpu3, yaml_reader_and_extra_gpu3, yaml_reader_or_minimal_gpu3, yaml_reader_or_adversarial_gpu3, yaml_reader_or_extra_gpu3, yaml_reader_not_minimal_gpu3, yaml_reader_not_adversarial_gpu3, yaml_reader_not_extra_gpu3, yaml_reader_add_minimal_gpu3, yaml_reader_add_adversarial_gpu3, yaml_reader_add_extra_gpu3, yaml_reader_sub_minimal_gpu3, yaml_reader_sub_adversarial_gpu3, yaml_reader_sub_extra_gpu3]
fourth_yaml_input_files = [yaml_reader_and_minimal_gpu4, yaml_reader_and_adversarial_gpu4, yaml_reader_and_extra_gpu4, yaml_reader_or_minimal_gpu4, yaml_reader_or_adversarial_gpu4, yaml_reader_or_extra_gpu4, yaml_reader_not_minimal_gpu4, yaml_reader_not_adversarial_gpu4, yaml_reader_not_extra_gpu4, yaml_reader_add_minimal_gpu4, yaml_reader_add_adversarial_gpu4, yaml_reader_add_extra_gpu4, yaml_reader_sub_minimal_gpu4, yaml_reader_sub_adversarial_gpu4, yaml_reader_sub_extra_gpu4]
fifth_yaml_input_files = [yaml_reader_and_minimal_gpu5, yaml_reader_and_adversarial_gpu5, yaml_reader_and_extra_gpu5, yaml_reader_or_minimal_gpu5, yaml_reader_or_adversarial_gpu5, yaml_reader_or_extra_gpu5, yaml_reader_not_minimal_gpu5, yaml_reader_not_adversarial_gpu5, yaml_reader_not_extra_gpu5, yaml_reader_add_minimal_gpu5, yaml_reader_add_adversarial_gpu5, yaml_reader_add_extra_gpu5, yaml_reader_sub_minimal_gpu5, yaml_reader_sub_adversarial_gpu5, yaml_reader_sub_extra_gpu5]
yaml_input_array = [yaml_input_files, sec_yaml_input_files, third_yaml_input_files, fourth_yaml_input_files, fifth_yaml_input_files];
multi_runs_array = [[2,3], [5], [5], [2,3], [0], [5], [4,1], [5], [5], [3,2], [0], [5], [1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
break_run_index_file = 0
for yaml_input_files in yaml_input_array:
	index_file = 0
	for yaml_reader in yaml_input_files:

		if yaml_reader != None:

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
			elif index_file % 3== 2:
				yaml_averager_out.write("Extra \n") 
			max_fitness = 0
			num_runs = multi_runs_array[index_file][break_run_index_file]
			data = conf_load(yaml_reader)
			print('yaml reader is ', yaml_reader)
			avg_len_rules = 0
			avg_fitness = 0
			for run in data['runs']:
				if run >= existing_run and run < num_runs + existing_run:
					total_len_rules = 0
					total_fitness = 0
					#print("run is ", run)
					for gen in data['runs'][run]['generations']:
						if not int(index_file / 3) == 4 and (index_file % 3 == 2 or index_file %3 == 0) and gen == 11:
							fitness_array = [0] * 40
							rules_array = [0] * 40
							#print("gen is ", gen)
							for chrom_index in data['runs'][run]['generations'][gen]['rssnp_chromosomes']:
								print("chrom index is ", chrom_index, " in gen ", gen, " and run ", run)
								chrom =  data['runs'][run]['generations'][gen]['rssnp_chromosomes'][chrom_index]
								fitness_array[chrom_index] =chrom['chrom_fitness']
								if max_fitness < fitness_array[chrom_index]:
									max_fitness = fitness_array[chrom_index]
								rules_array[chrom_index] = len(chrom['rules'])
								#print("chrom is ", chrom)

							total_len_rules += sum(rules_array)/len(data['runs'][run]['generations'][gen]['rssnp_chromosomes'])
							total_fitness += sum(fitness_array)/len(data['runs'][run]['generations'][gen]['rssnp_chromosomes'])

					
					numgens = len(data['runs'][run]['generations'])
					#to cater for the case that gpusubextraminimal00.yaml and gpusubextra22.yaml only contains 11 generations per run
					if int(index_file / 3) == 4 and (index_file % 3 == 2 or index_file %3 == 0):
						numgens -= 1
					avg_len_rules += total_len_rules/numgens
					avg_fitness += total_fitness/numgens	

			
			avg_len_rules = avg_len_rules/num_runs
			avg_fitness = avg_fitness/num_runs
			yaml_averager_out.write("\n Highest fitness among runs and generations: " + str(max_fitness))
			yaml_averager_out.write("\nAvg len of rules: " + str(avg_len_rules))
			yaml_averager_out.write("\nAvg fitness: " + str(avg_fitness) + "\n")





			#print(data)
		index_file += 1
	break_run_index_file += 1