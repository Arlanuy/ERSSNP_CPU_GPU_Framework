import yaml
ga_params  = {
  'runs':
    {0:
      {
        'population_size': 2,
        'mutation_rate': 100,
        'fitness_function': 1,
        'selection_func': 1,
        'generations':
        { 0:
          {
	          'best_fitness_result': 100,
	          'best_chromosome_indexes': [0,1],
	          'rssnp_chromosomes':
	           {
		          0:#rssnp index
		          {
					'neurons': 4,
					'synapses': 3,
					'rules': [
					    [0, 2, [1, 0], 1, 1, 0],
					    [1, 2, [1, 0], 1, 1, 0],
					    [2, 3, [2, 0], 2, 1, 0],
					    [2, 3, [1, 0], 1, 0, 0], 
					],
					'init_config': [0, 0, 0, 0],
					'rule_status': [-1, -1, -1, -1],
					'input_neurons': [0, 1],
					'output_neurons': [3],
				    'out_pairs':
				   		[	   			
					      [([0, 0, 0, 0, 0, 1], [0, 0, 0, 1]), 
						  ([0, 0, 0, 0], [0, 0]), 
						  ([0, 0, 1, 0, 1, 0], [1, 0, 1, 0]), 
						  ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0]), 
						  ([0, 0, 0, 0, 0, 1], [0, 0, 0, 1])]
						]
					  
				  },

				  1:#rssnp index
				  {
				  	'neurons': 4,
					'synapses': 3,
					'rules': [
					    [0, 2, [1, 0], 1, 1, 0],
					    [1, 2, [1, 0], 1, 1, 0],
					    [2, 3, [2, 0], 2, 1, 0],
					    [2, 3, [1, 0], 1, 0, 0], 
					],
					'init_config': [0, 0, 0, 0],
					'rule_status': [-1, -1, -1, -1],
					'input_neurons': [0, 1],
					'output_neurons': [3],
				    'out_pairs':
				   	  	[		   	  	  	   	 
					      [([0, 0, 0, 0, 0, 1], [0, 0, 0, 1]), 
						  ([0, 0, 0, 0], [0, 0]), 
						  ([0, 0, 1, 0, 1, 0], [1, 0, 1, 0]), 
						  ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0]), 
						  ([0, 0, 0, 0, 0, 1], [0, 0, 0, 1])]
					  	]
				  }
					
	    	  }
	    	}
    	}

      }
    }
 }

    
def conf_load(filename):
	with open(filename, 'r') as stream:
		try:
			ga_params = yaml.safe_load(stream)

			#getting generation zero from run zero
			generation_zero = ga_params['runs'][0]['generations'][0]
			print(generation_zero)

			#getting rssnp index 0 in generation zero
			rssnp_zero = generation_zero['rssnp_chromosomes'][0]
			print(rssnp_zero) 

			#getting out_pairs indexed 0 at rssnp indexed 0 in generation zero
			out_pairs_zero = rssnp_zero['out_pairs'][0]
			print(out_pairs_zero)
		except yaml.YAMLError as exc:
			print(exc)

def conf_save(filename):
	with open(filename, 'w+') as out:
	  	doc = yaml.safe_dump(ga_params, out)

filename = 'ga_conf_out.yaml'
conf_save(filename)
conf_load(filename)




