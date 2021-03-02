import re, os
import yaml

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

class SNPGeneticAlgoEval:
    opt_fitness = 0
    max_fitness  = 0
    no_of_gen = 0
    no_of_run = 0
    # list_of_runs = []    
    # file_name = None
    # ga_params = None
    #path name is now loadfile_name
    def run(self, genetic_algo, system, size, function, runs, generations, mutation_rate, path_name, selection_func, start_new = True, start_from_gen = False):
        max_fitness_in_run = 0
        self.ga_params = conf_load(path_name)
        self.no_of_run = self.ga_params['run_total'] + self.ga_params['runs_pending']
        self.no_of_gen = self.ga_params['gen_total'] + self.ga_params['gens_pending']
        self.file_name = path_name
        #print("ga params in run is " + str(ga_params))
        if start_new == True:
            print("Starting anew")
            start = 0
            middle = self.ga_params['run_total']
        else:
            start = self.ga_params['run_total']
            middle = 0
        for run in range(start, start + middle + self.ga_params['runs_pending']):
            print("run run baby # " + str(run))
            # Perform a single run of the GA framework
            run_fitness = genetic_algo.simulate(system, size, function, generations, mutation_rate, path_name, run, selection_func, start_from_gen)            
            if  run_fitness > max_fitness_in_run:
                max_fitness_in_run = run_fitness
                if max_fitness_in_run >= self.ga_params['goal_fitness']:
                    break

        print("max fit is " + str(max_fitness_in_run))
        self.max_fitness = max_fitness_in_run
        self.ga_params = conf_load(path_name)
        self.ga_params['max_fitness_in_runs'] = max_fitness_in_run
        conf_save(path_name, self.ga_params)
 
        print("Likelihood of Evolution Leap: ", self.likelihoodOfEvolLeap())#, file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        print("Likelihood of Optimality: ", self.likelihoodOfOptimality())#, file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        print("Average Fitness Value: ", self.avgFitnessValue())#, file=open(path_name + "/Run" + str(run) + "/run.txt","a"))

    def likelihoodOfOptimality(self):
        no_of_opt_run = 0

        # Iterate through all runs
        for run in range(self.no_of_run):
            # Get highest fitness for that run
            #fitness = int(re.search('\'fitness\': (.*), \'out', run[-1]).group(1))
            fitness = self.ga_params['runs'][run]['max_fitness_in_run']
            # Increase number of optimal run if highest fitness is greater than the optimal fitness
            if fitness >= self.opt_fitness: no_of_opt_run += 1
        
        # Get average optimal runs
        return no_of_opt_run / self.no_of_run

    def avgFitnessValue(self):
        fitness_sum = 0
        # Iterate through all runs
        run_index = 0
        for run in range(self.no_of_run):
            # Get highest fitness for that run
            #fitness = int(re.search('\'fitness\': (.*), \'out', run[-1]).group(1))
            fitness = self.ga_params['runs'][run]['max_fitness_in_run']
            run_index += 1
            # Add highest fitness to the sum of fitness for all runs
            fitness_sum += fitness
        
        # Get average fitness 
        return fitness_sum / (self.no_of_run * self.max_fitness)

    def likelihoodOfEvolLeap(self):
        evol_leap = 0

        # Iterate through all runs
        for run in range(self.no_of_run):
            prev_fitness = 0
            leaps = 0
            # Iterate through every generation of a run
            for gen in range(self.no_of_gen):
                # Get fitness of the generation
                #fitness = int(re.search('\'fitness\': (.*), \'out', gen).group(1))
                fitness = self.ga_params['runs'][run]['max_fitness_in_run']
                # If the fitness is greater than the previous gen's fitness, increase number of leap for that run
                if fitness > prev_fitness: leaps += 1
                prev_fitness = fitness
            
            # Get summation of the average no of leaps of all runs
            evol_leap += (leaps / self.no_of_gen)
            print("evol leap is " + str(evol_leap))
        print('no of run is ' + str(self.no_of_run))
        # Return summatoon of the average no of leaps of all runs
        return evol_leap / self.no_of_run
