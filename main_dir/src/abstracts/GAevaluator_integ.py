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
    list_of_runs = []    
    file_name = None
    #path name is now loadfile_name
    def run(self, genetic_algo, system, size, function, runs, generations, mutation_rate, path_name, selection_func, start_new = True):
        max_fitness_in_run = 0
        self.no_of_run = runs 
        self.no_of_gen = generations
        ga_params = conf_load(path_name)
        #print("ga params in run is " + str(ga_params))
        if start_new == True:
            start = 0
        else:
            start = ga_params['gen_total']  - 1
        for run in range(start, start + self.no_of_run):
            print("run run baby # " + str(run))
            # Perform a single run of the GA framework
            run_fitness = genetic_algo.simulate(system, size, function, self.no_of_gen, mutation_rate, path_name, run, selection_func, start_new)
            self.updateRuns(path_name, run)

            if  run_fitness > max_fitness_in_run:
                max_fitness_in_run = run_fitness

        self.max_fitness = max_fitness_in_run
        ga_params = conf_load(self.file_name)
        ga_params['max_fitness_in_runs'] = max_fitness_in_run
        conf_save(self.file_name, ga_params)
 
        # print("Likelihood of Evolution Leap: ", self.likelihoodOfEvolLeap(), file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        # print("Likelihood of Optimality: ", self.likelihoodOfOptimality(), file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        print("Average Fitness Value: ", self.avgFitnessValue())#, file=open(path_name + "/Run" + str(run) + "/run.txt","a"))

    def likelihoodOfOptimality(self):
        no_of_opt_run = 0

        # Iterate through all runs
        for run in self.list_of_runs:
            # Get highest fitness for that run
            fitness = int(re.search('\'fitness\': (.*), \'out', run[-1]).group(1))
            # Increase number of optimal run if highest fitness is greater than the optimal fitness
            if fitness >= self.opt_fitness: no_of_opt_run += 1
        
        # Get average optimal runs
        return no_of_opt_run / self.no_of_run

    def avgFitnessValue(self):
        fitness_sum = 0
        ga_params = conf_load(self.file_name)
        # Iterate through all runs
        run_index = 0
        for run in range(self.no_of_run):
            # Get highest fitness for that run
            #fitness = int(re.search('\'fitness\': (.*), \'out', run[-1]).group(1))
            fitness = ga_params['runs'][run]['max_fitness_in_run']
            run_index += 1
            # Add highest fitness to the sum of fitness for all runs
            fitness_sum += fitness
        
        # Get average fitness 
        return fitness_sum / (self.no_of_run * self.max_fitness)

    def likelihoodOfEvolLeap(self):
        evol_leap = 0

        # Iterate through all runs
        for run in self.list_of_runs:
            prev_fitness = 0
            leaps = 0
            # Iterate through every generation of a run
            for gen in run:
                # Get fitness of the generation
                fitness = int(re.search('\'fitness\': (.*), \'out', gen).group(1))
                # If the fitness is greater than the previous gen's fitness, increase number of leap for that run
                if fitness > prev_fitness: leaps += 1
                prev_fitness = fitness
            
            # Get summation of the average no of leaps of all runs
            evol_leap += (leaps / self.no_of_gen)
            print("evol leap is " + evol_leap)

        # Return summatoon of the average no of leaps of all runs
        return evol_leap / self.no_of_run

    def updateRuns(self, file_name, run_index):
        array = []
        # Open a run
        ga_params = conf_load(file_name)
        self.file_name = file_name
        array.append(ga_params["runs"][run_index]["generations"])
        self.list_of_runs.append(array)