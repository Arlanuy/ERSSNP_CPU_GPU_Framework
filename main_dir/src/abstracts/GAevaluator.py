import re, os

class SNPGeneticAlgoEval:
    opt_fitness = 0
    max_fitness = 0
    no_of_gen = 0
    no_of_run = 0
    list_of_runs = []

    def run(self, genetic_algo, system, size, function, generations, mutation_rate, results_dir, path_name, selection_func):
        path_name = results_dir + "/" + path_name
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        for run in range(self.no_of_run):
            # Create folder for each run of a GA (all generations)
            folder = path_name + "/Run" + str(run) + "/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            print("Run:",run)
            # Perform a single run of the GA framework
            genetic_algo.simulate(system, size, function, generations, mutation_rate, path_name, run, selection_func)
            self.updateRuns(path_name, run)

        # print("Likelihood of Evolution Leap: ", self.likelihoodOfEvolLeap(), file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        # print("Likelihood of Optimality: ", self.likelihoodOfOptimality(), file=open(path_name + "/Run" + str(run) + "/run.txt","a"))
        print("Average Fitness Value: ", self.avgFitnessValue(), file=open(path_name + "/Run" + str(run) + "/run.txt","a"))

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

        # Iterate through all runs
        for run in self.list_of_runs:
            # Get highest fitness for that run
            fitness = int(re.search('\'fitness\': (.*), \'out', run[-1]).group(1))
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

        # Return summatoon of the average no of leaps of all runs
        return evol_leap / self.no_of_run

    def updateRuns(self, file_name, run_index):
        array = []
        # Open a run
        with open(file_name + "/Run" + str(run_index) + "/run.txt", "r") as file:
            for line in file:
                # Get the highest fitness of every generation
                if line.startswith("0"):
                    array.append(line[2:])
        
        self.list_of_runs.append(array)