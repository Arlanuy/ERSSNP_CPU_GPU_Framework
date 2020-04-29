   '''
    def randomize(self, mutation_rate=2):
        """
        
            Randomly decrease the resources of the system

            Returns
            -------
            RSSNP System
                An RSSNP system with mutated variables

        """
        deleted_rules = []
        for i in range(0, self.m):
            # Do not change anything about input neurons
            if self.rule[i][0] in self.inputs or self.rule[i][1] in self.inputs:
                continue

            # Do not change the output neuron
            if self.rule[i][0] in self.outputs or self.rule[i][1] in self.outputs:
                continue

            # Small chance to delete rule
            if randint(0, mutation_rate-1) == 0:
                deleted_rules.append(i)
                continue

            # Change connected synapses
            if randint(0, mutation_rate-1) == 0:
                # Cannot change input and output neurons connection
                self.rule[i][randint(0, 1)] = randint(0, self.n - 1)
                while self.rule[i][0] in self.outputs:  # Cannot have the environment fire spikes
                    self.rule[i][randint(0, 1)] = randint(0, self.n - 1)

                # cannot have synapse pointing to the same neuron
                while self.rule[i][0] == self.rule[i][1]:
                    self.rule[i][randint(0, 1)] = randint(0, self.n - 1)

            # Modify regular expression
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][2] = (randint(1, self.rule[i][2][0]), self.rule[i][2][1])
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][2] = (self.rule[i][2][0], randint(0, self.rule[i][2][1]))

            # Consumed spikes must be within regular expression
            if self.rule[i][2][0] < self.rule[i][3] and self.rule[i][2][1] == 0:
                self.rule[i][3] = self.rule[i][2][0]

            # Change consumed spikes
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][3] = randint(1, self.rule[i][3])

            # Change produced spikes
            if randint(0, mutation_rate-1) == 0:
                self.rule[i][4] = randint(0, self.rule[i][4])

            # Cannot produce more spikes than consumed
            if self.rule[i][4] > self.rule[i][3]:
                self.rule[i][4] = self.rule[i][3]

        delete_counter = 0
        for index in deleted_rules:
            self.rule.pop(index - delete_counter)
            self.m -= 1
            delete_counter += 1
            
        # Change initial configuration
        for i in range(0, self.n):
            if randint(0,mutation_rate-1) == 0:
                self.configuration_init[i] = randint(0, self.configuration_init[i])

        return self
'''
    __global__ int randomize(Rule *rule_mat, int population, muation_rate, int* deleted_rules){

        int tidx = threadIdx.x + blockIdx.x * blockDim.x;
        printf("source %d %d %d %d\\n", r1[tidx],r2[tidx],p1[tidx],p2[tidx]);
        int i = t->size_rssnp[p1[tidx]]+r1[tidx];

        # Do not change anything about input neurons
        if self.rule[i][0] in self.inputs or self.rule[i][1] in self.inputs{
            return 0;
        }

        # Do not change the output neuron
        if self.rule[i][0] in self.outputs or self.rule[i][1] in self.outputs {
            return 0;
        }

        # Small chance to delete rule
        if randint(0, mutation_rate-1) == 0 {
            deleted_rules[i] = 1;
        }
        #I am stuck translating this
        # Change connected synapses
        if randint(0, mutation_rate-1) == 0{
            # Cannot change input and output neurons connection
            rule_mat[i][randint(0, 1)] = randint(0, self.n - 1)
            while self.rule[i][0] in self.outputs:  # Cannot have the environment fire spikes
                self.rule[i][randint(0, 1)] = randint(0, self.n - 1)

            # cannot have synapse pointing to the same neuron
            while self.rule[i][0] == self.rule[i][1]:
                self.rule[i][randint(0, 1)] = randint(0, self.n - 1)
        }

        # Modify regular expression
        if randint(0, mutation_rate-1) == 0:
            self.rule[i][2] = (randint(1, self.rule[i][2][0]), self.rule[i][2][1])
        if randint(0, mutation_rate-1) == 0:
            self.rule[i][2] = (self.rule[i][2][0], randint(0, self.rule[i][2][1]))

        # Consumed spikes must be within regular expression
        if self.rule[i][2][0] < self.rule[i][3] and self.rule[i][2][1] == 0:
            self.rule[i][3] = self.rule[i][2][0]

        # Change consumed spikes
        if randint(0, mutation_rate-1) == 0:
            self.rule[i][3] = randint(1, self.rule[i][3])

        # Change produced spikes
        if randint(0, mutation_rate-1) == 0:
            self.rule[i][4] = randint(0, self.rule[i][4])

        # Cannot produce more spikes than consumed
        if self.rule[i][4] > self.rule[i][3]:
            self.rule[i][4] = self.rule[i][3]

    delete_counter = 0
    for index in deleted_rules:
        self.rule.pop(index - delete_counter)
        self.m -= 1
        delete_counter += 1
        
    # Change initial configuration
    for i in range(0, self.n):
        if randint(0,mutation_rate-1) == 0:
            self.configuration_init[i] = randint(0, self.configuration_init[i])

    }

    void main() {
        int population = sizeof(size_rssnp)/sizeof(int);
        dim3 dimBlock(population, population, 1);    
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock3(population/2, population/2, 1);    
        dim3 dimGrid3(1, 1, 1);
        int* deleted_rules = malloc(population * sizeof(int));
        mutation_rate = 2; #this means 20% since 1/2 = 50%

        randomize<<<dimGrid3,  dimBlock3>>>(rule_mat, population, mutation_rate, deleted_rules);

    }