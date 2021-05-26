import itertools
import numpy as np
from copy import deepcopy
from greenery.lego import parse
from random import randint

def egcd(a, b):
    '''
        Performs extended Euclidean Algorithm to find the GCF of 2 numbers 
    '''
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    '''
        Calculates the modular inverse of a modulo m
    '''
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

class RSSNPSystem:
    """

    Definition to represent an RSSNP System
    with m rules, n neruons, and l synapses.

    Attributes
    ----------
    configuration: numpy int array
        The Configuration Vector CF = [cf_1, ..., c_i, ..., cf_m]
        where
            c_i is the amount of spikes in neuron i at time t

    rule: numpy tuple array rl_j = (x_j, y_j, E_j, c_j, p_j, d_j)
        The current Rule Representation RL = [rl_1, ..., rl_j, ..., rl_m]
        where
            x_j is the source neuron
            y_j is the destination neuron
            E_j is the regular expression
            c_j is the amount of spikes to be consumed
            p_j is the amount of spikes to be produced
            d_j is the delay

    ruleStatus: numpy int array
        The Rule Status Vector RS = [rs_1, ..., rs_j, ..., rs_m]
        where
            rs_j =  t'_j,   if rl_j is active and will send spikes at t + t'_j;
                    -1,     otherwise

    systemState: numpy tuple pair
        The System State SS = (configuration, ruleStatus)
        where
            configuration is the Configuration Vector
            ruleStatus is the Rule Status Vector

    synapseStatus: numpy int array
        The Synapse Status Vector SY = [sy_1, ..., sy_h, ..., sy_l]
        where
            sy_h =  1,  if syn_h is open at time t
                    0,  otherwise

    applicability: numpy int array
        The Applicability Vector AP = [ap_1, ..., ap_j, ..., ap_m]
        where
            ap_j =  1, if rl_j is applicable
                    0, otherwise

    activation: numpy int array
        The Activation Vector AC = [ac_1, ..., ac_j, ..., ac_m]
        where
            ac_j =  1, if rl_j will be applied
                    0, otherwise

    loss: numpy int array
        The Loss Vector LS = [ls_1, ..., ls_i, ..., ls_n]
        where
            ls_i =  c_j, amount of spikes to be consumed for neuron i
                    0, otherwise
    """

    l = 0
    m = 0  # Number of Rules
    n = 0  # Number of Neurons
    rule = []  # Rules in the System
    configuration = []
    configuration_init = []
    ruleStatus = []
    synapseDict = {}  # Dictionary of Synapse
    inputs = []
    outputs = None  # neuron which contains the output of the system (environment) / must not fire rules
    in_spiketrain = []  # input spike trains (Example: {'index': 0, 'input': '010101'})
    out_spiketrain = []

    def __repr__(self):
        return str(self.rule) + " | " + str(self.configuration_init)

    def initializeSynapseDict(self):
        """

            Initializes the dictionary to be used in synMap

        """

        ctr = 0
        for j in range(0, self.m):
            rule = self.rule[j]
            # Adds a new item in the dict if pair doesn't exist
            if (rule[0], rule[1]) not in self.synapseDict:
                self.synapseDict[(rule[0], rule[1])] = ctr
                ctr += 1

    def synMap(self, x_j, y_j):
        """

            Maps the pair (x_j, y_j) to its respective synapse

            Returns
            -------
            int
                The synapse number containing the pair

        """

        return self.synapseDict[(x_j, y_j)]

    def synapseStat(self, ruleStatus):
        """

            Computes for the status of the synapses

            Returns
            -------
            numpy vector
                The Synapse Status Vector at time t

        """

        # Initialized the synapseStatus vector
        synapseStatus = np.full(self.l, 1, dtype=int)
        for j in range(0, self.m):
            rule = self.rule[j]
            # Maps the pair (x_j, y_j) to its respective synapse
            h = self.synMap(rule[0], rule[1])
            # Change the status of a synapse if it contains an active rule
            if ruleStatus[j] >= 0:
                synapseStatus[h] = 0

        return synapseStatus

    def ruleApplicability(self, configuration, synapseStatus):
        """

            Determines which rules are applicable

            Returns
            -------
            numpy vector
                The Applicability Vector at time t

        """
        #print("config",configuration)
        # Initializes the applicability vector
        applicability = np.zeros(self.m, dtype=int)
        for j in range(0, self.m):
            rule = self.rule[j]
            # Maps the pair (x_j, y_j) to its respective synapse
            h = self.synMap(rule[0], rule[1])
            # Checks if the rule can be applied
            if synapseStatus[h] == 1 and configuration[rule[0]] >= rule[3]:
                if rule[2][1] == 0 and configuration[rule[0]] == rule[2][0]:
                    applicability[j] = 1
                elif rule[2][1] != 0 and (
                        configuration[rule[0]] - rule[2][0]) % rule[2][1] == 0:
                    applicability[j] = 1

        return applicability

    def activationVectors(self, applicability):
        """

            Determines which rules will be activated per time step

            Returns
            -------
            list
                List containing the Activation Vectors at time t

        """
        #print("app =",applicability)
        activationMatrix = self.synapseRestriction(applicability)
        finalActivationMatrix = self.costRestriction(activationMatrix)

        return finalActivationMatrix

    def synapseRestriction(self, applicability):
        """

            Determines which rules will be activated considering the synapse restriction

            Returns
            -------
            list
                List containing the temporary Activation Vectors at time t

        """

        SA = []
        for h in range(self.l):
            SA.append([])

        for j in range(0, self.m):
            if applicability[j] == 1:
                rule = self.rule[j]
                h = self.synMap(rule[0], rule[1])
                SA[h].append(j + 1)

        for h in range(0, self.l):
            if len(SA[h]) == 0:
                SA[h].append(0)

        SC = []
        for elem in itertools.product(*SA):
            SC.append(elem)

        activationMatrix = []
        for elem in SC:
            activation = [0] * self.m
            for h in range(0, self.l):
                j = elem[h]
                if j != 0:
                    activation[j - 1] = 1

            activationMatrix.append(activation)
            #print("AM = ",activationMatrix)
        return activationMatrix

    def costRestriction(self, activationMatrix):
        """

            Determines which rules will be activated considering all restrictions

            Returns
            -------
            list
                List containing the final Activation Vectors at time t

        """

        finalActivationMatrix = []

        for activation in activationMatrix:
            DC = []

            for i in range(0, self.n):
                DC.append([])

            for j in range(0, self.m):
                if activation[j] == 1:
                    DC[self.rule[j][0]].append(self.rule[j][3])

            for i in range(0, self.n):
                if len(DC[i]) == 0:
                    DC[i].append(0)
            #print("DC = ",DC)
            CC = []
            for elem in itertools.product(*DC):
                CC.append(elem)
            #print("CC = ", CC)
            finalActivation = activation

            for elem in CC:
                #print("elem ",elem)
                #print("rule[3] elem[rule[0]]")
                for j in range(0, self.m):
                    rule = self.rule[j]
                    #print("j=",j)
                    #print(rule[3],"         ",elem[rule[0]])
                    if rule[3] > elem[rule[0]]:
                        finalActivation[j] = 0

            finalActivationMatrix.append(finalActivation)
            #print("FAM = ",finalActivationMatrix)
        return finalActivationMatrix

    def applyRules(self, activationMatrix, ruleStatus):
        """


        """
        lossMatrix = []
        ruleStatusMatrix = []

        for activation in activationMatrix:
            ls = [0 for i in range(0, self.n)]
            rs = [0 for i in range(0, self.m)]

            for j in range(0, self.m):
                rs[j] = ruleStatus[j]
                if activation[j] == 1:
                    rule = self.rule[j]
                    ls[rule[0]] = rule[3]
                    rs[j] = rule[5]
            lossMatrix.append(ls)
            ruleStatusMatrix.append(rs)

        return ruleStatusMatrix, lossMatrix

    def nextState(self, configuration, ruleStatusMatrix, lossMatrix):
        configurationMatrix = []
        nextRuleStatusMatrix = []
        for row in range(0, len(ruleStatusMatrix)):
            cf = []
            for i in range(0, self.n):
                cf.append(configuration[i] - lossMatrix[row][i])

            ruleStatus = ruleStatusMatrix[row]
            for j in range(0, self.m):
                if ruleStatusMatrix[row][j] == 0:
                    y_j = self.rule[j][1]
                    p_j = self.rule[j][4]
                    cf[y_j] += p_j
                if ruleStatusMatrix[row][j] >= 0:
                    ruleStatusMatrix[row][j] -= 1

            configurationMatrix.append(cf)
            nextRuleStatusMatrix.append(ruleStatus)
        return configurationMatrix, nextRuleStatusMatrix

    def main(self, initSystemState, maxSteps):
        unexploredStates = [initSystemState]
        exploredStates = []
        step = 0

        # Get length of longest input spike train
        max_input_bits = 0
        for input_neuron in self.in_spiketrain:
            if max_input_bits < len(input_neuron['input']):
                max_input_bits = len(input_neuron['input'])
        #print("max input bits is ", max_input_bits)
        self.initializeSynapseDict()
        
        from_input_state = False
        while unexploredStates != [] and step <= 3*maxSteps:
            systemState = unexploredStates.pop(0)

            self.configuration = systemState[0]
            self.ruleStatus = systemState[1]

            synapseStatus = self.synapseStat(self.ruleStatus)
            applicability = self.ruleApplicability(self.configuration, synapseStatus)
            activation = self.activationVectors(applicability)
            ruleStatusMatrix, lossMatrix = self.applyRules(activation, self.ruleStatus)
            configurationMatrix, nextRuleStatusMatrix = self.nextState(self.configuration, ruleStatusMatrix, lossMatrix)

            if from_input_state:
                from_input_state = False
            else:
                exploredStates.append(systemState)

            for row in range(0, len(ruleStatusMatrix)):
                nextSystemState = (configurationMatrix[row],nextRuleStatusMatrix[row])
                self.out_spiketrain.append(nextSystemState[0][self.outputs])
                nextSystemState[0][self.outputs] = 0
                if step <= max_input_bits:
                    # Add input spike train to neurons
                    for input_neuron in self.in_spiketrain:
                        nextSystemState[0][input_neuron['index']] += input_neuron['input'][step] if step < len(input_neuron['input']) else 0
                    unexploredStates.append(nextSystemState)
                    from_input_state = True
                elif not (nextSystemState in (unexploredStates + exploredStates)):
                    unexploredStates.append(nextSystemState)
            step += 1
           
        #return self.out_spiketrain[0:max_input_bits]
        return self.out_spiketrain

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
            if self.rule[i][0] == self.outputs or self.rule[i][1] == self.outputs:
                continue

            # Small chance to delete rule
            if randint(0, mutation_rate-1) == 0:
                deleted_rules.append(i)
                continue

            # Change connected synapses
            if randint(0, mutation_rate-1) == 0:
                # Cannot change input and output neurons connection
                self.rule[i][randint(0, 1)] = randint(0, self.n - 1)
                while self.rule[i][0] == self.outputs:  # Cannot have the environment fire spikes
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

    def intersectionCheck(self, i, j):
        # Same synapse
        if self.rule[i][1] == self.rule[j][1]:
            return False
        # Different synapse, but same source neuron with varying consumed spikes
        elif self.rule[i][3] != self.rule[j][3]:
            return False
        else:
            return True

    def checkDeterministic(self):
        # delete duplicate rules because for some reason the code below doesnt fix this problem
        deleted_rules = []
        for i in range(0, self.m - 1):
            for j in range(i + 1, self.m):
                if self.rule[i] == self.rule[j] and not j in deleted_rules:    # identical rules
                    deleted_rules.append(j)
        delete_ctr = 0
        for index in deleted_rules:
            self.rule.pop(index - delete_ctr)
            self.m -= 1
            delete_ctr += 1

        for i in range(0, self.m - 1):
            for j in range(i + 1, self.m):
                # disregard if from different source neurons
                if self.rule[i][0] != self.rule[j][0]: continue
                
                # value will intersect if no kleene star
                if self.rule[i][2][0] == self.rule[j][2][0]:
                    if not self.intersectionCheck(i,j):
                        return False
                # intersection exists
                if str(parse("a{" + str(self.rule[i][2][0]) + "}(a{" + str(self.rule[i][2][1]) + "})*") & parse("a{" + str(self.rule[j][2][0]) + "}(a{" + str(self.rule[j][2][1]) + "})*")) != '[]':                   
                    if not self.intersectionCheck(i, j):
                        return False

                # e_i = self.rule[i][2]
                # e_j = self.rule[j][2]

                # if e_i[1] == 0 and e_j[1] == 0:
                #     if e_i[0] == e_j[0]:
                #         if not self.intersectionCheck(i, j):
                #             return False
                # elif e_i[1] == 0:
                #     # Intersection
                #     if (e_i[0] - e_j[0]) / e_j[1] % 1 == 0:
                #         if not self.intersectionCheck(i, j):
                #             return False
                # elif e_j[1] == 0:
                #     # Intersection
                #     if (e_j[0] - e_i[0]) / e_i[1] % 1 == 0:
                #         if not self.intersectionCheck(i, j):
                #             return False
                # else:
                #     try:
                #         c1 = (e_j[0] - e_i[0]) % e_i[1] * modinv(e_j[1], e_i[1])
                #         c2 = (e_i[1] * c1 - (e_j[0] - e_i[0])) / e_j[1]

                #         # If regular expression of both rules have an intersection
                #         if c1 % 1 == 0 and c2 % 1 == 0:
                #             if not self.intersectionCheck(i, j):
                #                 return False
                #     except Exception:  # Modular inverse does not exist
                #         # value will intersect if j = 0
                #         if self.rule[i][2][0] == self.rule[j][2][0]:
                #             return False
                #         # intersection exists
                #         if str(parse("a{" + str(self.rule[i][2][0]) + "}(a{" + str(self.rule[i][2][1]) + "})*") & parse("a{" + str(self.rule[j][2][0]) + "}(a{" + str(self.rule[j][2][1]) + "})*")) != '[]':
                #             return False
                #         return True
        return True

    def checkPath(self, unexploredEdges, edges):
        exploredEdges = []

        while unexploredEdges != []:
            curr_edge = unexploredEdges.pop(0)
            
            if curr_edge[1] == self.outputs:
                return True
            
            for index, edge in enumerate(edges):
                if edge[0] == curr_edge[1]:
                    unexploredEdges.append(edge)
                    edges.pop(index)
            
            exploredEdges.append(curr_edge)
            
        return False

    def checkPathInToOut(self):
        for num in self.inputs:
            edges = deepcopy(self.rule)
            unexploredEdges = []
            
            # Add all edges connected to the input node
            for index, edge in enumerate(edges):
                if edge[0] == num:
                    unexploredEdges.append(edge)
                    edges.pop(index)
            
            if not self.checkPath(unexploredEdges, edges):
                return False
        
        return True

    def isValid(self):
        if self.checkPathInToOut() and self.checkDeterministic():
            return True
        return False

def assign_rssnp(rssnp):
    # initialize the RSSNP system
    system = RSSNPSystem()

    system.n = rssnp['neurons']    # number of neurons
    system.l = rssnp['synapses']   # number of synapses
    system.m = len(rssnp['rules'])    # number of rules

    system.rule = rssnp['rules']

    system.configuration_init = rssnp['init_config']    # starting configuration (number of spikes per neuron)
    system.ruleStatus = rssnp['rule_status']            # initial status of each rule (set to -1)
    system.inputs = rssnp['input_neurons']
    system.outputs = rssnp['output_neuron']

    return system
 