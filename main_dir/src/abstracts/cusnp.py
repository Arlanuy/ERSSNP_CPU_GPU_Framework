import math
import numpy as np
from itertools import combinations, permutations

class SnpSystem:
    """

    Definition to represent an SNP System having
    m neurons and n rules.

    Attributes
    ----------
    config : numpy int array
        The Configuration Vector C = {c1,...,cm}
        where ci is the amount of spikes in neuron
        i at time k
    spiking : numpy int array
        A Spiking Vector S = {s1,...,sn} where
            si = 1 if Ei is satisfied and ri is applied;
                 0 otherwise
    status : numpy int array
        The Status Vector at time k St = {st1,...,stm}
        where for each i in [1, m]
            sti = 1 if neuron m is open;
                  0 if neuron m is closed
    ruleRepresent : numpy tuple array ri = (E,j,d',c,o)
        The current Rule Representation R = {r1,...,rn}
        where
            E is the regular expression for rule i,
            j is the neuron that contains the rule,
            d' determines if the rule is fired or not,
                or if it is on delay
            c is the number of spikes consumed in j if
                r is applied
            o is the number of spikes produced by j if
                r is applied
    delays : numpy int array
        The Delay Vector D = {d1,...,dn} contains the delay
        count for each rule in the system
    loss : numpy int array
        The Loss Vector LV = {lv1,lv2,...,lvm} where for i
        in [1, m], lvi is the number of consumed spikes
        in neuron i at step k
    gain : numpy int array
        The Gain Vector GV = {gv1,gv2,...,gvm} where
        for each i in [1, m], gv is the number of spikes
        received by neuron i at step k
    transitionMatrix : ordered set of vectors
        The Transition Matrix TV = {tv1,...,tvm} where for 
        each i in [1, n], tvi = {p1,...,pm} st if rule i is
        in neuron s
            pj = number of spikes produced by rule i, 
                    if s and j are connected;
                 0 otherwise
    indicator : numpy int array
        The Indicator Vector IV = {iv1,...,ivm} indicates
        which rule will produce spikes at time k
    removingMatrix : list of vectors
        The Removing Matrix RM = {rm1,rm2,...,rmn} where
        for each i in [1, n], rmi = {t1,...,tm} st if rule
        i is in neuron s
            tj = number of spikes consumed by rule i 
                    if s = j;
                 0 otherwise
    netGain : numpy int array
        The Net Gain Vector is defined as NG = C(k+1) - C(k)
    neuronType: nump int array
        The Neuron Type NT = {nt1,...,ntm} indicates
        the type of the neuron such that
            nt1 = -1 if output neuron
                  1 if input neuron
                  0 otherwise

    """
    config = np.zeros( (1,10), dtype=int)               # Configuration Vector
    spiking = None                                      # Spiking Vector
    status = np.zeros( (1,10), dtype=int)               # Status Vector
    ruleRepresent = np.zeros((3,10), dtype=int)         # Rule Representation 
    delays = np.zeros( (1,10), dtype=int)               # Delay Vector
    loss = None                                         # Loss Vector
    gain = None                                         # Gain Vector
    transitionMatrix = np.zeros( (3,10), dtype=int)     # Tranistion Matrix
    indicator = None                                    # Indicator Vector
    removingMatrix = np.zeros( (3,10), dtype=int)       # Removing Matrix
    netGain = None                                      # Net Gain Vector
    neuronType = np.zeros( (1,10), dtype=int)           # Neuron Type Vector

    def neuronCount(self):
        """

        Return the number of neurons in the system

        Returns
        -------
        int
            The number of neurons in the system

        """
        return len(self.config)
    
    def ruleCount(self):
        """

        Return the number of rules in the system

        Returns
        -------
        int
            The number of rules in the system

        """
        return len(self.ruleRepresent)

    def generateValidSpikingVectors(self):
        """

        Return a list of valid spiking vectors

        Returns
        -------
        list of vectors
            A list of spiking vectors with no
            conflicting rules

        """
        neurons_of_rules = [rule[1] for rule in self.ruleRepresent]
        combi = []
        div = -1
        
        for rule in neurons_of_rules:
            if div != rule:
                combi.append(bin(2**(neurons_of_rules.count(rule)-1))[2:])
                div = rule

        validSpikingVectors = []

        for string in combi:
            validSpikingVectors.append(list(set(permutations([int(x) for x in string],len(string)))))

        return validSpikingVectors
            

    def computeSpikingVector(self, validSpikingVectors):
        """

        Computes for the Spiking Vector at time k

        Returns
        -------
        numpy vector
            The Spiking Vector of size n at time k

        """
        # Initialize Spiking Vector
        self.spiking = np.zeros(self.ruleCount(), dtype=int)

        i = 0
        for rule in self.ruleRepresent:
            if self.status[rule[1]] == 0:
                self.spiking[i] = 0                            # Neuron that owns the rule is closed
            else:
                if rule[0] == self.config[rule[1]]:            # Regex matches
                    self.spiking[i] = 1                        # Rule E is satisfied in C(k)
                else:
                    self.spiking[i] = 0                        # Rule E did not match with C(k)
            i += 1

        return self.spiking      

    def getInput(self):
        """

        Updates the Configuration Vector
        with user input

        """
        for i in range(0,self.neuronCount()):
            if self.neuronType[i] == 1:
                self.config[i] += int(input("Enter input for neuron "+str(i)+": "))

    def getOutput(self, outType="sum"):
        """
        
        Return the output of the system to
        the environment at time k

        Returns
        -------
        int or string
            The total number or concatenation
            of spikes fired by all the output
            neurons at time k

        """
        if outType == "sum":
            output = 0  # default output is 0 obviously
        elif outType == "concat":
            output = [str(0) for i in self.neuronType if self.neuronType[i] == -1] # default output is empty
            output_index = 0 # index of output string
        else:
            return None

        i = 0   # index counter for rules
        for rule in self.ruleRepresent:
            # rule fires and neuron is an output neuron
            if self.indicator[i] == 1 and self.neuronType[rule[1]] == -1:
                if outType == "sum":
                    output += rule[4]
                elif outType == "concat":
                    output[output_index] = str(rule[4])
                    output_index += 1
            i += 1

        if outType == "concat":
            output = ''.join(output)

        return output

    def simulateSNP(self):
        """

        Main simulation algorithm for CuSNP

        Returns
        -------
        numpy vector
            The next Configuration Vector (time k+1)

        """
        # Reset LV, GV, NG and IV (set to zero vectors)
        self.loss = np.zeros(self.neuronCount(), dtype=int)
        self.gain = np.zeros(self.neuronCount(), dtype=int)
        self.netGain = np.zeros(self.neuronCount(), dtype=int)
        self.indicator = np.zeros(self.ruleCount(), dtype=int)

        i = 0   # index counter for S
        for rule in self.ruleRepresent:
            if self.spiking[i] == 1:                # Case 1
                rule[2] = self.delays[i]
                self.indicator[i] = 0
                self.status[rule[1]] = 0
                if rule[2] == 0:                    # Case 3
                    self.indicator[i] = 1
                    self.status[rule[1]] = 1
            elif rule[2] == 0:                      # Case 2
                self.indicator[i] = 1
                self.status[rule[1]] = 1
            i += 1

        # Calculate the loss vector
        self.loss = self.spiking.reshape(1,self.ruleCount()) @ self.removingMatrix
        
        self.gain = self.indicator.reshape(1,self.ruleCount()) @ self.transitionMatrix
        self.netGain = (self.status * self.gain) - self.loss
        
        # Next configuration vector
        newC = (self.config + self.netGain).reshape(self.neuronCount())

        # Countdown
        for rule in self.ruleRepresent:
            if rule[2] != -1:
                rule[2] -= 1

        return newC
