# ERSSNP_CPU_GPU_Framework
An evolving spiking neural P systems with rules on synapses framework in GPU 
-COARE Edition Environment

To run this program, in the COARE Interface of DOST, we use a Putty and private/public keys in order to connect to the network such as in https://www.ssh.com/academy/ssh/putty/windows/puttygen

There are multiple Bash scripts(can only be used within COARE) and some Python scripts (can only be used outside of COARE for analyzing the produced output of COARE) that serves as a user interface for this project and we run them in the COARE server via sbatch candminsave.bash

1.) Anatomy of a BASH Filename (applicable to all Bash scripts here in main_dir)

A. c		B. and		C. min		D. save	E. bash

A. is categorized into c and g. C denotes that it is the CPU of COARE that have done the computation, whereas, G denotes that it is the GPU units  of COARE instead 

B. and denotes the AND gate RSSP. Other forms of RSSNP are OR Gate, NOT Gate, ADD Module and SUB module

C. min characterizes the topology of the RSSNP as to whether it is minimal, adversarial or with extra rules.

D. save characterizes that the yaml file being produced is just a savefile starting from scratch whereas load denotes that the yaml file being produced is a loadfile from an existing savefile.

E. .bash denotes that the script can only be executed in a Bash environment which the COARE LINUX Slurm understands


2.)  coare_cpu_timer_adder_for_yaml_format.py


CPU Counterpart that serves as a dependent time analyzer program from the output of main.py with filename as specified in Step 1 of part 3 above. You can execute this program via the command python coare_cpu_timer_adder_for_yaml_format.py. The corresponding output of this is cpu_time_test.txt.  The output will then have four metrics as its output namely: Average selection time,  Average evaluate time, Average crossover time, and Average total time. 

3.)  coare_gpu_timer_adder_for_yaml_format.py

GPU counterpart

Due to the intermittent nature of some of the run by which we executed 2gandminload.bash and 2gorminload.bash even after executing the original bash gand/orminload bash scripts. Even after getting the output time_test.txt, we have to compute, as of the moment, the average of the result being produced from the intermittent runs manually via a digital calculator. The computation of which is done in the file calc_gpu_time_average_result.txt

4.)  coare_cpu_yaml_averager.py

This serves as a dependent YAML analyzer program from the savefile of main.py with filename as specified by the user in the console queries of Step 2 of part 3 above. You can execute this program via the command python yaml_averager.py. For purpose of simplicity, tutorial_time boolean is set to true to only include the test file and for this tutorial to work. To specify the output filename of this savefile just replace the filename that was used in the first line of yaml_averager.py. The output will then have three metrics as its output namely: Highest fitness achieved,  Average fitness achieved, and Average number of rules. 

5.)  coare_gpu_revised_averager.py

GPU counterpart

Due to the intermittent nature of some of the run by which we executed 2gandminload.bash and 2gorminload.bash even after executing the original bash gand/orminload bash scripts. Even after getting the output data_test.txt, we have to compute, as of the moment, the average of the result being produced from the intermittent runs manually via a digital calculator. The computation of which is done in the file calc_gpu_data_average_result.txt
