# ERSSNP_CPU_GPU_Framework
An evolving spiking neural P systems with rules on synapses framework in GPU

To run this program, Python 3 is a must
Also, a version of CUDA > Version 9 is needed and so nvcc command in the command line must work 
If nvcc command doesn't work try installing a (Microsoft Visual Studio) compatible to the CUDA version you have.
After all of this, just run pip install -r requirements.txt inside main_dir and you are good to go.

There are five Python programs that serves as a user interface for this project

1.) cpu_simulate_function.py

This serves as a stand-alone RSSNP simulator wherein given certain inputs like the RSSNP configuration and maxsteps which is described in the RSSNP paper of (Cabarle et.al ) we print the output spikes of the environment neuron in the console. You can run this program via typing in the command line: python cpu_simulate_function.py

2.) draw_function.py
This serves as a stand-alone drawing program of a certain input RSSNP configuration. It outputs a figure in PNG format of the RSSNP as provided in the input.

By RSSNP configuration from 1 and 2, what we mean is the RSSNPs as listed in src/RSSNP_list.py


3.) main.py

This serves as a sole entry point to the main gist of the project

Step 1
From the src/abstracts directory, to specify a record file for the time execution. Look for the files gpu_timer.py and cpu_timer.py in order to replace the default given value of the variable "name" to the desired name of the file

Step 2
Run the program via command line with python main.py in the main directory 

Step 3
For the sample inputs towards saving an creating an initial experiment involving AND minimal gate, please refer to the file default_answer_to_save

Step 4
For the sample inputs towards loading from an experiment involving AND minimal gate, please refer to the file default_answer_to_load

4.)  timer_adder.py

This serves as a dependent time analyzer program from the output of main.py with filename as specified in Step 1 of part 3 above. You can execute this program via the command python timer_adder.py. For purpose of simplicity, tutorial_time boolean is set to true to only include the test file and for this tutorial to work. To specify the output filename of this text just replace the filename that was used in the first line of timer_adder.py.  The output will then have four metrics as its output namely: Average selection time,  Average evaluate time, Average crossover time, and Average total time. 

5.)  yaml_averager.py

This serves as a dependent YAML analyzer program from the savefile of main.py with filename as specified by the user in the console queries of Step 2 of part 3 above. You can execute this program via the command python yaml_averager.py. For purpose of simplicity, tutorial_time boolean is set to true to only include the test file and for this tutorial to work. To specify the output filename of this savefile just replace the filename that was used in the first line of yaml_averager.py. The output will then have three metrics as its output namely: Highest fitness achieved,  Average fitness achieved, and Average number of rules. 