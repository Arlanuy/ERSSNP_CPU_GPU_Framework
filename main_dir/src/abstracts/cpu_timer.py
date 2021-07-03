import os

name = "cpuandminimal00outreal.txt"

def timer_write(ga_name, start, finish):
    timer_out_cpu = open(os.getcwd()+ "\\timer_directory\\" + name, "a+")
    timer_out_cpu.write(ga_name + " CPU time is " + str(finish - start) + "\n")

def timer_write_run(run_index):
    timer_out_cpu = open(os.getcwd()+ "\\timer_directory\\" + name, "a+")
    timer_out_cpu.write("Run index is " + str(run_index) + "\n")
    timer_out_cpu.close()

