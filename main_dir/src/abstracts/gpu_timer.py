import pycuda.driver as drv
import os

def timer_write(ga_name, exec_time):
    timer_out_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandextra22outreal.txt", "a+")
    timer_out_gpu.write(ga_name + " GPU time is " + str(exec_time) + "\n")
    timer_out_gpu.close()

def timer_write_run(run_index):
    timer_out_gpu = open(os.getcwd()+ "\\timer_directory\\gpuandextra22outreal.txt", "a+")
    timer_out_gpu.write(" Run index is " + str(run_index) + "\n")
    timer_out_gpu.close()

class GpuTimer(object):
    def __init__(self):
        self.tic_was_called = False
        self.toc_was_called = False
    def tic(self):
        self.tic_was_called = True
        self.start = drv.Event()
        self.end = drv.Event()
        self.start.record() # start timing
    def toc(self):
        self.end.record() # end timing
        self.toc_was_called = True
        # calculate the run length
        self.end.synchronize()
        self.secs = self.start.time_till(self.end)*1e-3 # [msec]-->[sec]
    def time(self):
        if self.tic_was_called and self.toc_was_called:
            return 'GpuTimer: secs = {0}'.format(self.secs)
        else:
            return 'Unused GpuTimer'
