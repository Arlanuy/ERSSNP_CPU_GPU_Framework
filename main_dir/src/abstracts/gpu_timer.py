import pycuda.driver as drv
import os, yaml

filename = os.path.join(os.getcwd(), 'timer_directory')
filename = os.path.join(filename, "gpuandminimal00outreal.yaml")
timer_params = {}
timer_params['run_indexes'] = {}

def conf_save(filename, timer_params):
    with open(filename, 'w+') as out:
        doc = yaml.safe_dump(timer_params, out)
        
if not os.path.exists(filename):
    conf_save(filename, timer_params)

run_index_state = 0

def conf_load(filename):
    with open(filename, 'r') as stream:
        try:
            timer_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return timer_params


def timer_write(ga_name, exec_time):
    timer_params = conf_load(filename)
    timer_params['run_indexes'][run_index_state][ga_name] += exec_time
    conf_save(filename, timer_params)

def timer_write_run(run_index):
    timer_params = conf_load(filename)
    global run_index_state
    run_index_state = run_index
    print("initialized at run ", run_index)
    timer_params['run_indexes'][run_index_state] = {}
    timer_params['run_indexes'][run_index_state]['Selection'] = 0
    timer_params['run_indexes'][run_index_state]['Crossover'] = 0
    timer_params['run_indexes'][run_index_state]['Evaluate'] = 0
    conf_save(filename, timer_params)


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
            return self.secs
        else:
            return 'Unused GpuTimer'
