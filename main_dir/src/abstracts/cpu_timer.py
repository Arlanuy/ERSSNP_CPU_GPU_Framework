import os, yaml, sys

filename = os.path.join(os.getcwd(), 'timer_directory')
filename = os.path.join(filename, sys.argv[11])

timer_params = {}
timer_params['run_indexes'] = {}

def conf_save(filename, timer_params):
    with open(filename, 'w+') as out:
        doc = yaml.safe_dump(timer_params, out)

if not os.path.exists(filename):
    conf_save(filename, timer_params)

def conf_load(filename):
    with open(filename, 'r') as stream:
        try:
            timer_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return timer_params


run_index_state = 0

def timer_write(ga_name, start, finish):
	timer_params = conf_load(filename)
	timer_params['run_indexes'][run_index_state][ga_name] += finish - start
	conf_save(filename, timer_params)

def timer_write_run(run_index):
	timer_params = conf_load(filename)
	global run_index_state
	run_index_state = run_index
	timer_params['run_indexes'][run_index_state] = {}
	timer_params['run_indexes'][run_index_state]['Selection'] = 0
	timer_params['run_indexes'][run_index_state]['Crossover'] = 0
	timer_params['run_indexes'][run_index_state]['Evaluate'] = 0
	conf_save(filename, timer_params)

