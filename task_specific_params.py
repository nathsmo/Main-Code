from collections import namedtuple

# task specific params
TaskTSP = namedtuple('TaskTSP', ['task_name', 
						'input_dim', 
						'n_nodes',
						'decode_len'])
TaskVRP = namedtuple('TaskVRP', ['task_name', 
						'input_dim',
						'n_nodes' ,
						'n_cust',
						'decode_len',
						'capacity',
						'demand_max'])


task_lst = {}

# TSP10
tsp10 = TaskTSP(task_name = 'tsp',
			  input_dim=2,
			  n_nodes = 10,
			  decode_len=10)
task_lst['tsp10'] = tsp10

# TSP20
tsp20 = TaskTSP(task_name = 'tsp',
			  input_dim=2,
			  n_nodes = 20,
			  decode_len=20)
task_lst['tsp20'] = tsp20

# TSP50
tsp50 = TaskTSP(task_name = 'tsp',
			  input_dim=2,
			  n_nodes = 50,
			  decode_len=50)
task_lst['tsp50'] = tsp50

# TSP100
tsp100 = TaskTSP(task_name = 'tsp',
			  input_dim=2,
			  n_nodes = 100,
			  decode_len=100)
task_lst['tsp100'] = tsp100
