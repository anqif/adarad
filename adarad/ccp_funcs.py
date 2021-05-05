import numpy as np
from collections import Counter
from time import time

import cvxpy.settings as cvxpy_s
from cvxpy import Constant

def ccp_solve(prob, d, d_parm, d_init = None, h_slack = Constant(0), ccp_verbose = False, full_hist = False, *args, **kwargs):
	if d_init is None:
		d_init = np.zeros(d_parm.shape)

	# Problem parameters.
	max_iter = kwargs.pop("max_iter", 50)  # Maximum iterations.
	ccp_eps = kwargs.pop("ccp_eps", 1e-3)  # Stopping tolerance.
	iter_print = np.maximum(max_iter // 10, 1)

	# Validate parameters.
	if max_iter <= 0:
		raise ValueError("max_iter must be a positive integer.")
	if ccp_eps < 0:
		raise ValueError("ccp_eps must be a non-negative scalar.")
	
	k = 0
	setup_time = 0
	solve_time = 0
	finished = False
	obj_cur = np.inf
	d_cur = d_init
	dose_list = []
	status_list = []
	h_slack_sum = np.zeros(max_iter)

	start = time()
	while not finished:
		if ccp_verbose and k % iter_print == 0:
			print("Iteration:", k)

		# Solve linearized problem.
		d_parm.value = d_cur
		prob.solve(*args, **kwargs)
		if prob.status not in cvxpy_s.SOLUTION_PRESENT:
			raise RuntimeError("Solver failed with status {0}".format(prob.status))
		setup_time += 0 if prob.solver_stats.setup_time is None else prob.solver_stats.setup_time
		solve_time += prob.solver_stats.solve_time
		status_list.append(prob.status)

		# Save entire history of doses.
		if full_hist:
			dose_list.append(d.value.copy())
		h_slack_sum[k] = np.sum(h_slack.value)

		# Check stopping criterion.
		obj_diff = obj_cur - prob.value
		finished = (k + 1) >= max_iter or obj_diff <= ccp_eps

		# Update linearization point and objective.
		d_cur = d.value
		obj_cur = prob.value
		k = k + 1
	end = time()

	# Take majority as final status.
	status, status_count = Counter(status_list).most_common(1)[0]
	doses = dose_list if full_hist else d.value
	return {"obj": obj_cur, "status": status, "num_iters": k, "total_time": end - start, "setup_time": setup_time,
			"solve_time": solve_time, "doses": doses, "health_slacks": np.array(h_slack_sum[:k])}
