import numpy as np
import cvxpy.settings as cvxpy_s

class RunProfile(object):
    def __init__(self, use_admm=False, slack_weight="default"):
        self.use_admm = use_admm
        self.slack_weight = slack_weight

class RunOutput(object):
    def __init__(self):
        self.status = None
        self.objective = np.nan
        self.optimal_variables = {'beams': None, 'doses': None, 'health': None}

    @property
    def feasible(self):
        return self.status in cvxpy_s.SOLUTION_PRESENT

    @property
    def beams(self):
        return self.optimal_variables["beams"]

    @property
    def doses(self):
        return self.optimal_variables["doses"]

    @property
    def health(self):
        return self.optimal_variables["health"]

class SolverStats(object):
    def __init__(self):
        # self.solver_name = None
        self.solve_time = None
        # self.setup_time = None
        self.num_iters = None
        # self.extra_stats = None

class RunRecord(object):
    def __init__(self, use_admm=False, slack_weight='default'):
        self.profile = RunProfile(use_admm=use_admm, slack_weight=slack_weight)
        self.output = RunOutput()
        self.solver_stats = SolverStats()

    @property
    def feasible(self):
        return self.output.feasible

    @property
    def status(self):
        return self.output.status

    @property
    def beams(self):
        return self.output.beams

    @property
    def doses(self):
        return self.output.doses

    @property
    def health(self):
        return self.output.health
