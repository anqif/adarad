import numpy as np
import cvxpy.settings as cvxpy_s

class RunProfile(object):
    def __init__(self, use_admm=False, use_slack=False, slack_weight="default"):
        self.use_admm = use_admm
        self.use_slack = use_slack
        self.slack_weight = slack_weight

class RunOutput(object):
    def __init__(self):
        self.status = None
        self.objective = np.nan
        self.optimal_variables = {'beams': None, 'doses': None, 'health': None, 'slacks': None}
        self.residuals = {'primal': None, 'dual': None}

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

    @property
    def slacks(self):
        return self.optimal_variables["slacks"]

    @property
    def primal_residuals(self):
        return self.residuals["primal"]

    @property
    def dual_residuals(self):
        return self.residuals["dual"]

class SolverStats(object):
    def __init__(self):
        # self.solver_name = None
        self.total_time = None
        self.solve_time = None
        # self.setup_time = None
        self.num_iters = None
        # self.extra_stats = None

class RunRecord(object):
    def __init__(self, use_admm=False, use_slack=False, slack_weight='default'):
        self.profile = RunProfile(use_admm=use_admm, use_slack=use_slack, slack_weight=slack_weight)
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

    @property
    def slacks(self):
        return self.output.slacks
