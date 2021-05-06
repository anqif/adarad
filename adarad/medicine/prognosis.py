import numpy as np
from adarad.utilities.data_utils import check_prog_parms

# Health prognosis with a given optimization.
def health_prog_quad(h_init, T, alpha = None, beta = None, gamma = None, doses = None, health_map = lambda h,d,t: h):
    K = h_init.shape[0]
    alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)

    h_prog = np.zeros((T+1,K))
    h_prog[0] = h_init
    for t in range(T):
        h_prog[t+1] = health_map(h_prog[t] - alpha[t]*doses[t] - beta[t]*doses[t]**2 + gamma[t], doses[t], t)
    return h_prog

def health_prog_lin(h_init, T, alpha = None, beta = None, gamma = None, doses = None, d_parm = None, health_map = lambda h,d,t: h):
    K = h_init.shape[0]
    alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)
    if d_parm is None:
        d_parm = np.zeros((T, K))

    h_prog = np.zeros((T + 1, K))
    h_prog[0] = h_init
    for t in range(T):
        h_prog[t+1] = health_map(h_prog[t] - alpha[t]*doses[t] - beta[t]*d_parm[t]*(2*doses[t] - d_parm[t]) + gamma[t], doses[t], t)
    return h_prog

def health_prog_act(h_init, T, alpha = None, beta = None, gamma = None, doses = None, is_target = None, health_map = lambda h,d,t: h):
    K = h_init.shape[0]
    alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)
    return health_prog_act_range(h_init, 0, T, alpha = alpha, beta = beta, gamma = gamma, doses = doses,
                                 is_target = is_target, health_map = health_map)

def health_prog_act_range(h_init, t_s, T, alpha = None, beta = None, gamma = None, doses = None, is_target = None, health_map = lambda h,d,t: h):
    K = h_init.shape[0]
    if t_s > T or t_s < 0:
        raise ValueError("t_s must be an integer in [0, {0}]".format(T))

    alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T, is_range = True)
    if is_target is None:
        is_target = np.full((K,), False)
    if is_target.shape not in [(K,), (K, 1)]:
        raise ValueError("is_target must have dimensions ({0},)".format(K))

    # Allow separate health maps for targets and organs-at-risk.
    if callable(health_map):
        health_map_ptv = health_map_oar = health_map   # TODO: Do we need to make a deep copy to avoid clashes?
    elif isinstance(health_map, dict):
        if "target" not in health_map:
            raise ValueError("health_map must contain key 'target'")
        if "organ" not in health_map:
            raise ValueError("health_map must contain key 'organ'")
        health_map_ptv = health_map["target"]
        health_map_oar = health_map["organ"]
    else:
        raise ValueError("health_map must be a function or a dictionary of functions")

    # h_prog[:, is_target] = health_prog_quad(h_init[is_target], T, alpha[:, is_target], beta[:, is_target],
    #									   gamma[:, is_target],
    #									   doses[:, is_target], d_parm[:, is_target], health_map)
    # h_prog[:, ~is_target] = health_prog_quad(h_init[~is_target], T, alpha[:, ~is_target], beta[:, ~is_target],
    #										 gamma[:, ~is_target],
    #										 doses[:, ~is_target], health_map)

    h_prog = np.zeros((T - t_s + 1, K))
    h_prog[0] = h_init
    h_idx = 0
    for t in range(t_s, T):
        h_quad_expr = h_prog[h_idx] - alpha[t]*doses[t] - beta[t]*doses[t]**2 + gamma[t]
        h_prog[h_idx+1,is_target] = health_map_ptv(h_quad_expr[is_target], doses[t,is_target], t)
        h_prog[h_idx+1,~is_target] = health_map_oar(h_quad_expr[~is_target], doses[t,~is_target], t)
        h_idx = h_idx + 1
    return h_prog

def health_prog_est(h_init, T, alpha = None, beta = None, gamma = None, doses = None, d_parm = None, is_target = None, health_map = lambda h,d,t: h):
    K = h_init.shape[0]
    alpha, beta, gamma, doses = check_prog_parms(alpha, beta, gamma, doses, K, T)
    if d_parm is None:
        d_parm = np.zeros((T, K))
    if is_target is None:
        is_target = np.full((K,), False)
    if is_target.shape not in [(K,), (K,1)]:
        raise ValueError("is_target must have dimensions ({0},)".format(K))

    # Allow separate health maps for targets and organs-at-risk.
    if callable(health_map):
        health_map_ptv = health_map_oar = health_map   # TODO: Do we need to make a deep copy to avoid clashes?
    elif isinstance(health_map, dict):
        if "target" not in health_map:
            raise ValueError("health_map must contain key 'target'")
        if "organ" not in health_map:
            raise ValueError("health_map must contain key 'organ'")
        health_map_ptv = health_map["target"]
        health_map_oar = health_map["organ"]
    else:
        raise ValueError("health_map must be a function or a dictionary of functions")

    h_prog = np.zeros((T + 1, K))
    h_prog[:,is_target] = health_prog_lin(h_init[is_target], T, alpha[:, is_target], beta[:, is_target], gamma[:, is_target],
                                          doses[:,is_target], d_parm[:,is_target], health_map_ptv)
    h_prog[:,~is_target] = health_prog_quad(h_init[~is_target], T, alpha[:, ~is_target], beta[:, ~is_target], gamma[:, ~is_target],
                                            doses[:,~is_target], health_map_oar)
    return h_prog
