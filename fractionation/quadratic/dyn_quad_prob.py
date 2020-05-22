import cvxpy
import numpy as np
from cvxpy import *

# Construct optimal control problem.
def build_dyn_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov=0):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Define variables.
    b = Variable((T_treat, n), nonneg=True, name="beams")   # Beams.
    h = Variable((T_treat + 1, K), name="health")           # Health statuses.
    d = vstack([A_list[t] * b[t] for t in range(T_treat)])  # Doses.
    d_parm = Parameter(d.shape, nonneg=True, name="dose parameter")  # Dose point around which to linearize target dynamics.

    # Objective function.
    obj = dyn_quad_obj(d, h, patient_rx)

    # Health dynamics for treatment stage.
    h_lin = h[:-1] - multiply(alpha, d) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d))
    h_taylor = h_lin - multiply(multiply(beta, d_parm), 2*d - d_parm)

    constrs = [h[0] == h_init]
    for t in range(T_treat):
        # For PTV, approximate dynamics via a first-order Taylor expansion.
        constrs.append(h[t+1, patient_rx["is_target"]] == h_taylor[t, patient_rx["is_target"]])

        # For OAR, relax dynamics constraint to an upper bound that is always tight at optimum.
        constrs.append(h[t+1, ~patient_rx["is_target"]] <= h_quad[t, ~patient_rx["is_target"]])

    # Additional beam constraints.
    if "beam_constrs" in patient_rx:
        constrs += rx_to_constrs(b, patient_rx["beam_constrs"])

    # Additional dose constraints.
    if "dose_constrs" in patient_rx:
        constrs += rx_to_constrs(d, patient_rx["dose_constrs"])

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        constrs += rx_to_quad_constrs(h[1:], patient_rx["health_constrs"])

    # Health dynamics for recovery stage.
    # TODO: Should we return h_r or calculate it later?
    if T_recov > 0:
        gamma_r = gamma[T_treat:]

        h_r = Variable((T_recov, K), name="recovery")
        constrs_r = [h_r[0] == h[-1] + gamma_r[0]]
        for t in range(T_recov - 1):
            constrs_r.append(h_r[t + 1] == h_r[t] + gamma_r[t + 1])

        # Additional health constraints during recovery.
        if "recov_constrs" in patient_rx:
            constrs_r += rx_to_quad_constrs(h_r, patient_rx["recov_constrs"])
        constrs += constrs_r

    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d, d_parm