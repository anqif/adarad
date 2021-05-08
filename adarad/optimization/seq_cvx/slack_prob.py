from cvxpy import *

from adarad.optimization.seq_cvx.dyn_prob import dyn_quad_obj
from adarad.optimization.penalty import rx_to_slack_quad_penalty
from adarad.optimization.constraint import rx_to_constrs

# Construct optimal control problem with slack health/dose constraints.
def build_dyn_slack_quad_prob(A_list, alpha, beta, gamma, h_init, patient_rx, T_recov = 0, use_h_dyn_slack = False,
                              h_dyn_slack_weight = 0, h_bnd_slack_weights = 1):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))

    # Main variables.
    b = Variable((T_treat, n), nonneg=True, name="beams")  # Beams.
    h = Variable((T_treat + 1, K), name="health")  # Health statuses.
    d = vstack([A_list[t] @ b[t] for t in range(T_treat)])  # Doses.
    d_parm = Parameter(d.shape, nonneg=True, name="dose parameter")  # Dose point around which to linearize dynamics.

    # Objective function.
    obj = dyn_quad_obj(d, h, patient_rx)

    # Health dynamics for optimization stage.
    h_lin = h[:-1] - multiply(alpha, d) + gamma[:T_treat]
    h_quad = h_lin - multiply(beta, square(d))
    h_taylor = h_lin - multiply(multiply(beta, d_parm), 2*d - d_parm)

    # Allow slack in health dynamics constraints.
    h_dyn_slack = Constant(0)
    if use_h_dyn_slack:
        h_dyn_slack = Variable((T_treat, K), nonneg=True, name="health dynamics slack")
        obj += h_dyn_slack_weight * sum(h_dyn_slack)  # TODO: Set slack weight relative to overall health penalty.
        h_taylor -= h_dyn_slack

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

    # Additional health constraints go into objective as a penalty.
    if "health_constrs" in patient_rx:
        obj += rx_to_slack_quad_penalty(h[1:], patient_rx["health_constrs"], patient_rx["is_target"], h_bnd_slack_weights)

    # Health dynamics for recovery stage.
    # TODO: Should we return h_r or calculate it later?
    if T_recov > 0:
        gamma_r = gamma[T_treat:]

        h_r = Variable((T_recov, K), name="recovery")
        constrs_r = [h_r[0] == h[-1] + gamma_r[0]]
        for t in range(T_recov - 1):
            constrs_r.append(h_r[t + 1] == h_r[t] + gamma_r[t + 1])

        # Additional health constraints during recovery go into objective as a penalty.
        if "recov_constrs" in patient_rx:
            obj += rx_to_slack_quad_penalty(h_r, patient_rx["recov_constrs"], patient_rx["is_target"], h_bnd_slack_weights)
        constrs += constrs_r

    # Final problem.
    prob = Problem(Minimize(obj), constrs)
    return prob, b, h, d, d_parm, h_dyn_slack
