from adarad.optimization.penalty import *
from adarad.optimization.constraint import *

# Objective function for static problem.
def dyn_stat_obj(d_var, h_var, patient_rx):
    K = d_var.shape[0]
    if h_var.shape[0] != K:
        raise ValueError("h_var must have exactly {0} rows".format(K))
    if patient_rx["dose_goal"].shape not in [(K,), (K,1)]:
        raise ValueError("dose_goal must have dimensions ({0},)".format(K))
    if patient_rx["health_goal"].shape not in [(K,), (K,1)]:
        raise ValueError("health_goal must have dimensions ({0},)".format(K))

    d_penalty = dose_penalty(d_var, patient_rx["dose_goal"], patient_rx["dose_weights"])
    h_penalty = health_penalty(h_var, patient_rx["health_goal"], patient_rx["health_weights"])
    return d_penalty + h_penalty

# Static optimal control problem in session t_static.
# OAR health status is linear-quadratic, while PTV health status is linear (beta = 0) with an optional slack term.
def build_stat_init_prob(A_list, alpha, beta, gamma, h_init, patient_rx, t_static=0, slack_oar_weight=1):
    T_treat = len(A_list)
    K, n = A_list[0].shape
    if h_init.shape[0] != K:
        raise ValueError("h_init must be a vector of {0} elements".format(K))
    if t_static < 0 or t_static >= T_treat or int(t_static) != t_static:
        raise ValueError("t_static must be an integer in [0, {0}]".format(T_treat - 1))

    # Extract parameters for session t_static.
    A_t = A_list[t_static]
    alpha_t = alpha[t_static]
    beta_t = beta[t_static]
    gamma_t = gamma[t_static]
    patient_rx_t = rx_slice(patient_rx, t_static, t_static + 1, squeeze=True)

    # Define variables.
    b = Variable((n,), nonneg=True, name="beams")  # Beams.
    d = A_t @ b  # Dose.

    # Health status after optimization.
    h_lin = h_init - multiply(alpha_t, d) + gamma_t  # Linear terms only.
    h_quad = h_lin - multiply(beta_t, square(d))  # Linear-quadratic dynamics.

    # Approximate PTV health status by linearizing around d = 0 (drop quadratic dose term).
    # Keep modeling OAR health status with linear-quadratic dynamics.
    is_target = patient_rx_t["is_target"]
    h_app_ptv = h_lin[is_target]
    h_app_oar = h_quad[~is_target]
    h_app = multiply(h_lin, is_target) + multiply(h_quad, ~is_target)

    # Objective function.
    d_penalty = dose_penalty(d, patient_rx_t["dose_goal"], patient_rx_t["dose_weights"])
    h_penalty_ptv = pos_penalty(h_app_ptv, patient_rx_t["health_goal"][is_target],
                                patient_rx_t["health_weights"][1][is_target])
    # h_penalty_oar = hinge_penalty(h_app_oar[~is_target], patient_rx_t["health_goal"][~is_target], [w[~is_target] for w in patient_rx_t["health_weights"]])
    h_penalty_oar = neg_penalty(h_app_oar, patient_rx_t["health_goal"][~is_target],
                                patient_rx_t["health_weights"][0][~is_target])
    h_penalty = h_penalty_ptv + h_penalty_oar
    obj_base = d_penalty + h_penalty

    # Additional beam constraints.
    constrs = []
    if "beam_constrs" in patient_rx and "upper" in patient_rx["beam_constrs"]:
        beam_upper = patient_rx["beam_constrs"]["upper"]
        if np.any(beam_upper < 0):
            raise ValueError("Beam upper bound must be nonnegative")
        constrs += constr_sum_upper(b, beam_upper, T_treat)

    # Additional dose constraints.
    if "dose_constrs" in patient_rx and "upper" in patient_rx["dose_constrs"]:
        dose_upper = patient_rx["dose_constrs"]["upper"]
        if np.any(dose_upper < 0):
            raise ValueError("Dose upper bound must be nonnegative")
        constrs += constr_sum_upper(d, dose_upper, T_treat)

    # Additional health constraints.
    if "health_constrs" in patient_rx:
        # Health bounds from final optimization session.
        patient_rx_fin = rx_slice(patient_rx, T_treat - 1, T_treat, squeeze=True)
        rx_fin_health_constrs_ptv = get_constrs_by_struct(patient_rx_fin["health_constrs"], is_target, struct_dim=0)
        rx_fin_health_constrs_oar = get_constrs_by_struct(patient_rx_fin["health_constrs"], ~is_target, struct_dim=0)

        # Add slack to lower bound on health of OARs, but keep strict upper bound on health of PTVs.
        constrs += rx_to_ptv_constrs(h_app_ptv, rx_fin_health_constrs_ptv)
        # constrs += rx_to_oar_constrs(h_app_oar, rx_fin_health_constrs_oar)
        h_slack = Variable((K,), nonneg=True)
        constrs += rx_to_oar_constrs(h_app_oar, rx_fin_health_constrs_oar, slack_lower=h_slack[~is_target])
        obj_slack = slack_oar_weight * sum(h_slack)
        # constr_oar, h_oar_slack = rx_to_oar_constrs_slack(h_app_oar, rx_fin_health_constrs_oar)
        # constrs += constr_oar
        # obj_slack = slack_oar_weight*sum(h_oar_slack)
    else:
        # h_oar_slack = Constant(np.zeros(h_app_oar.shape))
        h_slack = Constant(np.zeros((K,)))
        obj_slack = 0

    obj = obj_base + obj_slack
    prob = Problem(Minimize(obj), constrs)
    return prob, b, h_app, d, h_quad, h_slack
