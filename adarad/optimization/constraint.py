import cvxpy
import numpy as np
from cvxpy import *

# Select constraint bound matrices by structure index.
def get_constrs_by_struct(constrs, struct_idx, struct_dim = 1):
    if struct_dim not in [0, 1]:
        raise ValueError("struct_dim must be either 0 or 1")

    constrs_s = {}
    for key in constrs.keys():
        if np.isscalar(constrs[key]):
            constrs_s[key] = constrs[key]
        else:
            arr = constrs[key].copy()
            constrs_s[key] = arr[struct_idx] if struct_dim == 0 else arr[:,struct_idx]
    return constrs_s

def rx_slice(patient_rx, t_start, t_end, t_step=1, squeeze=True):
    t_slice = slice(t_start, t_end, t_step)

    rx_cur = patient_rx.copy()
    for goal_key in {"dose_goal", "health_goal"}:
        if goal_key in patient_rx:
            rx_old_slice = patient_rx[goal_key][t_slice]
            if squeeze:
                rx_old_slice = np.squeeze(rx_old_slice)
            rx_cur[goal_key] = rx_old_slice

    for constr_key in {"beam_constrs", "dose_constrs", "health_constrs"}:
        if constr_key in patient_rx:
            rx_cur[constr_key] = {}
            for lu_key in {"lower", "upper"}:
                if lu_key in patient_rx[constr_key]:
                    # rx_old_slice = patient_rx[constr_key][lu_key][t_slice]
                    constr_val = patient_rx[constr_key][lu_key]
                    if np.isscalar(constr_val):
                        rx_old_slice = constr_val
                    else:
                        rx_old_slice = constr_val[t_slice]
                        if squeeze:
                            rx_old_slice = np.squeeze(rx_old_slice)
                    rx_cur[constr_key][lu_key] = rx_old_slice
    return rx_cur

# Extract lower bound constraints from patient prescription.
def rx_to_lower_constrs(expr, rx_lower, only_oar = False, slack = 0):
    if np.any(rx_lower == np.inf):
        raise ValueError("Lower bound cannot be infinity")

    if np.isscalar(rx_lower):
        if np.isfinite(rx_lower):
            return expr >= rx_lower - slack
    else:
        if rx_lower.shape != expr.shape:
            bnd_str = "rx_lower" if only_oar else "rx_lower of non-targets"
            raise ValueError("{0} must have dimensions {1}".format(bnd_str, expr.shape))
        is_finite = np.isfinite(rx_lower)
        if np.any(is_finite):
            is_slack_scalar = slack.is_scalar() if isinstance(slack, Expression) else np.isscalar(slack)
            if is_slack_scalar:
                return expr[is_finite] >= rx_lower[is_finite] - slack
            else:
                if slack.shape != expr.shape:
                    raise ValueError("slack must be a scalar or array of dimensions".format(expr.shape))
                return expr[is_finite] >= rx_lower[is_finite] - slack[is_finite]
    return

# Extract upper bound constraints from patient prescription.
def rx_to_upper_constrs(expr, rx_upper, only_ptv = False, slack = 0):
    if np.any(rx_upper == -np.inf):
        raise ValueError("Upper bound cannot be negative infinity")

    if np.isscalar(rx_upper):
        if np.isfinite(rx_upper):
            return expr <= rx_upper + slack
    else:
        if rx_upper.shape != expr.shape:
            bnd_str = "rx_upper" if only_ptv else "rx_upper of targets"
            raise ValueError("{0} must have dimensions {1}".format(bnd_str, expr.shape))
        is_finite = np.isfinite(rx_upper)
        if np.any(is_finite):
            is_slack_scalar = slack.is_scalar() if isinstance(slack, Expression) else np.isscalar(slack)
            if is_slack_scalar:
                return expr[is_finite] <= rx_upper[is_finite] + slack
            else:
                if slack.shape != expr.shape:
                    raise ValueError("slack must be a scalar or array of dimensions".format(expr.shape))
                return expr[is_finite] <= rx_upper[is_finite] + slack[is_finite]
    return

# Extract constraints from patient prescription.
def rx_to_constrs(expr, rx_dict):
    constrs = []

    # Lower bound.
    if "lower" in rx_dict:
        c_lower = rx_to_lower_constrs(expr, rx_dict["lower"])
        if c_lower is not None:
            constrs.append(c_lower)

    # Upper bound.
    if "upper" in rx_dict:
        c_upper = rx_to_upper_constrs(expr, rx_dict["upper"])
        if c_upper is not None:
            constrs.append(c_upper)
    return constrs

# Extract constraints from patient prescription.
def rx_to_quad_constrs(expr, rx_dict, is_target, struct_dim = 1, slack_lower = 0, slack_upper = 0):
    if struct_dim not in [0, 1]:
        raise ValueError("struct_dim must be either 0 or 1")
    is_slack_lo_scalar = slack_lower.is_scalar() if isinstance(slack_lower, Expression) else np.isscalar(slack_lower)
    is_slack_hi_scalar = slack_upper.is_scalar() if isinstance(slack_upper, Expression) else np.isscalar(slack_upper)
    constrs = []

    # Lower bound.
    if "lower" in rx_dict:
        # rx_lower = rx_dict["lower"]
        if struct_dim == 0:
            rx_lower_ptv = rx_dict["lower"][is_target]
            rx_lower_oar = rx_dict["lower"][~is_target]
            slack_lower_oar = slack_lower if is_slack_lo_scalar else slack_lower[~is_target]
            expr_oar = expr[~is_target]
        else:
            rx_lower_ptv = rx_dict["lower"][:,is_target]
            rx_lower_oar = rx_dict["lower"][:,~is_target]
            slack_lower_oar = slack_lower if is_slack_lo_scalar else slack_lower[:,~is_target]
            expr_oar = expr[:,~is_target]

        if not np.all(np.isneginf(rx_lower_ptv)):
            raise ValueError("Lower bound must be negative infinity for all targets")

        c_lower = rx_to_lower_constrs(expr_oar, rx_lower_oar, only_oar = True, slack = slack_lower_oar)
        if c_lower is not None:
            constrs.append(c_lower)

    # Upper bound.
    if "upper" in rx_dict:
        # rx_upper = rx_dict["upper"]
        if struct_dim == 0:
            rx_upper_ptv = rx_dict["upper"][is_target]
            rx_upper_oar = rx_dict["upper"][~is_target]
            slack_upper_ptv = slack_upper if is_slack_hi_scalar else slack_upper[is_target]
            expr_ptv = expr[is_target]
        else:
            rx_upper_ptv = rx_dict["upper"][:,is_target]
            rx_upper_oar = rx_dict["upper"][:,~is_target]
            slack_upper_ptv = slack_upper if is_slack_hi_scalar else slack_upper[:,is_target]
            expr_ptv = expr[:,is_target]

        if not np.all(np.isinf(rx_upper_oar)):
            raise ValueError("Upper bound must be infinity for all non-targets")

        c_upper = rx_to_upper_constrs(expr_ptv, rx_upper_ptv, only_ptv = True, slack = slack_upper_ptv)
        if c_upper is not None:
            constrs.append(c_upper)
    return constrs

# Health bound constraints for PTV.
def rx_to_ptv_constrs(h_ptv, rx_dict_ptv, slack_upper = 0):
    constrs = []

    # Lower bound.
    if "lower" in rx_dict_ptv and not np.all(np.isneginf(rx_dict_ptv["lower"])):
        raise ValueError("Lower bound must be negative infinity for all targets")

    # Upper bound.
    if "upper" in rx_dict_ptv:
        c_upper = rx_to_upper_constrs(h_ptv, rx_dict_ptv["upper"], only_ptv = True, slack = slack_upper)
        if c_upper is not None:
            constrs.append(c_upper)
    return constrs

# Health bound constraints for OAR.
def rx_to_oar_constrs(h_oar, rx_dict_oar, slack_lower = 0):
    constrs = []

    # Lower bound.
    if "lower" in rx_dict_oar:
        c_lower = rx_to_lower_constrs(h_oar, rx_dict_oar["lower"], only_oar = True, slack = slack_lower)
        if c_lower is not None:
            constrs.append(c_lower)

    # Upper bound.
    if "upper" in rx_dict_oar and not np.all(np.isinf(rx_dict_oar["upper"])):
        raise ValueError("Upper bound must be infinity for all non-targets")
    return constrs

# def rx_to_oar_constrs_slack(h_oar, rx_dict_oar):
#     if "upper" in rx_dict_oar and not np.all(np.isinf(rx_dict_oar["upper"])):
#         raise ValueError("Upper bound must be infinity for all non-targets")
#
#     constrs = []
#     h_slack = Constant(0)
#     if "lower" in rx_dict_oar:
#         rx_lower = rx_dict_oar["lower"]
#         if np.any(rx_lower == np.inf):
#             raise ValueError("Lower bound cannot be infinity")
#
#         if np.isscalar(rx_lower):
#             if np.isfinite(rx_lower):
#                 h_slack = Variable(h_oar.shape, nonneg=True, name="OAR health lower bound slack")
#                 constrs.append(h_oar >= rx_lower - h_slack)
#         else:
#             if rx_lower.shape != h_oar.shape:
#                 raise ValueError("rx_lower must have dimensions {0}".format(h_oar.shape))
#             is_finite = np.isfinite(rx_lower)
#             if np.any(is_finite):
#                 h_slack = Variable(h_oar.shape, nonneg=True, name="OAR health lower bound slack")
#                 constrs.append(h_oar[is_finite] >= rx_lower[is_finite] - h_slack[is_finite])
#     return constrs, h_slack

def constr_sum_upper(expr, upper, T_treat):
    n = expr.shape[0]
    if np.isscalar(upper):
        if np.isfinite(upper):
            return [expr <= T_treat * upper]
        else:
            return []
    elif upper.shape == (T_treat, n):
        upper_sum = np.sum(upper, axis = 0)
        is_finite = np.isfinite(upper_sum)
        if np.any(is_finite):
            return [expr[is_finite] <= upper_sum[is_finite]]
        else:
            return []
    else:
        raise TypeError("Upper bound must be a scalar or array with dimensions ({0},{1})".format(T_treat, n))
