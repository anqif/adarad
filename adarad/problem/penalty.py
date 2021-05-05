import cvxpy
import numpy as np
from cvxpy import *
from collections import defaultdict

# Sum-of-squares penalty function.
def square_penalty(var, goal=None, weights=None):
    if goal is None:
        goal = np.zeros(var.shape)
    if weights is None:
        weights = np.ones(var.shape)
    if np.any(weights < 0):
        raise ValueError("weights must all be nonnegative")
    # if weights.shape != var.shape:
    #    raise ValueError("weights must be of size {0}".format(var.shape))
    return weights @ square(var - goal)

# Hinge penalty function.
def hinge_penalty(var, goal=None, weights=None):
    if goal is None:
        goal = np.zeros(var.shape)
    if weights is None:
        weights = [np.ones(var.shape), np.ones(var.shape)]
    if len(weights) != 2:
        raise ValueError("weights must be a list of two arrays")
    for w in weights:
        if np.any(w < 0):
            raise ValueError("weights must all be nonnegative")
        # if w.shape != var.shape:
        #    raise ValueError("weights must be a list of objects of size {0}".format(var.shape))

    w_under, w_over = weights
    return w_under @ neg(var - goal) + w_over @ pos(var - goal)

# Penalty functions.
dose_penalty = square_penalty
health_penalty = hinge_penalty

# Convert constraints to slack penalty.
def rx_to_slack_quad_penalty(expr, rx_dict, is_target, weights = None):
    T, K = expr.shape
    if weights is None:
        weights = np.ones((T,K))
    elif np.isscalar(weights):
        weights = np.full((T,K), weights)
    elif weights.shape != (T,K):
        raise ValueError("weights must have dimensions ({0},{1})".format(T,K))
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    penalty = 0
    # Lower bound.
    if "lower" in rx_dict:
        rx_lower = rx_dict["lower"][:,~is_target]
        expr_oar = expr[:,~is_target]
        w_lower = weights[:,~is_target]

        if np.any(rx_lower == np.inf):
            raise ValueError("Lower bound cannot be infinity")
        if np.isscalar(rx_lower):
            if np.isfinite(rx_lower):
                penalty += sum(multiply(w_lower, pos(rx_lower - expr_oar)))
        else:
            if rx_lower.shape != expr_oar.shape:
                raise ValueError("rx_lower must have dimensions {0}".format(expr_oar.shape))
            is_finite = np.isfinite(rx_lower)
            if np.any(is_finite):
                penalty += sum(multiply(w_lower[is_finite], pos(rx_lower[is_finite] - expr_oar[is_finite])))

    # Upper bound.
    if "upper" in rx_dict:
        rx_upper = rx_dict["upper"][:,is_target]
        expr_ptv = expr[:,is_target]
        w_upper = weights[:,is_target]

        if np.any(rx_upper == -np.inf):
            raise ValueError("Upper bound cannot be negative infinity")

        if np.isscalar(rx_upper):
            if np.isfinite(rx_upper):
                penalty += sum(multiply(w_upper, pos(expr_ptv - rx_upper)))
        else:
            if rx_upper.shape != expr_ptv.shape:
                raise ValueError("rx_upper must have dimensions {0}".format(expr_ptv.shape))
            is_finite = np.isfinite(rx_upper)
            if np.any(is_finite):
                penalty += sum(multiply(w_upper[is_finite], pos(expr_ptv[is_finite] - rx_upper[is_finite])))
    return penalty
