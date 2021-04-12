import numpy as np
from warnings import warn
from fractionation.medicine.physics import BeamSet

def load_dose_matrix(data, T, K):
    if "dose_matrix" not in data:
        warn("Case profile must include dose_matrix")
        return None
    A_list = np.load(data["dose_matrix"])

    if isinstance(A_list, np.ndarray):
        if A_list.shape[0] != K:
            raise ValueError("Dose matrix must have same number of rows as structures")
        A_list = T * [A_list]
    elif isinstance(A_list, list):
        if len(A_list) != T:
            raise ValueError("List of dose matrices must equal treatment length")

        m, n = A_list[0].shape
        for A in A_list:
            if A.shape != (m, n):
                raise ValueError("All dose matrices must have the same dimensions")
            if A.shape[0] != K:
                raise ValueError("All dose matrices must have same number of rows as structures")
    else:
        raise ValueError("Dose matrix file must load a list or numpy array")
    return A_list

def load_beams(data, T, n):
    if "beams" in data:
        # Beam positions.
        beam_args = dict()
        if "angles" in data["beams"]:
            if isinstance(data["beams"]["angles"], str):
                beam_args["angles"] = np.load(data["beams"]["angles"])
            else:
                beam_args["angles"] = data["beams"]["angles"]
        if "bundles" in data["beams"]:
            beam_args["bundles"] = data["beams"]["bundles"]
        if "offset" in data["beams"]:
            beam_args["offset"] = data["beams"]["offset"]
        beams = BeamSet(**beam_args) if beam_args else None

        # Beam intensity lower bound.
        lower = data["beams"].get("lower_bound", 0)
        if np.isscalar(lower):
            # lower_mat = np.full((T, n), lower)
            lower_bnd = lower
        else:
            if isinstance(lower, np.ndarray):
                lower_bnd = lower
            elif isinstance(lower, str):
                lower_bnd = np.load(lower)
            else:
                raise ValueError("Beam upper bound must be a scalar, array, or path string")
            if lower_bnd.shape != (T, n):
                raise ValueError("Beam lower bound matrix must have dimensions ({0},{1})".format(T, n))

        # Beam intensity upper bound.
        upper = data["beams"].get("upper_bound", np.inf)
        if np.isscalar(upper):
            # upper_mat = np.full((T, n), upper)
            upper_bnd = upper
        else:
            if isinstance(upper, np.ndarray):
                upper_bnd = upper
            elif isinstance(upper, str):
                upper_bnd = np.load(upper)
            else:
                raise ValueError("Beam upper bound must be a scalar, array, or path string")
            if upper_bnd.shape != (T, n):
                raise ValueError("Beam upper bound matrix must have dimensions ({0},{1})".format(T, n))
    else:
        beams = None
        lower_bnd = 0
        upper_bnd = np.inf

    return beams, lower_bnd, upper_bnd
