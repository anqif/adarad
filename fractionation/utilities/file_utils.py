import numpy as np
import yaml
import pickle
from io import open

def yaml_to_dict(path):
	with open(path, "r") as stream:
		try:
			data = yaml.safe_load(stream)
		except yaml.YAMLError as err:
			print(err)
	
	T = data["treatment_length"]
	K = len(data["structures"])
	A_list, visuals = load_bio_mats(data)
	n = A_list[0].shape[1]
	
	# Patient biology.
	bio = dict()
	bio["dose_matrices"] = A_list
	bio["alpha"] = np.zeros((T,K))
	bio["beta"] = np.zeros((T,K))
	bio["gamma"] = np.zeros((T,K))
	bio["health_init"] = np.zeros((K,))
	
	# Treatment parameters and constraints.
	rx = dict()
	rx["is_target"] = np.full((K,), True)
	rx["dose_goal"] = np.zeros((T,K))
	rx["dose_weights"] = np.ones((K,))
	rx["dose_constrs"] = {"lower": np.zeros((T,K)), "upper": np.full((T,K), np.inf)}
	
	rx["health_goal"] = np.zeros((T,K))
	rx["health_weights"] = [np.ones((K,)), np.ones((K,))]
	rx["health_constrs"] = {"lower": np.full((T,K), -np.inf), "upper": np.full((T,K), np.inf)}
	rx["beam_constrs"] = load_beam_mats(data, T, n)

	for i in range(K):
		struct = data["structures"][i]
		
		bio["alpha"][:,i] = struct["alpha"]
		bio["beta"][:,i] = struct["beta"]
		bio["gamma"][:,i] = struct["gamma"]
		bio["health_init"][i] = struct["health"]["initial"]
		
		rx["is_target"][i] = struct["is_target"]
		rx["dose_goal"][:,i] = struct["dose"].get("goal", 0)
		rx["dose_weights"][i] = struct["dose"].get("weight", 1)
		rx["dose_constrs"]["lower"][:,i] = struct["dose"].get("lower_bound", 0)
		rx["dose_constrs"]["upper"][:,i] = struct["dose"].get("upper_bound", np.inf)
		
		rx["health_goal"][i] = struct["health"]["goal"]
		if struct["is_target"]:
			rx["health_weights"][0][i] = struct["health"].get("under", 0)
			rx["health_weights"][1][i] = struct["health"].get("over", 1)
			rx["health_constrs"]["upper"][:,i] = struct["health"].get("upper_bound", np.inf)
		else:
			rx["health_weights"][0][i] = struct["health"].get("under", 1)
			rx["health_weights"][1][i] = struct["health"].get("over", 0)
			rx["health_constrs"]["lower"][:,i] = struct["health"].get("lower_bound", -np.inf)

	return bio, rx, visuals

def load_bio_mats(data):
	T = data["treatment_length"]
	K = len(data["structures"])
	
	# Load dose matrices.
	A_list = np.load(data["dose_matrix"])
	if isinstance(A_list, np.ndarray):
		if A_list.shape[0] != K:
			raise ValueError("Dose matrix must have same number of rows as structures")
		A_list = T*[A_list]
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
	
	# Load visualization matrices if provided.
	if "visualization" in data:
		n_beams = A_list[0].shape[1]
		dviz = data["visualization"]
		
		visuals = dict()
		visuals["beam_angles"] = np.load(dviz["beam_angles"])
		visuals["beam_offsets"] = np.load(dviz["beam_offsets"])
		
		n_angles = len(visuals["beam_angles"])
		n_offsets = len(visuals["beam_offsets"])
		if (n_angles*n_offsets) != n_beams:
			raise ValueError("Total number of beam angles/offsets inconsistent with dose matrix")

		with open(dviz["structure_regions"], "rb") as fp:
			vstr = visuals["structures"] = pickle.load(fp)
		if (vstr["x_grid"].shape != vstr["regions"].shape) or (vstr["y_grid"].shape != vstr["regions"].shape):
			raise ValueError("Inconsistent dimensions between structure x_grid, y_grid, and regions")
	else:
		visuals = None
	
	return A_list, visuals

def load_beam_mats(data, T, n):
	if "beams" in data:
		lower = data["beams"].get("lower_bound", 0)
		if np.isscalar(lower):
			lower_mat = np.full((T, n), lower)
		else:
			lower_mat = np.load(lower)
			if lower_mat.shape != (T, n):
				raise ValueError("Beam lower bound matrix must have dimensions ({0},{1})".format(T, n))

		upper = data["beams"].get("upper_bound", np.inf)
		if np.isscalar(upper):
			upper_mat = np.full((T, n), upper)
		else:
			upper_mat = np.load(upper)
			if upper_mat.shape != (T, n):
				raise ValueError("Beam upper bound matrix must have dimensions ({0},{1})".format(T, n))
	else:
		lower_mat = np.zeros((T,n))
		upper_mat = np.full((T,n), np.inf)

	return {"lower": lower_mat, "upper": upper_mat}
