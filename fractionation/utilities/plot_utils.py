import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize

from fractionation.utilities.data_utils import line_segments

# Extract subset of a colormap.
# http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_cmap(cmap, minval = 0.0, maxval = 1.0, n = 100):
    cmap_trunc = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n = cmap.name, a = minval, b = maxval),
        cmap(np.linspace(minval, maxval, n)))
    return cmap_trunc

# Modify colormap to enable transparency in smaller values.
def transp_cmap(cmap, lower = 0, upper = 1):
	cmap_transp = cmap(np.arange(cmap.N))
	cmap_transp[:,-1] = np.linspace(lower, upper, cmap.N)
	cmap_transp = ListedColormap(cmap_transp)
	return cmap_transp

# Plot structures.
def plot_structures(x, y, structures, title = None, one_idx = False, show = True, filename = None, *args, **kwargs):
	m, n = x.shape
	if y.shape != (m,n):
		raise ValueError("y must have dimensions ({0},{1})".format(m,n))
	if structures.shape != (m,n):
		raise ValueError("structures must have dimensions ({0},{1})".format(m,n))
	
	fig = plt.figure()
	fig.set_size_inches(10,8)
	
	labels = np.unique(structures + int(one_idx))
	lmin = np.min(labels)
	lmax = np.max(labels)
	levels = np.arange(lmin, lmax + 2) - 0.5
	
	ctf = plt.contourf(x, y, structures + int(one_idx), levels = levels, *args, **kwargs)
	
	plt.axhline(0, lw = 1, ls = ':', color = "grey")
	plt.axvline(0, lw = 1, ls = ':', color = "grey")
	plt.xticks([0])
	plt.yticks([0])
	
	if title is not None:
		plt.title(title)
	cbar = plt.colorbar(ctf, ticks = np.arange(lmin, lmax + 1))
	cbar.solids.set(alpha = 1)
	if show:
		plt.show()
	if filename is not None:
		fig.savefig(filename, bbox_inches = "tight", dpi = 300)

# Plot beams.
def plot_beams(b, angles, offsets, n_grid, stepsize = 10, maxcols = 5, standardize = False, title = None, one_idx = False,
			   show = True, filename = None, structures = None, struct_kw = dict(), *args, **kwargs):
	T = b.shape[0]
	n = b.shape[1]
	xlim = kwargs.pop("xlim", (-1,1))
	ylim = kwargs.pop("ylim", (-1,1))
	
	vmax_eps = 1e-3*(np.max(b) - np.min(b))
	# norm = kwargs.pop("norm", Normalize(vmin = np.min(b), vmax = np.max(b)))
	norm = kwargs.pop("norm", Normalize(vmin = 0, vmax = np.max(b) + vmax_eps))
	
	if len(angles)*len(offsets) != n:
		raise ValueError("len(angles)*len(offsets) must equal {0}".format(n))
	if standardize:
		b_mean = np.tile(np.mean(b, axis = 1), (n,1)).T
		b_std = np.tile(np.std(b, axis = 1), (n,1)).T
		b = (b - b_mean)/b_std
	if structures is not None:
		if len(structures) != 3:
			raise ValueError("structures must be a tuple of (x, y, labels)")
		struct_x, struct_y, struct_labels = structures
		
		# Set levels for structure colorbar.
		labels = np.unique(struct_labels + int(one_idx))
		lmin = np.min(labels)
		lmax = np.max(labels)
		struct_levels = np.arange(lmin, lmax + 2) - 0.5
	
	T_grid = np.arange(0, T, stepsize)
	if T_grid[-1] != T-1:
		T_grid = np.append(T_grid, T-1)   # Always include last time period.
	T_steps = len(T_grid)
	rows = 1 if T_steps <= maxcols else int(np.ceil(T_steps / maxcols))
	cols = min(T_steps, maxcols)
	
	fig, axs = plt.subplots(rows, cols)
	fig.set_size_inches(16,8)
	
	t = 0
	segments = line_segments(angles, offsets, n_grid, xlim, ylim)   # Create collection of beams.
	for t_step in range(T_steps):
		if rows == 1:
			ax = axs if cols == 1 else axs[t_step]
		else:
			ax = axs[int(t_step / maxcols), t_step % maxcols]
		
		# Plot anatomical structures.
		if structures is not None:
			ctf = ax.contourf(struct_x, struct_y, struct_labels + int(one_idx), levels = struct_levels, **struct_kw)
		
		# Set colors based on beam intensity.
		lc = LineCollection(segments, norm = norm, *args, **kwargs)
		lc.set_array(np.asarray(b[t]))
		ax.add_collection(lc)
		
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		# ax.set_title("$b({0})$".format(t+1))
		ax.set_title("$t = {0}$".format(t+1))
		t = min(t + stepsize, T-1)
	
	# Display colorbar for structures by label.
	if structures is not None:
		fig.subplots_adjust(left = 0.2)
		cax_left = fig.add_axes([0.125, 0.15, 0.02, 0.7])
		cbar = fig.colorbar(ctf, cax = cax_left, ticks = np.arange(lmin, lmax + 1), label = "Structure Label")
		cbar.solids.set(alpha = 1)
		cax_left.yaxis.set_label_position("left")
	
	# Display colorbar for entire range of intensities.
	fig.subplots_adjust(right = 0.8)
	cax_right = fig.add_axes([0.85, 0.15, 0.02, 0.7])
	lc = LineCollection(np.zeros((1,2,2)), norm = norm, *args, **kwargs)
	lc.set_array(np.array([np.min(b), np.max(b)]))
	cbar = fig.colorbar(lc, cax = cax_right, label = "Beam Intensity")
	cbar.solids.set(alpha = 1)
	cax_right.yaxis.set_label_position("left")
	
	if title is not None:
		plt.suptitle(title)
	if show:
		plt.show()
	if filename is not None:
		fig.savefig(filename, bbox_inches = "tight", dpi = 300)

# Plot health curves.
def plot_health(h, curves = [], stepsize = 10, maxcols = 5, T_treat = None, bounds = None, title = None, label = "Treated",
				ylim = None, indices = None, one_idx = False, show = True, filename = None, *args, **kwargs):
	# if len(h.shape) == 1:
	#	h = h[:,np.newaxis]
	T = h.shape[0] - 1
	m = h.shape[1]
	
	if bounds is not None:
		lower, upper = bounds
	else:
		lower = upper = None
	if indices is None:
		indices = np.arange(m) + int(one_idx)
	
	rows = 1 if m <= maxcols else int(np.ceil(m / maxcols))
	cols = min(m, maxcols)
	left = rows*cols - m

	fig, axs = plt.subplots(rows, cols, sharey = True)
	fig.set_size_inches(16,8)
	if ylim is not None:
		plt.setp(axs, ylim = ylim)
	for i in range(m):
		if rows == 1:
			ax = axs if cols == 1 else axs[i]
		else:
			ax = axs[int(i / maxcols), i % maxcols]
		ltreat, = ax.plot(range(T+1), h[:,i], label = label, *args, **kwargs)
		handles = [ltreat]
		for curve in curves:
			# if len(curve["h"].shape) == 1:
			#	curve["h"] = curve["h"][:,np.newaxis]
			curve_kw = curve.get("kwargs", {})
			lcurve, = ax.plot(range(T+1), curve["h"][:,i], label = curve["label"], **curve_kw)
			handles += [lcurve]
		# lnone, = ax.plot(range(T+1), h_prog[:,i], ls = '--', color = "red")
		ax.set_title("$h_{{{0}}}(t)$".format(indices[i]))
		# ax.set_title("$s = {{{0}}}$".format(indices[i]))
		
		# Label transition from treatment to recovery period.
		xt = np.arange(0, T, stepsize)
		xt = np.append(xt, T)
		if T_treat is not None:
			ax.axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
			xt = np.append(xt, T_treat)
		ax.set_xticks(xt)
		
		# Plot lower and upper bounds on h(t) for t = 1,...,T.
		if lower is not None:
			ax.plot(range(1,T+1), lower[:,i], lw = 1, ls = "--", color = "cornflowerblue")
		if upper is not None:
			ax.plot(range(1,T+1), upper[:,i], lw = 1, ls = "--", color = "cornflowerblue")
	
	for col in range(left):
		axs[rows-1, maxcols-1-col].set_axis_off()
	
	if len(curves) != 0:
		ax.legend(handles = handles, loc = "upper right", borderaxespad = 1)
		# fig.legend(handles = handles, loc = "center right", borderaxespad = 1)
		# fig.legend(handles = [ltreat, lnone], labels = ["Treated", "Untreated"], loc = "center right", borderaxespad = 1)
	
	if title is not None:
		plt.suptitle(title)
	if show:
		plt.show()
	if filename is not None:
		fig.savefig(filename, bbox_inches = "tight", dpi = 300)

# Plot treatment curves.
def plot_treatment(d, curves = [], stepsize = 10, maxcols = 5, T_treat = None, bounds = None, title = None, label = "Treatment",
				   ylim = None, one_idx = False, show = True, filename = None, *args, **kwargs):
	T = d.shape[0]
	n = d.shape[1]
	
	if bounds is not None:
		lower, upper = bounds
	else:
		lower = upper = None
	
	rows = 1 if n <= maxcols else int(np.ceil(n / maxcols))
	cols = min(n, maxcols)
	left = rows*cols - n
	
	fig, axs = plt.subplots(rows, cols, sharey = True)
	fig.set_size_inches(16,8)
	if ylim is not None:
		plt.setp(axs, ylim = ylim)
	for j in range(n):
		if rows == 1:
			ax = axs if cols == 1 else axs[j]
		else:
			ax = axs[int(j / maxcols), j % maxcols]
		ldose, = ax.plot(range(1,T+1), d[:,j], label = label, *args, **kwargs)
		handles = [ldose]
		for curve in curves:
			curve_kw = curve.get("kwargs", {})
			lcurve, = ax.plot(range(1,T+1), curve["d"][:,j], label = curve["label"], **curve_kw)
			handles += [lcurve]
		# ax.plot(range(1,T+1), d[:,j])
		ax.set_title("$d_{{{0}}}(t)$".format(j + int(one_idx)))
		
		# Label transition from treatment to recovery period.
		xt = np.arange(1, T, stepsize)
		xt = np.append(xt, T)
		if T_treat is not None:
			ax.axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
			xt = np.append(xt, T_treat)
		ax.set_xticks(xt)
		
		# Plot lower and upper bounds on d(t) for t = 1,...,T.
		if lower is not None:
			ax.plot(range(1,T+1), lower[:,j], lw = 1, ls = "--", color = "cornflowerblue")
		if upper is not None:
			ax.plot(range(1,T+1), upper[:,j], lw = 1, ls = "--", color = "cornflowerblue")
	
	for col in range(left):
		axs[rows-1, maxcols-1-col].set_axis_off()
	
	if len(curves) != 0:
		ax.legend(handles = handles, loc = "upper right", borderaxespad = 1)
	
	if title is not None:
		plt.suptitle(title)
	if show:
		plt.show()
	if filename is not None:
		fig.savefig(filename, bbox_inches = "tight", dpi = 300)

# Plot primal and dual residuals.
def plot_residuals(r_primal, r_dual, normalize = False, title = None, semilogy = False, show = True, filename = None, *args, **kwargs):
	if normalize:
		r_primal = r_primal/r_primal[0] if r_primal[0] != 0 else r_primal
		r_dual = r_dual/r_dual[0] if r_dual[0] != 0 else r_dual
	
	fig = plt.figure()
	fig.set_size_inches(12,8)
	if semilogy:
		plt.semilogy(range(len(r_primal)), r_primal, label = "Primal", *args, **kwargs)
		plt.semilogy(range(len(r_dual)), r_dual, label = "Dual", *args, **kwargs)
	else:
		plt.plot(range(len(r_primal)), r_primal, label = "Primal", *args, **kwargs)
		plt.plot(range(len(r_dual)), r_dual, label = "Dual", *args, **kwargs)

	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Residual")
	
	if title:
		plt.title(title)
	if show:
		plt.show()
	if filename is not None:
		fig.savefig(filename, bbox_inches = "tight", dpi = 300)

# Plot slack in health dynamics constraint.
def plot_slacks(slack, title = None, semilogy = False, show = True, filename = None, *args, **kwargs):
	fig = plt.figure()
	fig.set_size_inches(12, 8)

	if semilogy:
		plt.semilogy(range(len(slack)), slack, *args, **kwargs)
	else:
		plt.plot(range(len(slack)), slack, *args, **kwargs)

	# plt.gca().set_xlim(0, len(slack))
	plt.gca().set_ylim(bottom = 0)

	plt.xlabel("Iteration (s)")
	plt.ylabel("Total Slack")

	if title:
		plt.title(title)
	if show:
		plt.show()
	if filename is not None:
		fig.savefig(filename, bbox_inches = "tight", dpi = 300)
