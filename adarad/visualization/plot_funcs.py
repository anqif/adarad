import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from adarad.utilities.beam_utils import line_segments

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

# Single plot of structure with beams.
def plot_struct_beams(x, y, structures, b, angles, offsets, n_grid, title = None, one_idx = False, show = True, filename = None, 
                      beam_kw = dict(), *args, **kwargs):
    n = b.shape[0]
    xlim = kwargs.pop("xlim", (-1,1))
    ylim = kwargs.pop("ylim", (-1,1))
    vmax_eps = 1e-3*(np.max(b) - np.min(b))
    # norm = beam_kw.pop("norm", Normalize(vmin = np.min(b), vmax = np.max(b)))
    norm = beam_kw.pop("norm", Normalize(vmin = 0, vmax = np.max(b) + vmax_eps))

    if len(angles)*len(offsets) != n:
        raise ValueError("len(angles)*len(offsets) must equal {0}".format(n))

    plot_structures(x, y, structures, title = title, one_idx = one_idx, show = False, filename = None, *args, **kwargs)
    ax = plt.gca()    # Get plot axis.
    fig = plt.gcf()   # Get plot figure.

    # Plot beam intensities.
    segments = line_segments(angles, offsets, n_grid, xlim, ylim)
    lc = LineCollection(segments, norm = norm, **beam_kw)
    lc.set_array(np.asarray(b))
    ax.add_collection(lc)

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
plot_health = lambda h, *args, **kwargs: plot_single(h, "h", one_shift = False, *args, **kwargs)

# Plot optimization curves.
plot_treatment = lambda d, *args, **kwargs: plot_single(d, "d", one_shift = True, *args, **kwargs)

def plot_stacked(v_list, maxcols = 5, T_treat = None, title = None, ylim = None, show = True, filename = None, figsize = (16,16), *args, **kwargs):
    v_len = len(v_list)
    sharex = kwargs.pop("sharex", "col")
    sharey = kwargs.pop("sharey", "row")

    if "v" not in v_list[0]:
        raise ValueError("v_list dicts must all contain a key 'v' mapped to a matrix")
    m = v_list[0]["v"].shape[1]

    for v_dict in v_list:
        if "v" not in v_dict:
            raise ValueError("v_list dicts must all contain a key 'v' mapped to a matrix")
        if "varname" not in v_dict:
            raise ValueError("v_list dicts must all contain a key 'varname' mapped to a string")
        if v_dict["v"].shape[1] != m:
            raise ValueError("v_list['v'] must be matrices with the same number of columns {0}".format(m))

    rows = 1 if m <= maxcols else int(np.ceil(m / maxcols))
    cols = min(m, maxcols)
    fig, axs = plt.subplots(v_len*rows, cols, sharex = sharex, sharey = sharey, *args, **kwargs)
    fig.set_size_inches(*figsize)
    # fig.tight_layout()
    if ylim is not None:
        plt.setp(axs, ylim = ylim)

    r_start = 0
    for v_dict in v_list:
        v = v_dict.pop("v")
        varname = v_dict.pop("varname")
        plot_internal(v, axs[r_start:(r_start + rows)], varname, maxcols = maxcols, T_treat = T_treat, **v_dict)
        r_start = r_start + rows

    if title is not None:
        plt.suptitle(title)
    if show:
        plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches = "tight", dpi = 300)

def plot_single(h, varname, curves = [], stepsize = 10, maxcols = 5, T_treat = None, bounds = None, title = None, subtitles = None, label = "Treated", 
                ylim = None, one_idx = False, one_shift = False, show = True, filename = None, figsize = (16,8), *args, **kwargs):
    # if len(h.shape) == 1:
    #	h = h[:,np.newaxis]
    m = h.shape[1]
    rows = 1 if m <= maxcols else int(np.ceil(m / maxcols))
    cols = min(m, maxcols)

    fig, axs = plt.subplots(rows, cols, sharey = True)
    fig.set_size_inches(*figsize)
    # fig.tight_layout()
    if ylim is not None:
        plt.setp(axs, ylim = ylim)

    plot_internal(h, axs, varname, curves, stepsize, maxcols, T_treat, bounds, subtitles, label, one_idx, one_shift, *args, **kwargs)

    if title is not None:
        plt.suptitle(title)
    if show:
        plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches = "tight", dpi = 300)

def plot_internal(h, axs, varname, curves = [], stepsize = 10, maxcols = 5, T_treat = None, bounds = None, subtitles = None, label = "Treated",
                  one_idx = False, one_shift = False, *args, **kwargs):
    # if len(h.shape) == 1:
    #	h = h[:,np.newaxis]
    T = h.shape[0]
    m = h.shape[1]
    if m == 0 or T == 0:
        raise ValueError("Nothing to plot since variable shape is ({0},{1})".format(T,m))
    T_start = int(one_shift)
    T_end = T + int(one_shift)

    if bounds is not None:
        lower, upper = bounds
    else:
        lower = upper = None

    indices = np.arange(m) + int(one_idx)
    if subtitles is None:
        subtitles = ["${0}_{{{1}}}(t)$".format(varname, index) for index in indices]
    if len(subtitles) != m:
        raise ValueError("subtitles must be a list of length", m)

    # Determine plot dimensions.
    if hasattr(axs, "shape"):
        if len(axs.shape) == 1:
            ax_type = 1
        else:
            ax_type = 2
    else:
        ax_type = 0

    for i in range(m):
        if ax_type == 0:
            ax = axs
        elif ax_type == 1:
            ax = axs[i]
        else:
            ax = axs[int(i / maxcols), i % maxcols]

        ltreat, = ax.plot(range(T_start, T_end), h[:,i], label = label, *args, **kwargs)
        handles = [ltreat]
        for curve in curves:
            # if len(curve[varname].shape) == 1:
            #	curve[varname] = curve[varname][:,np.newaxis]
            curve_kw = curve.get("kwargs", {})
            lcurve, = ax.plot(range(T_start, T_end), curve[varname][:,i], label = curve["label"], **curve_kw)
            handles += [lcurve]
        # lnone, = ax.plot(range(T_start, T_end), h_prog[:,i], ls = '--', color = "red")
        ax.set_title(subtitles[i])
        # ax.set_title("${0}_{{{1}}}(t)$".format(varname, indices[i]))
        # ax.set_title("$s = {{{0}}}$".format(indices[i]))

        # Label transition from optimization to recovery period.
        xt = np.arange(T_start, T_end - 1, stepsize)
        xt = np.append(xt, T_end - 1)
        if T_treat is not None and T_treat < T:
            ax.axvline(x = T_treat, lw = 1, ls = ':', color = "grey")
            xt = np.append(xt, T_treat)
        ax.set_xticks(xt)

        # Plot lower and upper bounds on h(t) for t = 1,...,T.
        if lower is not None:
            ax.plot(range(1, T_end), lower[:,i], lw = 1, ls = "--", color = "cornflowerblue")
        if upper is not None:
            ax.plot(range(1, T_end), upper[:,i], lw = 1, ls = "--", color = "cornflowerblue")

    left = axs.size - m if hasattr(axs, "size") else 0
    for col in range(left):
        axs[-1, maxcols-1-col].set_axis_off()

    if len(curves) != 0:
        ax.legend(handles = handles, loc = "upper right", borderaxespad = 1)
        # fig.legend(handles = handles, loc = "center right", borderaxespad = 1)
        # fig.legend(handles = [ltreat, lnone], labels = ["Treated", "Untreated"], loc = "center right", borderaxespad = 1)

# Plot primal and dual residuals.
def plot_residuals(r_primal, r_dual, normalize = False, title = None, semilogy = False, show = True, filename = None, figsize = (12,8), *args, **kwargs):
    if r_primal is None and r_dual is None:
        raise ValueError("Both primal and dual residuals are None")
    if (r_primal is not None and r_dual is not None) and len(r_primal) != len(r_dual):
        raise ValueError("Primal and dual residual vectors must have same length")

    # TODO: Handle case when r_primal and r_dual are lists of 1-D arrays (e.g., for ADMM + MPC).
    if normalize:
        r_primal = r_primal/r_primal[0] if r_primal[0] != 0 else r_primal
        r_dual = r_dual/r_dual[0] if r_dual[0] != 0 else r_dual

    fig = plt.figure()
    fig.set_size_inches(*figsize)
    if semilogy:
        if r_primal is not None:
            plt.semilogy(range(len(r_primal)), r_primal, label = "Primal Residual", *args, **kwargs)
        if r_dual is not None:
            plt.semilogy(range(len(r_dual)), r_dual, label = "Dual Residual", *args, **kwargs)
    else:
        if r_primal is not None:
            plt.plot(range(len(r_primal)), r_primal, label = "Primal Residual", *args, **kwargs)
        if r_dual is not None:
            plt.plot(range(len(r_dual)), r_dual, label = "Dual Residual", *args, **kwargs)

    plt.legend()
    plt.xlabel("Iteration")
    # plt.ylabel("$||r|| _2$")
    plt.ylabel("$\ell_2$-norm of Residual")

    if title:
        plt.title(title)
    if show:
        plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches = "tight", dpi = 300)

# Plot slack in health dynamics constraint.
def plot_slacks(slack, title = None, semilogy = False, show = True, filename = None, figsize = (12,8), maxcols = 5, *args, **kwargs):
    if isinstance(slack, np.ndarray):
        slack = [slack]
    if not isinstance(slack, list):
        raise TypeError("slack must be a 1-D array or list of 1-D arrays")

    T = len(slack)
    rows = 1 if T <= maxcols else int(np.ceil(T / maxcols))
    cols = min(T, maxcols)

    fig, axs = plt.subplots(rows, cols, sharey = True)
    fig.set_size_inches(*figsize)

    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)
    plt.xlabel("Iteration (s)")
    plt.ylabel("Total Health Slack")
    # fig.text(0.5, 0.04, "Iteration (s)", ha='center')
    # fig.text(0.06, 0.5, "Total Health Slack", va='center', rotation='vertical')

    # Determine plot dimensions.
    if hasattr(axs, "shape"):
        if len(axs.shape) == 1:
            ax_type = 1
        else:
            ax_type = 2
    else:
        ax_type = 0

    for t in range(T):
        if ax_type == 0:
            ax = axs
        elif ax_type == 1:
            ax = axs[t]
        else:
            ax = axs[int(t / maxcols), t % maxcols]

        n_iters = len(slack[t])
        if semilogy:
            ax.semilogy(range(n_iters), slack[t], *args, **kwargs)
        else:
            ax.plot(range(n_iters), slack[t], *args, **kwargs)

        if T != 1:
            ax.set_title("$t = {{{0}}}$".format(t))
        # ax.set_xlim(0, n_iters)
        ax.set_ylim(bottom = 0)

    # Hide unused subplots.
    left = axs.size - T if hasattr(axs, "size") else 0
    for col in range(left):
        axs[-1,maxcols-1-col].set_axis_off()

    if title:
        plt.suptitle(title)
    if show:
        plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches = "tight", dpi = 300)
