import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from fractionation.case import Case
from fractionation.record import RunRecord
from fractionation.utilities.plot_utils import plot_single
from fractionation.utilities.data_utils import line_segments

class CasePlotter(object):
    def __init__(self, case, figsize=(16,8), one_idx=False, struct_kw=dict()):
        if not isinstance(case, Case):
            raise TypeError("'case' must be of type Case")

        # Case information.
        self.__anatomy = case.anatomy
        self.__beam_angles = case.physics.beam_angles
        self.__beam_offsets = case.physics.beam_offsets

        # Plot setup.
        self.__figsize = figsize
        self.__one_idx = one_idx
        self.__struct_kw = struct_kw

    @property
    def anatomy(self):
        return self.__anatomy

    @property
    def beam_angles(self):
        return self.__beam_angles

    @property
    def beam_offsets(self):
        return self.__beam_offsets

    @property
    def figsize(self):
        return self.__figsize

    @property
    def struct_kw(self):
        return self.__struct_kw

    @figsize.setter
    def figsize(self, data):
        if not (isinstance(data, tuple) and len(data) == 2):
            raise TypeError("'figsize' must be a tuple of positive integers")
        self.__figsize = data

    @struct_kw.setter
    def struct_kw(self, data):
        if not isinstance(data, dict):
            raise TypeError("'struct_kw' must be a dictionary")
        self.__struct_kw = data

    def plot_structures(self, title=None, show=True, filename=None, *args, **kwargs):
        if self.anatomy is None:
            raise ValueError("'anatomy' of case is undefined.")

        fig = plt.figure()
        fig.set_size_inches(**self.figsize)

        labels = np.unique(self.anatomy.regions + int(self.__one_idx))
        lmin = np.min(labels)
        lmax = np.max(labels)
        levels = np.arange(lmin, lmax + 2) - 0.5

        ctf = plt.contourf(self.anatomy.x, self.anatomy.y, self.anatomy.regions + int(self.__one_idx), levels=levels,
                           *args, **kwargs)

        plt.axhline(0, lw=1, ls=':', color="grey")
        plt.axvline(0, lw=1, ls=':', color="grey")
        plt.xticks([0])
        plt.yticks([0])

        if title is not None:
            plt.title(title)
        cbar = plt.colorbar(ctf, ticks=np.arange(lmin, lmax + 1))
        cbar.solids.set(alpha=1)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)

    def plot_beams(self, result, stepsize=10, maxcols=5, standardize=False, title=None, show=True, filename=None,
                   plot_anatomy=True, *args, **kwargs):
        m_grid, n_grid = self.anatomy.dims
        if m_grid != n_grid:
            raise NotImplementedError("plot_beams can only handle symmetric 2-D anatomical grids")
        b = result.beams if isinstance(result, RunRecord) else result

        T = b.shape[0]
        n = b.shape[1]
        xlim = kwargs.pop("xlim", (-1, 1))
        ylim = kwargs.pop("ylim", (-1, 1))

        vmax_eps = 1e-3 * (np.max(b) - np.min(b))
        # norm = kwargs.pop("norm", Normalize(vmin = np.min(b), vmax = np.max(b)))
        norm = kwargs.pop("norm", Normalize(vmin=0, vmax=np.max(b) + vmax_eps))

        if len(self.beam_angles) * len(self.beam_offsets) != n:
            raise ValueError("len(beam_angles)*len(beam_offsets) must equal {0}".format(n))
        if standardize:
            b_mean = np.tile(np.mean(result.beams, axis=1), (n, 1)).T
            b_std = np.tile(np.std(result.beams, axis=1), (n, 1)).T
            b = (b - b_mean) / b_std

        # Set levels for structure colorbar.
        labels = np.unique(self.anatomy.regions + int(self.__one_idx))
        lmin = np.min(labels)
        lmax = np.max(labels)
        struct_levels = np.arange(lmin, lmax + 2) - 0.5

        T_grid = np.arange(0, T, stepsize)
        if T_grid[-1] != T - 1:
            T_grid = np.append(T_grid, T - 1)  # Always include last time period.
        T_steps = len(T_grid)
        rows = 1 if T_steps <= maxcols else int(np.ceil(T_steps / maxcols))
        cols = min(T_steps, maxcols)

        fig, axs = plt.subplots(rows, cols)
        fig.set_size_inches(**self.figsize)

        t = 0
        segments = line_segments(self.beam_angles, self.beam_offsets, n_grid, xlim, ylim)  # Create collection of beams.
        for t_step in range(T_steps):
            if rows == 1:
                ax = axs if cols == 1 else axs[t_step]
            else:
                ax = axs[int(t_step / maxcols), t_step % maxcols]

            # Plot anatomical structures.
            if plot_anatomy:
                ctf = ax.contourf(self.anatomy.x, self.anatomy.y, self.anatomy.regions + int(self.__one_idx),
                                  levels=struct_levels, **self.struct_kw)

            # Set colors based on beam intensity.
            lc = LineCollection(segments, norm=norm, *args, **kwargs)
            lc.set_array(np.asarray(b[t]))
            ax.add_collection(lc)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # ax.set_title("$b({0})$".format(t+1))
            ax.set_title("$t = {0}$".format(t + 1))
            t = min(t + stepsize, T - 1)

        # Display colorbar for structures by label.
        if plot_anatomy:
            fig.subplots_adjust(left=0.2)
            cax_left = fig.add_axes([0.125, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(ctf, cax=cax_left, ticks=np.arange(lmin, lmax + 1), label="Structure Label")
            cbar.solids.set(alpha=1)
            cax_left.yaxis.set_label_position("left")

        # Display colorbar for entire range of intensities.
        fig.subplots_adjust(right=0.8)
        cax_right = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        lc = LineCollection(np.zeros((1, 2, 2)), norm=norm, *args, **kwargs)
        lc.set_array(np.array([np.min(b), np.max(b)]))
        cbar = fig.colorbar(lc, cax=cax_right, label="Beam Intensity")
        cbar.solids.set(alpha=1)
        cax_right.yaxis.set_label_position("left")

        if title is not None:
            plt.suptitle(title)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)

    def plot_health(self, result, *args, **kwargs):
        h = result.health if isinstance(result, RunRecord) else result
        return plot_single(h, "h", one_shift=False, one_idx=self.__one_idx, figsize=self.figsize, *args, **kwargs)

    def plot_treatment(self, result, *args, **kwargs):
        d = result.doses if isinstance(result, RunRecord) else result
        return plot_single(d, "d", one_shift=True, one_idx=self.__one_idx, figsize=self.figsize, *args, **kwargs)
