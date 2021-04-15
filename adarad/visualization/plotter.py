import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from adarad.medicine.case import Case
from adarad.visualization.history import RunRecord
from adarad.utilities.plot_utils import plot_single, plot_slacks, plot_residuals
from adarad.utilities.data_utils import line_segments

class StructMap(object):
    def __init__(self, x, y, regions):
        if not (isinstance(x, np.ndarray) and x.ndim == 2):
            raise TypeError("x must be a 2-D matrix")
        if not (isinstance(y, np.ndarray) and y.ndim == 2):
            raise TypeError("y must be a 2-D matrix")
        if not (isinstance(regions, np.ndarray) and regions.ndim == 2):
            raise TypeError("regions must be a 2-D matrix")
        if x.shape != regions.shape or y.shape != regions.shape:
            raise TypeError("x, y, and regions must have same dimensions")
        self.x = x
        self.y = y
        self.regions = regions

    @property
    def shape(self):
        return self.regions.shape

class CasePlotter(object):
    def __init__(self, case, figsize=(16,8), one_idx=False, struct_map=None, struct_kw=None):
        # Case information.
        if not isinstance(case, Case):
            raise TypeError("'case' must be of type Case")
        self.__case = case

        # Plot setup.
        if not (isinstance(figsize, tuple) and len(figsize) == 2):
            raise TypeError("figsize must be a tuple of length 2")
        self.__figsize = figsize
        self.__one_idx = bool(one_idx)

        # Visual map of anatomy.
        if not (struct_map is None or isinstance(struct_map, StructMap)):
            raise TypeError("struct_map must be of type StructMap")
        self.__struct_map = struct_map

        if struct_kw is None:
            struct_kw = dict()
        if not isinstance(struct_kw, dict):
            raise TypeError("struct_kw must be a dictionary")
        self.__struct_kw = struct_kw

    @property
    def case(self):
        return self.__case

    @case.setter
    def case(self, data):
        if not isinstance(data, Case):
            raise TypeError("case must be of type Case")

    @property
    def T_treat(self):
        return self.case.prescription.T_treat

    @property
    def beam_angles(self):
        return self.case.physics.beams.angles

    @property
    def beam_offsets(self):
        return self.__case.physics.beams.offsets

    @property
    def figsize(self):
        return self.__figsize

    @figsize.setter
    def figsize(self, data):
        if not (isinstance(data, tuple) and len(data) == 2):
            raise TypeError("figsize must be a tuple of positive integers")
        self.__figsize = data

    @property
    def struct_map(self):
        return self.__struct_map

    @struct_map.setter
    def struct_map(self, data):
        if not isinstance(data, StructMap):
            raise TypeError("struct_map must be a StructMap")
        self.__struct_map = data

    @property
    def struct_kw(self):
        return self.__struct_kw

    @struct_kw.setter
    def struct_kw(self, data):
        if not isinstance(data, dict):
            raise TypeError("struct_kw must be a dictionary")
        self.__struct_kw = data

    def plot_structures(self, title=None, show=True, filename=None, *args, **kwargs):
        if self.struct_map is None:
            raise ValueError("struct_map must be defined")

        fig = plt.figure()
        fig.set_size_inches(self.figsize)

        labels = np.unique(self.struct_map.regions + int(self.__one_idx))
        lmin = np.min(labels)
        lmax = np.max(labels)
        levels = np.arange(lmin, lmax + 2) - 0.5

        ctf = plt.contourf(self.struct_map.x, self.struct_map.y, self.struct_map.regions + int(self.__one_idx), levels=levels,
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
                         n_grid=None, plot_anatomy=True, plot_saved=False, *args, **kwargs):
        if plot_saved:
            raise NotImplementedError
        if plot_anatomy and self.struct_map is None:
            raise ValueError("struct_map must be defined in order to plot anatomical structures")
        if self.struct_map is not None and n_grid is None:
            m_grid, n_grid = self.struct_map.shape
            if m_grid != n_grid:
                raise NotImplementedError("plot_beams can only handle symmetric 2-D anatomical grids")
        elif self.struct_map is None and n_grid is not None:
            if n_grid <= 0:
                raise TypeError("n_grid must be a positive integer")
            n_grid = int(n_grid)
        else:
            raise ValueError("Exactly one of struct_map and n_grid must be defined")
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
        if plot_anatomy:
            labels = np.unique(self.struct_map.regions + int(self.__one_idx))
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
        fig.set_size_inches(self.figsize)

        t = 0
        segments = line_segments(self.beam_angles, self.beam_offsets, n_grid, xlim, ylim)  # Create collection of beams.
        for t_step in range(T_steps):
            if rows == 1:
                ax = axs if cols == 1 else axs[t_step]
            else:
                ax = axs[int(t_step / maxcols), t_step % maxcols]

            # Plot anatomical structures.
            if plot_anatomy:
                ctf = ax.contourf(self.struct_map.x, self.struct_map.y, self.struct_map.regions + int(self.__one_idx),
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

    def plot_treatment(self, result, plot_rec_div=False, plot_saved=False, saved_plans_kw=dict(), *args, **kwargs):
        d = result.doses if isinstance(result, RunRecord) else result
        T_treat_lab = self.T_treat if plot_rec_div else None   # Vertical line dividing treatment/recovery phases.
        bounds = self.case.prescription.dose_bounds_to_mats()

        curves = []
        if plot_saved:
            for name, plan in self.case.saved_plans.items():
                curve_kw = saved_plans_kw.get(name, {})
                curves += [{"d": plan.doses, "label": name, "kwargs": curve_kw}]
        return plot_single(d, "d", curves=curves, T_treat=T_treat_lab, bounds=bounds, one_shift=True, one_idx=self.__one_idx, figsize=self.figsize, *args, **kwargs)

    def plot_health(self, result, plot_rec_div=False, plot_untreated=True, plot_saved=False, untreated_kw=dict(), saved_plans_kw=dict(), *args, **kwargs):
        h = result.health if isinstance(result, RunRecord) else result
        T_treat_lab = self.T_treat if plot_rec_div else None   # Vertical line dividing treatment/recovery phases.
        bounds = self.case.prescription.health_bounds_to_mats()

        curves = []
        if plot_saved:
            for name, plan in self.case.saved_plans.items():
                curve_kw = saved_plans_kw.get(name, {})
                curves += [{"h": plan.health, "label": name, "kwargs": curve_kw}]
        if plot_untreated:
            h_prog = self.case.health_prognosis(self.T_treat)
            curves += [{"h": h_prog, "label": "Untreated", "kwargs": untreated_kw}]
        return plot_single(h, "h", curves=curves, T_treat=T_treat_lab, bounds=bounds, one_shift=False, one_idx=self.__one_idx, figsize=self.figsize, *args, **kwargs)

    @staticmethod
    def plot_slacks(result, *args, **kwargs):
        slacks = result.slacks if isinstance(result, RunRecord) else result
        plot_slacks(slacks, *args, **kwargs)

    @staticmethod
    def plot_residuals(result, *args, **kwargs):
        if not isinstance(result, RunRecord):
            raise TypeError("result must be of type RunRecord")
        if not result.profile.use_admm:
            raise RuntimeError("Residuals only available for ADMM")
        plot_residuals(result.output.primal_residuals, result.output.dual_residuals, *args, **kwargs)
