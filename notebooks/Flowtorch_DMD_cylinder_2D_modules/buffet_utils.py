"""
    helper functions
"""
import numpy as np
import pandas as pd
import torch as pt

from glob import glob
from os.path import join
from scipy.signal import welch
from typing import Union, Tuple
from scipy.interpolate import interp1d
from flowtorch.data import FOAMDataloader
from flowtorch.analysis import SVD

def load_force_coeffs(load_path, usecols=[0, 1, 4], names=["t", "cx", "cy"]) -> pd.DataFrame:
    dirs = sorted(glob(join(load_path, "postProcessing", "forces", "*")), key=lambda x: float(x.split("/")[-1]))
    coeffs = [pd.read_csv(join(p, "coefficient.dat"), sep=r"\s+", comment="#", header=None, usecols=usecols, names=names)
              for p in dirs]
    if len(coeffs) == 1:
        coeffs = coeffs[0]
    else:
        coeffs = pd.concat(coeffs)

    # remove duplicates (resulting from dt < write precision) and reset the idx
    coeffs.drop_duplicates(["t"], inplace=True)
    coeffs.reset_index(inplace=True, drop=True)
    return coeffs


def compute_fft(data: np.ndarray, dt: Union[float, int]) -> Tuple[np.ndarray, np.ndarray]:
    _f, _a = welch(data, 1/dt, nperseg=len(data), nfft=len(data), window="boxcar")
    return _f, _a


def interpolate_uniform(t: np.ndarray, data: np.ndarray):
    # get start and end time
    t_start, t_end = t[0], t[-1]

    # use standard interpolation to get values at const. dt
    _interpolator = interp1d(t, data, fill_value="extrapolate")
    dt = float("{:.1e}".format(t[-1] - t[-2]))

    t_new = np.arange(start=t_start + dt, stop=t_end, step=dt)
    return t_new, _interpolator(t_new)


def compute_svd(data_matrix: pt.Tensor, cell_area: pt.Tensor, rank: int = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Compute a weighted SVD for a given field, where the field is weighted by the cell area (2D) or volume (3D).

    For more information on determining the optimal rank, see the FlowTorch documentation:
    `flowtorch.analysis.svd.SVD.opt_rank <https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/flowtorch.analysis.html#flowtorch.analysis.svd.SVD.opt_rank>`_.

    :param data_matrix: Data matrix with snapshots; the last dimension is expected to represent temporal evolution
    :type data_matrix: pt.Tensor
    :param cell_area: Area (2D) or volume (3D) for each cell
    :type cell_area: pt.Tensor
    :param rank: Number of modes to compute the SVD. If ``None``, the optimal rank will be used
    :type rank: int | None
    :return: Tuple containing the singular values, modes, and mode coefficients as ``(s, U, V)``
    :rtype: Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
    """

    # compute mean
    mean = pt.mean(data_matrix, dim=-1, keepdim=True)

    # subtract the temporal mean
    _field_size = data_matrix.size()
    data_matrix -= mean

    if len(_field_size) == 2:
        # multiply by the sqrt of the cell areas to weight their contribution
        data_matrix *= cell_area.sqrt().unsqueeze(-1)

        # save either everything until the optimal rank or up to a user specified rank
        svd = SVD(data_matrix, rank=rank)

        
        return svd.s, svd.U / cell_area.sqrt().unsqueeze(-1), svd.V, mean

    else:
        # multiply by the sqrt of the cell areas to weight their contribution
        data_matrix *= cell_area.sqrt().unsqueeze(-1).unsqueeze(-1)

        # stack the data of all components for the SVD
        orig_shape = _field_size
        data_matrix = data_matrix.reshape((orig_shape[1] * orig_shape[0], orig_shape[-1]))

        # save either everything until the optimal rank or up to a user specified rank
        svd = SVD(data_matrix, rank=rank)

        # reshape the data back to ux, uy, uz
        new_shape = (orig_shape[0], orig_shape[1], svd.rank)

        return svd.s, svd.U.reshape(new_shape) / cell_area.sqrt().unsqueeze(-1).unsqueeze(-1), svd.V, mean

