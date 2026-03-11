"""
load_snapshots.py

This file helps load velocity data from OpenFOAM simulations.
It applies a spatial mask, selects time windows for training and future prediction, and returns the data in a structured format.

Functions:
- load_masked_matrix: Loads velocity snapshots for selected times and masked region.
- load_all_snapshots: Loads data for each Reynolds number, applies the mask, and returns training and future snapshots.
- load_test_snapshots: Loads data for a single Reynolds number not used in training (for testing or comparison).
"""

import numpy as np
import torch as pt
from pathlib import Path

# load training parameter

class training_snapshots:
    """
    Container for all training parameters μ ∈ P.

    Attributes
    ----------
    snapshots : dict[int → ndarray]
        Training snapshots X^(μ) for each Reynolds number μ.
        Shape: (2 * num_points, Nt_train)

    future_snapshots : dict[int → ndarray]
        Future snapshots X_future^(μ) for each Reynolds number μ.
        Shape: (2 * num_points, Nt_future)

    times : dict[int → list[float]]
        Training time instants T for each μ.

    future_times : dict[int → list[float]]
        Future time instants T_future for each μ.

    mask : torch.Tensor or ndarray
        Boolean mask selecting the spatial region of interest.

    coords : ndarray
        Masked spatial coordinates (num_points, 2).

    num_points : int
        Number of spatial points after masking.
    """
    def __init__(self):
        self.snapshots = {}
        self.future_snapshots = {}
        self.times = {}
        self.future_times = {}
        self.mask = None
        self.coords = None
        self.num_points = None


class test_snapshots:
    """
    Container for the test parameter μ*.

    Attributes
    ----------
    test_future_snapshots : ndarray
        Future snapshots X_future^(μ*) for the test parameter.
        Shape: (2 * num_points, Nt_future)

    test_future_times : list[float]
        Future time instants T_future used for the test parameter.

    mask : torch.Tensor or ndarray
        Same spatial mask as used for training.

    coords : ndarray
        Masked spatial coordinates (num_points, 2).

    num_points : int
        Number of spatial points after masking.
    """
    def __init__(self):
        self.test_future_snapshots = None
        self.test_future_times = None
        self.mask = None
        self.coords = None
        self.num_points = None


# Load mask

def load_masked_matrix(loader, mask_indices, times):
    """
    Load masked velocity snapshots [Ux; Uy] for a list of time instants.

    Parameters
    ----------
    loader : FOAMDataloader
        Loader for the OpenFOAM case.

    mask_indices : ndarray
        Indices of spatial points to keep.

    times : list[float]
        Time instants to load.

    Returns
    -------
    data_matrix : ndarray
        Snapshot matrix of shape (2 * num_points, Nt).

    num_points : int
        Number of masked spatial points.
    """
    num_points = len(mask_indices)
    num_times = len(times)
    data_matrix = np.zeros((2 * num_points, num_times), dtype=np.float64)

    for i, t in enumerate(times):
        snapshot = loader.load_snapshot("U", t).numpy()
        data_matrix[:num_points, i] = snapshot[mask_indices, 0]  # Ux
        data_matrix[num_points:, i] = snapshot[mask_indices, 1]  # Uy

    return data_matrix, num_points


# load training snapshots

def load_training_snapshots(Re_list, base_path, mask_box, FOAMDataloader,
                            training_window=(10.0, 15.0),
                            future_window=(15.0, 20.0),
                            sampling_step=1):
    """
    Load training and future snapshots for all training parameters μ ∈ P.

    Parameters
    ----------
    Re_list : list[int]
        List of Reynolds numbers used for training.

    base_path : str
        Base directory containing all simulation folders.

    mask_box : callable
        Function that computes a boolean mask from coordinates.

    FOAMDataloader : class
        Loader class for OpenFOAM cases.

    training_window : tuple(float, float)
        Time interval T used for training snapshots.

    future_window : tuple(float, float)
        Time interval T_future used for validation snapshots.

    sampling_step : int
        Downsampling factor for time instants.

    Returns
    -------
    training_snapshots
        Structured container with all training data.
    """

    data = training_snapshots()

    # Compute global mask once using the first parameter
    folder0 = Path(base_path).expanduser() / f"cylinder_2D_Re{Re_list[0]}"
    loader0 = FOAMDataloader(str(folder0))
    vertices0 = loader0.vertices[:, :2]

    mask = mask_box(vertices0, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(mask.numpy())[0]

    data.mask = mask
    data.coords = vertices0[mask_indices]
    data.num_points = len(mask_indices)

    # Load snapshots for each training parameter
    for Re in Re_list:
        folder = Path(base_path).expanduser() / f"cylinder_2D_Re{Re}"
        loader = FOAMDataloader(str(folder))
        times = loader.write_times

        # Training window
        train_times = [t for t in times
                       if training_window[0] <= float(t) <= training_window[1]][::sampling_step]

        # Future window
        future_times = [t for t in times
                        if future_window[0] <= float(t) <= future_window[1]][::sampling_step]

        # Load snapshots
        X_train, _ = load_masked_matrix(loader, mask_indices, train_times)
        X_future, _ = load_masked_matrix(loader, mask_indices, future_times)

        data.snapshots[Re] = X_train
        data.future_snapshots[Re] = X_future
        data.times[Re] = train_times
        data.future_times[Re] = future_times

        print(f"Loaded training and future snapshots for Re={Re}")

    return data



# Load test snapshots


def load_test_snapshots(Re, path, mask_box, FOAMDataloader,
                        future_times, sampling_step=1):
    """
    Load the test parameter μ* using the SAME future time window as training.

    Parameters
    ----------
    Re : int
        Test Reynolds number μ*.

    path : str
        Path to the test case folder.

    mask_box : callable
        Function that computes a boolean mask from coordinates.

    FOAMDataloader : class
        Loader class for OpenFOAM cases.

    future_times : list[float]
        Future time instants T_future used for training parameters.

    sampling_step : int
        Downsampling factor for time instants.

    Returns
    -------
    _
        Structured container with test future snapshots.
    """

    data = test_snapshots()

    # Load case
    loader = FOAMDataloader(path)
    vertices = loader.vertices[:, :2]

    # Compute mask (same mask_box → identical mask)
    mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(mask.numpy())[0]

    data.mask = mask
    data.coords = vertices[mask_indices]
    data.num_points = len(mask_indices)

    # Use SAME future times as training
    sampled_times = future_times[::sampling_step]

    data.test_future_times = sampled_times

    # Load masked snapshots
    X_future, _ = load_masked_matrix(loader, mask_indices, sampled_times)
    data.test_future_snapshots = X_future

    print(f"Loaded test future snapshots for Re={Re}")

    return data


