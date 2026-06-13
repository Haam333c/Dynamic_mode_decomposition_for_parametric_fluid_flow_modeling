"""
load_snapshots.py

Loads masked OpenFOAM fields for the 2D cylinder training and test datasets.
Supports selectable fields: U, p, omega, cl.
"""

import numpy as np
import torch as pt
from pathlib import Path


class training_snapshots:
    """
    Container for all training parameters.
    """
    def __init__(self):
        self.snapshots = {}
        self.future_snapshots = {}
        self.times = {}
        self.future_times = {}
        self.mask = None
        self.coords = None
        self.num_points = None
        self.cl_train = {}
        self.cl_train_times = {}
        self.cl_future = {}
        self.cl_future_times = {}


class test_snapshots:
    """
    Container for the test parameter.
    """
    def __init__(self):
        self.test_future_snapshots = None
        self.test_future_times = None
        self.mask = None
        self.coords = None
        self.num_points = None
        self.cl_test = None
        self.cl_test_times = None


def load_masked_matrix(loader, mask_indices, times, fields=("U",)):
    """
    Load masked field snapshots for selected times.
    """
    field_dims = {"U": 2, "p": 1, "omega": 1}
    spatial_fields = [f for f in fields if f in field_dims]

    num_points = len(mask_indices)
    num_times = len(times)
    total_components = sum(field_dims[f] for f in spatial_fields)

    data_matrix = np.zeros((total_components * num_points, num_times))

    for i, t in enumerate(times):
        row = 0
        for f in spatial_fields:
            snap = loader.load_snapshot(f, t).numpy()  
            if f == "U":
                data_matrix[row:row+num_points, i] = snap[mask_indices, 0]
                data_matrix[row+num_points:row+2*num_points, i] = snap[mask_indices, 1]
                row += 2 * num_points
            else:
                data_matrix[row:row+num_points, i] = snap[mask_indices]
                row += num_points

    return data_matrix, num_points


def load_cl(path):
    """
    Load lift coefficient time series from OpenFOAM forces file.
    """
    forces_file = Path(path) / "postProcessing/forces/0/coefficient.dat"
    data = np.loadtxt(forces_file, comments="#")
    return data[:, 0], data[:, 4]


def load_training_snapshots(Re_list, base_path, mask_box, FOAMDataloader,
                            training_window=(10.0, 15.0),
                            future_window=(15.0, 20.0),
                            sampling_step=1,
                            fields=("U",)):

    data = training_snapshots()

    folder0 = Path(base_path).expanduser() / f"cylinder_2D_Re{Re_list[0]}"
    loader0 = FOAMDataloader(str(folder0))
    vertices0 = loader0.vertices[:, :2]

    mask = mask_box(vertices0, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(mask.numpy())[0]

    data.mask = mask
    data.coords = vertices0[mask_indices]
    data.num_points = len(mask_indices)

    for Re in Re_list:
        folder = Path(base_path).expanduser() / f"cylinder_2D_Re{Re}"
        loader = FOAMDataloader(str(folder))
        times = loader.write_times   

        # Keep exact folder names for loading snapshots
        train_times_str = [t for t in times if training_window[0] <= float(t) <= training_window[1]][::sampling_step]
        future_times_str = [t for t in times if future_window[0] <= float(t) <= future_window[1]][::sampling_step]

        # Float versions for Cl slicing
        train_times_float = [float(t) for t in train_times_str]
        future_times_float = [float(t) for t in future_times_str]

        X_train, _ = load_masked_matrix(loader, mask_indices, train_times_str, fields)
        X_future, _ = load_masked_matrix(loader, mask_indices, future_times_str, fields)

        data.snapshots[Re] = X_train
        data.future_snapshots[Re] = X_future
        data.times[Re] = train_times_float
        data.future_times[Re] = future_times_float

        if "cl" in fields:
            t_cl, cl = load_cl(folder)

            mask_train = (t_cl >= training_window[0]) & (t_cl <= training_window[1])
            mask_future = (t_cl >= future_window[0]) & (t_cl <= future_window[1])

            data.cl_train[Re] = cl[mask_train]
            data.cl_train_times[Re] = t_cl[mask_train]

            data.cl_future[Re] = cl[mask_future]
            data.cl_future_times[Re] = t_cl[mask_future]

        print(f"Loaded training and future snapshots for Re={Re}")

    return data


def load_test_snapshots(Re, path, mask_box, FOAMDataloader,
                        future_times, sampling_step=1,
                        fields=("U",)):

    data = test_snapshots()

    loader = FOAMDataloader(path)
    vertices = loader.vertices[:, :2]

    mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(mask.numpy())[0]

    data.mask = mask
    data.coords = vertices[mask_indices]
    data.num_points = len(mask_indices)

    # future_times is float list from training → convert to exact folder names
    sampled_times_str = [t for t in loader.write_times if float(t) in future_times]
    sampled_times_float = [float(t) for t in sampled_times_str]

    data.test_future_times = sampled_times_float

    X_future, _ = load_masked_matrix(loader, mask_indices, sampled_times_str, fields)
    data.test_future_snapshots = X_future

    if "cl" in fields:
        t_cl, cl = load_cl(path)
        mask_future = (t_cl >= sampled_times_float[0]) & (t_cl <= sampled_times_float[-1])
        data.cl_test = cl[mask_future]
        data.cl_test_times = t_cl[mask_future]

    print(f"Loaded test future snapshots for Re={Re}")

    return data
