"""
buffet_snapshots.py

Loads masked OpenFOAM fields for buffet training and test datasets.
Supports selectable fields: U, p, cl.
"""

import numpy as np
from pathlib import Path


class buffet_training_snapshots:
    def __init__(self):
        self.snapshots = {}
        self.times = {}
        self.mask = None
        self.coords = None
        self.num_points = None
        self.mask_indices = None      
        self.cl_train = {}
        self.cl_train_times = {}
        self.cl_raw = None
        self.cl_uniform = None


class buffet_test_snapshots:
    def __init__(self):
        self.snapshots = None
        self.times = None
        self.coords = None
        self.num_points = None
        self.cl_test = None
        self.cl_test_times = None
        self.cl_raw = None
        self.cl_uniform = None


# Utility: masked matrix loader
def load_masked_matrix(loader, mask_indices, times, fields=("U",)):
    field_dims = {"U": 2, "p": 1}
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


# TRAINING LOADER
def load_buffet_training_snapshots(
    param_list,
    base_path,
    mask_box,
    FOAMDataloader,
    training_window=(0.6, 0.8),
    sampling_step=1,
    fields=("U", "p", "cl"),
    interpolate_cl=True,
    interpolate_uniform_fn=None,
    load_force_coeffs_fn=None
):

    data = buffet_training_snapshots()

    # Compute mask
    folder0 = Path(base_path).expanduser() / f"buffet_alpha{param_list[0]}deg"
    loader0 = FOAMDataloader(str(folder0))
    vertices0 = loader0.vertices[:, [0, 2]]

    mask = mask_box(vertices0, lower=[-0.15, -0.2], upper=[2.5, 0.75])
    mask_indices = np.where(mask.numpy())[0]

    data.mask = mask
    data.coords = vertices0[mask_indices]
    data.num_points = len(mask_indices)
    data.mask_indices = mask_indices      

    # Loop over all training parameters
    for alpha in param_list:

        folder = Path(base_path).expanduser() / f"buffet_alpha{alpha}deg"
        loader = FOAMDataloader(str(folder))

        times_str = loader.write_times
        times_float = np.array([float(t) for t in times_str])

        mask_train = (times_float >= training_window[0]) & (times_float <= training_window[1])
        train_times_str = list(np.array(times_str)[mask_train][::sampling_step])
        train_times_float = list(times_float[mask_train][::sampling_step])

        X_train, _ = load_masked_matrix(loader, mask_indices, train_times_str, fields)

        data.snapshots[alpha] = X_train
        data.times[alpha] = train_times_float

        if "cl" in fields:
            coeffs = load_force_coeffs_fn(folder)
            t_cl_raw = coeffs["t"].values
            cl_raw   = coeffs["cy"].values

            if interpolate_cl:
                t_cl_uni, cl_uni = interpolate_uniform_fn(t_cl_raw, cl_raw)
            else:
                t_cl_uni, cl_uni = t_cl_raw, cl_raw

            cl_train = np.interp(train_times_float, t_cl_uni, cl_uni)

            data.cl_train[alpha] = cl_train
            data.cl_train_times[alpha] = train_times_float

        
            data.cl_raw = (t_cl_raw, cl_raw)
            data.cl_uniform = (t_cl_uni, cl_uni)

        print(f"Loaded training snapshots for α={alpha}° {X_train.shape}")

    return data


# TEST LOADER
def load_buffet_test_snapshots(
    param,
    base_path,
    FOAMDataloader,
    train_times,
    mask_indices,
    coords,
    num_points,
    fields=("U", "p", "cl"),
    interpolate_cl=True,
    interpolate_uniform_fn=None,
    load_force_coeffs_fn=None
):

    data = buffet_test_snapshots()

    folder = Path(base_path).expanduser() / f"buffet_alpha{param}deg"
    loader = FOAMDataloader(str(folder))

    data.coords = coords
    data.num_points = num_points

    # Match times
    times_str = loader.write_times
    test_times_str = [t for t in times_str if float(t) in train_times]

    # Load masked fields
    X_test, _ = load_masked_matrix(loader, mask_indices, test_times_str, fields)
    data.snapshots = X_test
    data.times = train_times

    # Load Cl
    if "cl" in fields:
        coeffs = load_force_coeffs_fn(folder)
        t_cl_raw = coeffs["t"].values
        cl_raw   = coeffs["cy"].values

        if interpolate_cl:
            t_cl_uni, cl_uni = interpolate_uniform_fn(t_cl_raw, cl_raw)
        else:
            t_cl_uni, cl_uni = t_cl_raw, cl_raw

        cl_test = np.interp(train_times, t_cl_uni, cl_uni)

        data.cl_test = cl_test
        data.cl_test_times = train_times

    
        data.cl_raw = (t_cl_raw, cl_raw)
        data.cl_uniform = (t_cl_uni, cl_uni)

    print(f"Loaded test snapshots for α={param}° {X_test.shape}")

    return data
