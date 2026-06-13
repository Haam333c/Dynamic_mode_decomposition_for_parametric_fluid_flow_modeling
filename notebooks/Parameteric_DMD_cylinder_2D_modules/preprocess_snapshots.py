import numpy as np

def preprocess_snapshots(snapshot_dict, Re_list):
    """
    Preprocess velocity snapshots by subtracting the global training mean flow.

    Steps:
    1. Stack all training snapshots across Reynolds numbers and time.
    2. Compute the global mean flow from the stacked training data.
    3. Subtract this mean flow from each training snapshot block.
    4. Stack the mean-subtracted training snapshots into a 3D array.

    Parameters:
    snapshot_dict : dict
        Raw velocity snapshots per Reynolds number (training).
    Re_list : list
        Reynolds numbers used for training.

    Returns:
    train_snapshots : ndarray
        Array of shape (n_Re, space_dim, n_time) with mean-subtracted training snapshots.
    mean_flow_train : ndarray
        Global mean flow vector from training data (space_dim,).
    snapshot_processed_dict : dict
        Dictionary of mean-subtracted snapshots per Re.
    """
    # Stack all training snapshots to compute global mean
    all_training_snapshots = np.concatenate(
        [snapshot_dict[Re].T for Re in Re_list],
        axis=0
    )
    mean_flow_train = np.mean(all_training_snapshots, axis=0)

    # Subtract training mean from each training block
    snapshot_processed_dict = {}
    for Re in Re_list:
        snapshots = snapshot_dict[Re].copy()
        snapshots -= mean_flow_train[:, None]
        snapshot_processed_dict[Re] = snapshots

    # Stack into array for PDMD
    train_snapshots = np.array([snapshot_processed_dict[Re] for Re in Re_list])

    return train_snapshots, mean_flow_train, snapshot_processed_dict



def compute_average_inlet_velocity(loader, tol=1e-3, time=None):
    """
    Computes the average inlet velocity magnitude at a given time by automatically
    detecting the inlet as the face with the minimum x-coordinate.

    Parameters:
    - loader: FOAMDataloader instance for a given Re case
    - tol: tolerance for identifying inlet points near the minimum x-coordinate
    - time: time step to extract velocity from (default: first available)

    Returns:
    - U_infty: average velocity magnitude at the inlet
    """
    if time is None:
        time = loader.write_times[0]  # use first available time if not specified

    snapshot = loader.load_snapshot("U", time)  # shape: (n_points, 3)
    vertices = loader.vertices[:, :2]  # shape: (n_points, 2)

    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()

    inlet_x = np.min(vertices[:, 0])
    inlet_mask = np.abs(vertices[:, 0] - inlet_x) < tol
    if not inlet_mask.any():
        raise ValueError(f"No inlet points found at x ≈ {inlet_x:.6f} (tol={tol})")

    u_inlet = snapshot[inlet_mask, 0]
    v_inlet = snapshot[inlet_mask, 1]
    if hasattr(u_inlet, "detach"):
        u_inlet = u_inlet.detach().cpu().numpy()
        v_inlet = v_inlet.detach().cpu().numpy()

    U_infty = np.mean(np.sqrt(u_inlet**2 + v_inlet**2))
    return U_infty
