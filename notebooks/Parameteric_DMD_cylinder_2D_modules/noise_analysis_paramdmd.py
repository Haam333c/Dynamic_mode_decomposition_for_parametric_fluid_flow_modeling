import numpy as np
import matplotlib.pyplot as plt


def add_noise_to_snapshots(snapshot_dict, Re_list, noise_level):
    """
    Adds signal-scaled Gaussian noise directly to each snapshot matrix X.

    Parameters:
    - snapshot_dict: dict {Re: array(space_dim, n_time)}
    - Re_list: list of Reynolds numbers
    - noise_level: percent noise (0–100)

    Returns:
    - snapshot_noisy_dict: dict {Re: noisy array(space_dim, n_time)}
    """

    def apply_noise_numpy(X, noise_level):
        # X is (space_dim, n_time)
        std = np.std(X, axis=1, keepdims=True)  # per spatial point
        noise = np.random.randn(*X.shape) * std * (noise_level * 0.01)
        return X + noise

    snapshot_noisy_dict = {}

    for Re in Re_list:
        X = snapshot_dict[Re]  # THIS is the matrix receiving noise
        if noise_level == 0:
            snapshot_noisy_dict[Re] = X.copy()
        else:
            snapshot_noisy_dict[Re] = apply_noise_numpy(X, noise_level)

    return snapshot_noisy_dict



def visualize_noise_levels(snapshot_noisy_dict_versions,
                           snapshot_clean,
                           sampled_times_dict,
                           Re_value,
                           noise_levels,
                           colors):
    """
    Creates a 2x2 subplot visualization of clean vs noisy snapshot magnitudes
    for multiple noise levels.

    Parameters:
    - snapshot_dict_noisy_versions: dict {noise_level: {Re: array(space_dim, n_time)}}
    - snapshot_clean: array (space_dim, n_time)
    - sampled_times_dict: dict {Re: list/array of times}
    - Re_value: Reynolds number being visualized
    - noise_levels: list of 4 noise levels to plot
    - colors: dict {noise_level: color_string}
    """

    # Extract time vector internally
    times = np.array(sampled_times_dict[Re_value], dtype=float)

    # Clean magnitude
    clean_mag = np.sqrt(np.sum(snapshot_clean**2, axis=0))

    # Compute global y-limits
    y_min = clean_mag.min()
    y_max = clean_mag.max()

    noisy_mags = {}

    for nl in noise_levels:
        noisy = snapshot_noisy_dict_versions[nl][Re_value].copy()

        # Normalization
        for i in range(noisy.shape[0]):
            clean_norm = np.linalg.norm(snapshot_clean[i])
            noisy_norm = np.linalg.norm(noisy[i])
            noisy[i] *= clean_norm / noisy_norm

        noisy_mag = np.sqrt(np.sum(noisy**2, axis=0))
        noisy_mags[nl] = noisy_mag

        y_min = min(y_min, noisy_mag.min())
        y_max = max(y_max, noisy_mag.max())

    # 2x2 subplot grid
    fig, axarr = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axarr = axarr.flatten()

    for j, nl in enumerate(noise_levels):
        noisy_mag = noisy_mags[nl]
        ax = axarr[j]

        # Clean magnitude (dashed)
        ax.plot(times, clean_mag, lw=2, ls="--", c="k", label="Clean Magnitude")

        # Noisy magnitude
        ax.plot(times, noisy_mag, lw=2, c=colors[nl],
                label=f"Noisy Magnitude ({nl}%)", alpha=0.85)

        ax.set_title(f"Noise Level = {nl}%", fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
        ax.tick_params(labelsize=12)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(y_min, y_max)

    # Collect legend entries from ALL subplots
    handles = []
    labels = []
    for ax in axarr:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates while preserving order
    unique = dict(zip(labels, handles))
    labels = list(unique.keys())
    handles = list(unique.values())

    # Title
    fig.suptitle(f"Noisy Training Snapshot Magnitude \n $Re$ ={Re_value}", fontsize=18, y=1.15)

    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.05), fontsize=12)

    fig.supxlabel("Time", y=0.02, fontsize=14)
    fig.text(-0.03, 0.5, "Velocity Magnitude", va='center',
             rotation='vertical', fontsize=14)

    plt.tight_layout()
    plt.show()






