import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FixedLocator

import seaborn as sns
from sklearn.utils.extmath import randomized_svd

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'   # Computer Modern


def plot_snapshot_magnitudes(snapshot_dict, sampled_times_dict, Re_list):
    """
    Plots raw snapshot velocity magnitudes over time for each Reynolds number.
    """
    n_re = len(Re_list)
    fig, axes = plt.subplots(n_re, 1, figsize=(12, 3 * n_re), sharex=True)

    for i, Re in enumerate(Re_list):
        # Global snapshot magnitudes (L2 norm over space at each sampled time)
        mags = np.linalg.norm(snapshot_dict[Re], axis=0)

        # Plot
        times = np.array(sampled_times_dict[Re], dtype=float)
        ax = axes[i]
        ax.plot(times, mags)
        ax.set_ylabel("Velocity Magnitude", fontsize=12)
        ax.set_title(f"$Re$ = {Re}", fontsize=14, pad=6)
        ax.grid(True, alpha=0.6)
        ax.legend(fontsize=16)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("$t$ (Time in seconds)", fontsize=16)
    plt.suptitle("Snapshot Velocity Magnitudes for DMD Training Parameters",
                 fontsize=18, y = 0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compute_pod(snapshot_dict, Re_list, n_components=100):
    snapshot_matrix = np.hstack([snapshot_dict[Re] for Re in Re_list])
    U, s, Vh = randomized_svd(snapshot_matrix, n_components=n_components)
    normalized_energy = s**2 / np.sum(s**2)
    cumulative_energy = np.cumsum(normalized_energy)
    residual_content = 1 - cumulative_energy
    return cumulative_energy, residual_content

def get_thresholds(residual_content, threshold=0.99, tau_list=None):
    if tau_list is None:
        tau_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    num_modes_99 = np.searchsorted(1 - residual_content, threshold) + 1
    tau_ranks = [(tau, np.where(residual_content > tau)[0][-1] + 1) for tau in tau_list]
    return num_modes_99, tau_ranks


def plot_cumulative_energy(cumulative_energy, threshold, num_modes_99):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative_energy) + 1), cumulative_energy, marker='o', label='Cumulative Energy')
    plt.axhline(threshold, color='red', linestyle='--', label='99% Threshold')
    plt.axvline(num_modes_99, color='green', linestyle='--', label=f'{num_modes_99} Modes')
    plt.text(0.65, 0.15,
             f"Modes for 99% energy: {num_modes_99}",
             transform=plt.gca().transAxes,
             fontsize=10, color='green')

    x_margin = 10
    x_min = max(1, num_modes_99 - x_margin)
    x_max = min(len(cumulative_energy), num_modes_99 + x_margin)
    plt.xlim(x_min, x_max)
    plt.ylim(threshold - 0.05, 1.01)

    plt.title("Cumulative Energy Retained by POD Modes", fontsize=16, pad=10)
    plt.xlabel("Number of Modes")
    plt.ylabel("Cumulative Energy")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_residual_energy(residual_content, tau_ranks):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(residual_content) + 1), residual_content, marker='o', color='tab:blue', label='Residual Energy')

    colors = plt.cm.viridis(np.linspace(0, 1, len(tau_ranks)))
    box_x = 0.75
    box_y = 0.95
    line_spacing = 0.06

    for i, (_, idx) in enumerate(tau_ranks):
        y_val = residual_content[idx - 1]
        ax.plot(idx, y_val, 'o', color=colors[i], markersize=8)
        ax.plot([box_x], [box_y - i * line_spacing], marker='o', color=colors[i],
                transform=ax.transAxes, markersize=6)
        ax.text(box_x + 0.02, box_y - i * line_spacing,
                f"Rank {idx}: Residual = {y_val:.1e}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='center',
                color='black')

    ax.set_title("Residual Content (Cumulative Energy)", fontsize=16, pad=10)
    ax.set_xlabel("Number of Modes")
    ax.set_ylabel("Residual Energy")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_dmd_modal_comparison(
    pdmd, Re_list, sampled_times_dict,
    Re_value, U_ref_dict, L_ref,
    rom, snapshot_processed_dict, n_modes_to_plot=5):

    """
    Plots a comparison of true vs ParametricDMD modal coefficients
    for a given Reynolds number, using non-dimensional time.
    """
    if Re_value not in Re_list:
        raise ValueError(f"$Re$ = {Re_value} not found in Re_list")

    # Index of target Re in training list
    index = np.where(np.array(Re_list) == Re_value)[0][0]
    times = np.array(sampled_times_dict[Re_value], dtype=float)

    # Non-dimensionalize time
    U_ref = U_ref_dict[Re_value]
    time_nd = (times - times[0]) * U_ref / L_ref

    # Extract modal coefficients
    modal_true = rom.transform(snapshot_processed_dict[Re_value])
    modal_dmd  = pdmd.training_modal_coefficients[index][:, :modal_true.shape[1]]

    # Create subplots
    fig, axes = plt.subplots(n_modes_to_plot, 1,
                             figsize=(12, 2.8 * n_modes_to_plot),
                             sharex=True)

    for mode in range(n_modes_to_plot):
        ax = axes[mode]
        ax.plot(time_nd, modal_true[mode],
                color="tab:blue", lw=1.8, label="True")
        ax.plot(time_nd, modal_dmd[mode],
                color="tab:orange", lw=1.8, linestyle="--", label="Trained DMD")

        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title(f"Mode $\\Phi_{{{mode}}}$", fontsize=13, pad=6)
        ax.grid(alpha=0.6)
        ax.legend(fontsize=11, loc="upper right")
        ax.set_xlim(time_nd[0], time_nd[-1])
        ax.set_xticks(np.linspace(time_nd[0], time_nd[-1], 6))

    axes[-1].set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)

    fig.align_ylabels(axes)
    plt.suptitle(
        f"Modal Coefficient Dynamics — "
        f"True vs Trained ParametricDMD \n Training Parameter $(Re={Re_value})$",
        fontsize=16, y=0.97
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def plot_dmd_fft_comparison(
    pdmd, Re_list, Re_target, L_ref, nu,
    rom, snapshot_processed_dict, n_plot=4, dt=0.01, ref_St=0.2):

    # Index of target Re in training list
    index = np.where(np.array(Re_list) == Re_target)[0][0]

    # Extract modal coefficients
    modal_true = rom.transform(snapshot_processed_dict[Re_target])
    modal_dmd  = pdmd.training_modal_coefficients[index][:, :modal_true.shape[1]]

    # Frequency axis
    n_time = modal_true.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=dt)

    # Reference velocity
    U_ref = Re_target * nu / L_ref

    # Strouhal number
    St = freqs * L_ref / U_ref

    # Create subplots
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.5 * n_plot), sharex=True)
    fig.suptitle(
        f"Frequency Spectrum of Modal Coefficient Dynamics — True vs Trained ParametricDMD\n"
        f"Training Parameter $(Re = {Re_target})$",
        fontsize=16, y=0.96
    )

    for mode in range(n_plot):
        modal_true_clean = np.asarray(modal_true[mode], dtype=float)
        modal_dmd_clean  = np.asarray(modal_dmd[mode], dtype=float)

        fft_true = np.abs(np.fft.rfft(modal_true_clean))
        fft_dmd  = np.abs(np.fft.rfft(modal_dmd_clean))

        ax = axes[mode]
        ax.plot(St, fft_true, color="tab:blue")
        ax.plot(St, fft_dmd, color="tab:orange", linestyle="--")
        ax.set_ylabel("Spectral Amplitude", fontsize=12)
        ax.grid(True)
        ax.set_title(f"Mode $\\Phi_{{{mode}}}$", fontsize=12)
        ax.legend(["True", "Trained DMD"])

    axes[-1].set_xlabel(r"$St = f L_{ref} / U_{ref}$", fontsize=13)
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def plot_flow_comparison_dmd_vs_true(
    Re_target, 
    Re_list,
    t_start,
    t_end,
    granularity,
    rom,
    pdmd,
    mean_flow,   
    sampled_times_dict,
    snapshot_dict,
    masked_coords_dict,
    num_points_dict,
    L_ref=0.1,
    cylinderX=0.2,
    cylinderY=0.2,
    radius=0.05,
    cmap='icefire'
):
    # Colormap setup
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)

    # Time setup
    i = np.where(Re_list == Re_target)[0][0]
    time_vec = np.array(sampled_times_dict[Re_target], dtype=float)

    # Physical times chosen by the user
    selected_times = np.arange(t_start, t_end + 1e-9, granularity)

    # Vectorized nearest-index mapping (no manual matching)
    t_indices = np.searchsorted(time_vec, selected_times)

    # Clip to valid range
    t_indices = np.clip(t_indices, 0, len(time_vec) - 1)

    # Spatial setup
    coords = masked_coords_dict[Re_target] / L_ref
    num_points = num_points_dict[Re_target]
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1])

    # Trained DMD modal coefficients
    modal_dmd = pdmd.training_modal_coefficients[i]

    # Precompute fields
    fields_per_row = []
    for t_idx in t_indices:
        coeff_t = modal_dmd[:, t_idx]

        # Correct reconstruction
        U_dmd = rom.expand(coeff_t) + mean_flow
        U_true = snapshot_dict[Re_target][:, t_idx]

        ux_dmd, uy_dmd = np.real(U_dmd[:num_points]), np.real(U_dmd[num_points:])
        ux_true, uy_true = np.real(U_true[:num_points]), np.real(U_true[num_points:])

        mag_dmd = np.sqrt(ux_dmd**2 + uy_dmd**2)
        mag_true = np.sqrt(ux_true**2 + uy_true**2)
        residual = mag_true - mag_dmd

        fields_per_row.append((mag_true, mag_dmd, residual, time_vec[t_idx]))

    # Global scaling
    mag_true_all = np.concatenate([f[0] for f in fields_per_row])
    mag_dmd_all = np.concatenate([f[1] for f in fields_per_row])
    global_min = min(mag_true_all.min(), mag_dmd_all.min())
    global_max = max(mag_true_all.max(), mag_dmd_all.max())
    fixed_ticks_main = np.linspace(global_min, global_max, 10)
    sm_shared = ScalarMappable(norm=Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
    sm_shared.set_array([])

    residual_all = np.concatenate([np.abs(f[2]) for f in fields_per_row])
    resid_min, resid_max = residual_all.min(), residual_all.max()
    fixed_ticks_resid = np.linspace(resid_min, resid_max, 10)
    sm_resid = ScalarMappable(norm=Normalize(vmin=resid_min, vmax=resid_max), cmap=cmap)
    sm_resid.set_array([])

    # Plotting
    n_rows = len(fields_per_row)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3 * n_rows))
    fig.suptitle(f"Flow Comparison — True vs ParametricDMD\n Training Parameter $(Re = {Re_target})$", fontsize=18, y=0.96)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (mag_true, mag_dmd, residual, t_val) in enumerate(fields_per_row):
        # True
        ax_true = axes[row, 0]
        ax_true.tricontourf(tri, mag_true, levels=200, cmap=cmap, vmin=global_min, vmax=global_max)
        ax_true.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_true.set_aspect("equal")
        fig.colorbar(sm_shared, ax=ax_true, shrink=1, pad=0.02, ticks=fixed_ticks_main)
        ax_true.text(-0.30, 0.5, f"$t$ = {t_val:.2f} s",
                     transform=ax_true.transAxes,
                     fontsize=16, rotation=90,
                     va="center", ha="center",
                     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

        # DMD
        ax_dmd = axes[row, 1]
        ax_dmd.tricontourf(tri, mag_dmd, levels=200, cmap=cmap, vmin=global_min, vmax=global_max)
        ax_dmd.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_dmd.set_aspect("equal")
        fig.colorbar(sm_shared, ax=ax_dmd, shrink=1, pad=0.02, ticks=fixed_ticks_main)

        # Residual
        ax_res = axes[row, 2]
        ax_res.tricontourf(tri, residual, levels=200, cmap=cmap, vmin=resid_min, vmax=resid_max)
        ax_res.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_res.set_aspect("equal")
        fig.colorbar(sm_resid, ax=ax_res, shrink=1, pad=0.02, ticks=fixed_ticks_resid)

        for ax in [ax_true, ax_dmd, ax_res]:
            ax.set_xlabel(r"$x/L_{ref}$", fontsize=12)
            ax.set_ylabel(r"$y/L_{ref}$", fontsize=12)
            ax.tick_params(labelbottom=True, labelleft=True)

    axes[0, 0].set_title("True Magnitude", fontsize=15, pad=12)
    axes[0, 1].set_title("ParametricDMD", fontsize=15, pad=12)
    axes[0, 2].set_title(r"Residual $(U_{\text{True}} - U_{\text{ParametricDMD}})$", fontsize=15, pad=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    plt.show()





def plot_dmd_reconstruction_error(
    Re_target,
    Re_list,
    rom,
    pdmd,
    mean_flow_train,
    sampled_times_dict,
    snapshot_dict,
    L_ref,
    nu,
    color='tab:orange'
):
    """
    Plot relative L2 reconstruction error for a training Reynolds number using DMD.
    Uses training mean flow (mean-only preprocessing, no normalization).
    Time axis is nondimensionalized: t* = t U_ref / L_ref.
    """

    # Index of target Re in training list
    i = np.where(Re_list == Re_target)[0][0]
    time_vec = np.array(sampled_times_dict[Re_target], dtype=float)

    # Reference velocity for nondimensionalization
    U_ref = Re_target * nu / L_ref
    time_star = (time_vec - time_vec[0]) * U_ref / L_ref

    # True snapshots
    X_true = snapshot_dict[Re_target]

    # Trained DMD modal coefficients
    modal_dmd = pdmd.training_modal_coefficients[i]

    # Reconstruct using POD expansion
    X_recon = rom.expand(modal_dmd) + mean_flow_train[:, None]

    # Compute relative error over time
    abs_error = np.linalg.norm(X_true - X_recon, axis=0)
    rel_error = abs_error / np.linalg.norm(X_true, axis=0)

    # Plot with nondimensional time
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time_star, rel_error, lw=2, color=color)
    ax.set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=13)
    ax.set_title(f"ParametricDMD Reconstruction Error\n Training Parameter $(Re = {Re_target})$", fontsize=15)
    ax.grid(True, alpha=0.6)
    ax.set_xlim(time_star.min(), time_star.max())
    ax.margins(x=0)
    ax.set_xticks(np.linspace(time_star.min(), time_star.max(), 6))
    plt.tight_layout()
    plt.show()




def plot_dmd_forecast_error(
    Re_target,
    Re_list,
    rom,
    pdmd,
    mean_flow_train,
    sampled_times_dict,
    snapshot_future_dict,
    L_ref,
    nu,
    color='tab:blue'
):
    """
    Plot relative L2 forecast error for a training Reynolds number using ParametricDMD.
    Uses training mean flow (mean-only preprocessing, no normalization).
    Time axis is nondimensionalized: t* = t U_ref / L_ref.
    """

    # Index of target Re in training list
    i = np.where(Re_list == Re_target)[0][0]
    time_vec = np.array(sampled_times_dict[Re_target], dtype=float)

    # Reference velocity for nondimensionalization
    U_ref = Re_target * nu / L_ref
    time_star = (time_vec - time_vec[0]) * U_ref / L_ref   # nondimensional time

    # True snapshots
    X_future = snapshot_future_dict[Re_target]
    X_true = X_future


    # Forecasted modal coefficients (from pdmd)
    modal_forecast = pdmd.forecasted_modal_coefficients[i]
    X_forecast_centered = rom.expand(modal_forecast)

    # Add mean back for physical-field error
    X_forecast = X_forecast_centered + mean_flow_train[:, None]


    # Compute relative error over time
    abs_error = np.linalg.norm(X_true - X_forecast, axis=0)
    rel_error = abs_error / np.linalg.norm(X_true, axis=0)
    perc_error = rel_error * 100.0

    # Mean percentage error
    mean_error = perc_error.mean()

    # Plot with nondimensional time
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time_star, rel_error, lw=2, color=color)
    ax.set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=13)
    ax.set_title(f"ParametricDMD Forecast Reconstruction Error\n Training Parameter $(Re = {Re_target})$", fontsize=15)
    ax.grid(True, alpha=0.6)

    # Place percentage error text where legend would be
    ax.text(0.98, 0.95, f"Mean Error = {mean_error:.2f}%",
            transform=ax.transAxes, fontsize=12,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlim(time_star.min(), time_star.max())
    ax.margins(x=0)
    ax.set_xticks(np.linspace(time_star.min(), time_star.max(), 6))
    plt.tight_layout()
    plt.show()


def plot_dmd_modal_comparison_interp_vs_true(
    pdmd,
    snapshot_test,
    loader_test,
    Re_test,
    L_ref,
    nu,
    rom,
    times_test,         
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6
):

    """
    Compare DMD modal coefficients between interpolated and true data
    at a given test Reynolds number, using nondimensional time.
    Assumes test snapshots are already loaded from t0_phys onward.
    """

    # True time vector 
    sampled_times_test = np.array([float(t) for t in times_test], dtype=float)

    # Interpolated modal coefficients
    interp_modal = pdmd.interpolated_modal_coefficients[0]
    nT = interp_modal.shape[1]

    # Interpolated physical time vector
    interp_times = np.arange(nT) * dt_phys + t0_phys

    # Nondimensional time
    U_ref = Re_test * nu / L_ref
    interp_times_star = (interp_times - interp_times[0]) * U_ref / L_ref
    sampled_times_star = (sampled_times_test - sampled_times_test[0]) * U_ref / L_ref

    # Project true snapshots onto POD basis
    true_modal = rom.transform(snapshot_test)

    # Plot
    fig, axes = plt.subplots(n_modes_to_plot, 1,
                             figsize=(12, 2.8 * n_modes_to_plot),
                             sharex=True)

    fig.suptitle(
        f"Modal Coefficient Dynamics — True vs Interpolated ParametricDMD\n"
        f"$Unseen^*$ Parameter $(Re = {Re_test})$",
        fontsize=16, y=0.97
    )

    for mode_idx in range(n_modes_to_plot):
        ax = axes[mode_idx]

        ax.plot(
            sampled_times_star,true_modal[mode_idx].real,color="tab:blue", lw=1.5,label="True")

        ax.plot(
            interp_times_star,interp_modal[mode_idx].real,color="tab:orange", linestyle="--", lw=1.5,
            label="Interpolated ParametricDMD")

        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title(f"Mode $\\Phi_{{{mode_idx}}}$", fontsize=12, pad=6)
        ax.grid(True, alpha=0.6)
        ax.legend(fontsize=11, loc="center right")

    axes[-1].set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=14)
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



def plot_dmd_fft_comparison_interp_vs_true(
    pdmd,
    snapshot_test,
    loader_test,
    Re_test,
    L_ref,
    nu,
    rom,
    times_test, 
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6
):
    """
    FFT comparison of DMD modal coefficients between interpolated and true data
    at a given test Reynolds number, using nondimensional frequency (Strouhal number).
    Assumes test snapshots are already loaded from t0_phys onward.
    """

    # True time vector
    sampled_times_test = np.array([float(t) for t in times_test], dtype=float)

    # Interpolated modal coefficients
    interp_modal = pdmd.interpolated_modal_coefficients[0]
    nT = interp_modal.shape[1]

    # Interpolated physical time vector
    interp_times = np.arange(nT) * dt_phys + t0_phys

    # Nondimensional time
    U_ref = Re_test * nu / L_ref
    interp_times_star = (interp_times - interp_times[0]) * U_ref / L_ref
    sampled_times_star = (sampled_times_test - sampled_times_test[0]) * U_ref / L_ref

    # Project true snapshots onto POD basis
    true_modal = rom.transform(snapshot_test)

    # FFT frequency axis (dimensional)
    freqs = np.fft.rfftfreq(nT, d=dt_phys)

    # Convert to Strouhal number
    St = freqs * L_ref / U_ref

    # Plot FFT comparison
    fig, axes = plt.subplots(n_modes_to_plot, 1,
                             figsize=(10, 2.5 * n_modes_to_plot),
                             sharex=True)

    fig.suptitle(
        f"Frequency Spectrum of Modal Coefficient Dynamics — True vs Interpolated ParametricDMD\n"
        f"$Unseen^*$ Parameter $(Re = {Re_test})$",
        fontsize=16, y=0.96
    )

    for mode_idx in range(n_modes_to_plot):

        true_mode = true_modal[mode_idx].real
        interp_mode = interp_modal[mode_idx].real

        # FFT magnitudes
        fft_true = np.abs(np.fft.rfft(true_mode))
        fft_interp = np.abs(np.fft.rfft(interp_mode))

        ax = axes[mode_idx]
        ax.plot(St, fft_true, color="tab:blue")
        ax.plot(St, fft_interp, color="tab:orange", linestyle="--")

        ax.set_ylabel("Spectral Amplitude", fontsize=12)
        ax.set_title(f"Mode $\\Phi_{{{mode_idx}}}$", fontsize=12)
        ax.grid(True)
        ax.legend(["True", "Interpolated ParametricDMD"], fontsize=11)

    axes[-1].set_xlabel(r"$St = f L_{ref} / U_{ref}$", fontsize=14)
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def plot_flow_comparison_interpolated_dmd_vs_true(
    pdmd,
    snapshot_test,
    sampled_times_test,
    loader_test,
    mask_test,
    num_points_test,
    mean_flow_train,
    Re_test,
    t_start,
    t_end,
    granularity,
    dt_phys=0.01,
    L_ref=0.1,
    cylinderX=0.2,
    cylinderY=0.2,
    radius=0.05,
    cmap="jet"
):
    """
    Compare true vs interpolated ParametricDMD flow fields WITHOUT any alignment,
    slicing, trimming, or time matching. Each dataset is sampled independently.
    """

    # True data time vector (float)
    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    # Restrict true times to window
    mask_true = (sampled_times_test_float >= t_start) & (sampled_times_test_float <= t_end)
    true_indices_all = np.where(mask_true)[0]

    # Interpolated data time vector
    interp_modal = pdmd.interpolated_modal_coefficients[0]
    n_interp = interp_modal.shape[1]
    interp_times = np.arange(n_interp) * dt_phys + t_start

    # Restrict interpolated times to window
    mask_interp = (interp_times >= t_start) & (interp_times <= t_end)
    interp_indices_all = np.where(mask_interp)[0]

    # Select indices independently for true and interpolated
    step = max(1, int(granularity / dt_phys))
    true_indices = true_indices_all[::step]
    interp_indices = interp_indices_all[::step]

    # Number of rows = min length
    num_rows = min(len(true_indices), len(interp_indices))

    # Spatial triangulation
    coords_test = loader_test.vertices[mask_test.numpy(), :] / L_ref
    triang_test = mtri.Triangulation(coords_test[:, 0], coords_test[:, 1])

    # Compute global magnitude ranges
    mag_true_all = np.sqrt(
        snapshot_test[:num_points_test]**2 +
        snapshot_test[num_points_test:]**2
    ).real

    mag_interp_all = np.sqrt(
        (pdmd.reconstructed_data[0][:num_points_test] + mean_flow_train[:num_points_test, None])**2 +
        (pdmd.reconstructed_data[0][num_points_test:] + mean_flow_train[num_points_test:, None])**2
    ).real

    global_min = min(mag_true_all.min(), mag_interp_all.min())
    global_max = max(mag_true_all.max(), mag_interp_all.max())

    sm_shared = ScalarMappable(norm=Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
    sm_shared.set_array([])

    # Residual range
    resid_min, resid_max = 0.0, 0.0

    # Precompute residuals for scaling
    for r in range(num_rows):
        idx_t = true_indices[r]
        idx_i = interp_indices[r]

        u_x_t = snapshot_test[:num_points_test, idx_t]
        u_y_t = snapshot_test[num_points_test:, idx_t]
        mag_t = np.sqrt(u_x_t**2 + u_y_t**2).real

        U_raw = pdmd.reconstructed_data[0][:, idx_i] + mean_flow_train
        u_x_i = U_raw[:num_points_test]
        u_y_i = U_raw[num_points_test:]
        mag_i = np.sqrt(u_x_i**2 + u_y_i**2).real

        resid = np.abs(mag_t - mag_i)
        resid_min = min(resid_min, resid.min())
        resid_max = max(resid_max, resid.max())

    sm_resid = ScalarMappable(norm=Normalize(vmin=resid_min, vmax=resid_max), cmap=cmap)
    sm_resid.set_array([])

    # Plotting
    fig, axes = plt.subplots(num_rows, 3, figsize=(14, 3 * num_rows))
    fig.suptitle(
        f"Flow Comparison — True vs Interpolated ParametricDMD\n$Unseen^*$ Parameter (Re = {Re_test})",
        fontsize=18, y=0.96
    )

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(num_rows):

        idx_t = true_indices[row]
        idx_i = interp_indices[row]

        # TRUE FIELD
        u_x_t = snapshot_test[:num_points_test, idx_t]
        u_y_t = snapshot_test[num_points_test:, idx_t]
        mag_t = np.sqrt(u_x_t**2 + u_y_t**2).real

        # INTERPOLATED FIELD
        U_raw = pdmd.reconstructed_data[0][:, idx_i] + mean_flow_train
        u_x_i = U_raw[:num_points_test]
        u_y_i = U_raw[num_points_test:]
        mag_i = np.sqrt(u_x_i**2 + u_y_i**2).real

        # RESIDUAL
        resid = np.abs(mag_t - mag_i)

        # True plot
        axes[row, 0].tricontourf(triang_test, mag_t, levels=200, cmap=cmap,
                                 vmin=global_min, vmax=global_max)
        axes[row, 0].add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black"))
        axes[row, 0].set_aspect("equal")
        fig.colorbar(sm_shared, ax=axes[row, 0], shrink=1, pad=0.02)

        # Interpolated plot
        axes[row, 1].tricontourf(triang_test, mag_i, levels=200, cmap=cmap,
                                 vmin=global_min, vmax=global_max)
        axes[row, 1].add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black"))
        axes[row, 1].set_aspect("equal")
        fig.colorbar(sm_shared, ax=axes[row, 1], shrink=1, pad=0.02)

        # Residual plot
        axes[row, 2].tricontourf(triang_test, resid, levels=200, cmap=cmap,
                                 vmin=resid_min, vmax=resid_max)
        axes[row, 2].add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black"))
        axes[row, 2].set_aspect("equal")
        fig.colorbar(sm_resid, ax=axes[row, 2], shrink=1, pad=0.02)

    axes[0, 0].set_title("True Magnitude", fontsize=15)
    axes[0, 1].set_title("Interpolated ParametricDMD", fontsize=15)
    axes[0, 2].set_title("Residual", fontsize=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_interp_reconstruction_error(snapshot_test,
                                     times_test,
                                     pdmd,
                                     dt_phys,
                                     Re_test,
                                     mean_flow_train,
                                     L_ref,
                                     nu,
                                     t0_phys=15.0):
    """
    Relative L2 reconstruction error for ParametricDMD at a test Reynolds number.
    NO slicing, NO alignment, NO trimming. True and interpolated signals are used as-is.
    """

    # True CFD snapshots and times (raw)
    t_true = np.array(times_test, dtype=float)
    X_true = snapshot_test

    # Interpolated reconstruction (raw)
    X_interp_raw = pdmd.reconstructed_data[0].real
    X_interp = X_interp_raw + mean_flow_train[:, None]

    # Interpolated time vector (raw)
    n_interp = X_interp.shape[1]
    t_interp = np.arange(n_interp) * dt_phys + t0_phys

    # Nondimensional time
    U_ref = Re_test * nu / L_ref
    t_true_star = (t_true - t_true[0]) * U_ref / L_ref
    t_interp_star = (t_interp - t_interp[0]) * U_ref / L_ref

    # Compute relative error independently for each dataset
    min_len = min(X_true.shape[1], X_interp.shape[1])

    abs_err = np.linalg.norm(X_true[:, :min_len] - X_interp[:, :min_len], axis=0)
    rel_err = abs_err / np.linalg.norm(X_true[:, :min_len], axis=0)
    perc_err = rel_err * 100.0

    # Time vector for error plot (use interpolated time)
    t_star = t_interp_star[:min_len]

    # Mean error
    mean_err = perc_err.mean()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_star, rel_err, lw=2, color="tab:orange")

    ax.set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=13)
    ax.set_title(
        r"ParametricDMD Prediction Error"
        f"\n $Unseen^*$ Parameter $(Re = {Re_test})$",
        fontsize=15
    )
    ax.grid(True, alpha=0.6)

    ax.text(
        0.98, 0.95,
        f"Mean Error = {mean_err:.2f}%",
        transform=ax.transAxes,
        fontsize=12,
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    ax.set_xlim(t_star[0], t_star[-1])
    ax.set_xticks(np.linspace(t_star[0], t_star[-1], 6))

    plt.tight_layout()
    plt.show()



