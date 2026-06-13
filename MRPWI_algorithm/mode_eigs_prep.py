import numpy as np

"""
Interpolation-based parametric reduced-order models with dynamic mode decomposition.

This module implements the preprocessing stage required for PROM construction
based on the methodology of Du et al. (2024), including mode ranking,
conjugate‑pair handling, and phase realignment of complex DMD modes across
a parameter domain.

Reference:
L. Du, S. Zhang, R. Zhang, S. Zhang,
"Interpolation-based parametric reduced-order models with dynamic mode decomposition",
Eastern Institute for Advanced Study, Eastern Institute of Technology, Ningbo, China;
Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo, China;
School of Ocean and Civil Engineering, Shanghai Jiao Tong University, China.
"""

class mode_eigs_prep:
    """
    Preprocess DMD modes and eigenvalues across a parameter domain.

    This class performs three essential preprocessing operations required
    for parametric DMD, reduced-order modeling, and mode interpolation:

    1. Mode importance sorting
       - Selects the top `rank` modes based on amplitude or integral
         contribution (FlowTorch-style)
       - Supports frequency filtering
       - Ensures consistent ordering across all parameter values

    2. Conjugate‑pair reordering
       - Detects complex conjugate eigenvalue pairs
       - Orders each pair so that the mode with positive imaginary part
         appears first

    3. Phase alignment
       - Aligns the complex phase of each mode to a reference parameter
         (closest to parameter_test)
    """
    
    def __init__(self, parameter_list, parameter_test, modes_dict, eigs_dict, rank):

        self.parameter_list = sorted(parameter_list)
        self.parameter_test = parameter_test
        self.modes_raw      = modes_dict
        self.eigs_raw       = eigs_dict
        self.rank           = rank

        self.sorted_modes   = {}
        self.sorted_eigs    = {}
        self.sorted_indices = {}

        self.paired_modes   = {}
        self.paired_eigs    = {}

        self.aligned_modes  = {}
        self.aligned_eigs   = {}

        self._fitted = False

    def fit(self, dmd_dict, integral=True, f_min=-np.inf, f_max=np.inf):
        self._sort_modes(dmd_dict, integral=integral, f_min=f_min, f_max=f_max)
        self._pair_conjugates()
        self._align_modes()
        self._fitted = True
        return self

    def _sort_modes(self, dmd_dict, integral=True, f_min=-np.inf, f_max=np.inf):

        k = self.rank

        for param in self.parameter_list:

            Phi = self.modes_raw[param]
            s   = self.eigs_raw[param]

            top_idx = (
                dmd_dict[param]
                .top_modes(
                    n=k,
                    integral=integral,
                    f_min=f_min,
                    f_max=f_max
                )
                .detach()
                .cpu()
                .numpy()
            )

            self.sorted_indices[param] = top_idx
            self.sorted_modes[param]   = Phi[:, top_idx]
            self.sorted_eigs[param]    = s[top_idx]

    def _pair_conjugates(self):

        for param in self.parameter_list:

            Phi = self.sorted_modes[param]
            s   = self.sorted_eigs[param]
            k   = len(s)

            used = np.zeros(k, dtype=bool)
            modes_ordered = []
            eigs_ordered  = []

            for j in range(k):
                if used[j]:
                    continue

                eig = s[j]
                conj_idx = np.where(np.isclose(s, np.conj(eig)))[0]

                if len(conj_idx) > 0 and conj_idx[0] != j:
                    m = conj_idx[0]
                    used[j] = used[m] = True

                    if np.imag(eig) >= 0:
                        eigs_ordered.extend([eig, s[m]])
                        modes_ordered.extend([Phi[:, j], Phi[:, m]])
                    else:
                        eigs_ordered.extend([s[m], eig])
                        modes_ordered.extend([Phi[:, m], Phi[:, j]])
                else:
                    used[j] = True
                    eigs_ordered.append(eig)
                    modes_ordered.append(Phi[:, j])

            self.paired_modes[param] = np.column_stack(modes_ordered)
            self.paired_eigs[param]  = np.array(eigs_ordered)

    def _align_modes(self):
        self.param_ref = min(self.parameter_list, key=lambda p: abs(p - self.parameter_test))
        Phi_ref = self.paired_modes[self.param_ref]

        for param in self.parameter_list:

            Phi = self.paired_modes[param]
            aligned = []

            for j in range(Phi.shape[1]):
                angle = np.angle(np.vdot(Phi_ref[:, j], Phi[:, j]))
                aligned.append(Phi[:, j] * np.exp(-1j * angle))

            self.aligned_modes[param] = np.column_stack(aligned)
            self.aligned_eigs[param]  = self.paired_eigs[param]


    @property
    def sorted(self):
        self._check_fitted()
        return self.sorted_modes, self.sorted_eigs

    @property
    def paired(self):
        self._check_fitted()
        return self.paired_modes, self.paired_eigs

    @property
    def aligned(self):
        self._check_fitted()
        return self.aligned_modes, self.aligned_eigs

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing results.")
