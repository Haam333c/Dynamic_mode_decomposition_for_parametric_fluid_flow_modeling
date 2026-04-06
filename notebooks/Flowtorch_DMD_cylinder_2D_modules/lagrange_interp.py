import numpy as np

class lagrange_interp:
    """
    Interpolate aligned DMD modes, eigenvalues, and initial conditions
    across a parameter domain.
    """

    def __init__(self, parameter_list, parameter_test,
                 aligned_modes, aligned_eigs, x_0,
                 stencil_sizes=[2, 4, 6]):

        self.parameter_list = sorted(parameter_list)
        self.parameter_test = parameter_test

        self.modes_aligned = aligned_modes
        self.eigs_aligned  = aligned_eigs
        self.x_0_train     = x_0          

        self.stencil_sizes = stencil_sizes
        self.stencils      = {}

        self.interp_modes  = {}
        self.interp_eigs   = {}
        self.interp_x_0    = {}

        self._basis_weights = {}     # L_i(parameter_test)
        self._basis_polys   = {}     # L_i(mu) as poly1d

        self._fitted = False

    def _build_stencils(self):
        sorted_neighbors = sorted(
            self.parameter_list,
            key=lambda Re: abs(Re - self.parameter_test)
        )

        for Np in self.stencil_sizes:
            if Np % 2 == 1:
                Np -= 1
            stencil = sorted(sorted_neighbors[:Np])
            self.stencils[Np] = stencil

    def _lagrange(self, parameter_test, pts, values):
        Np = len(pts)
        result = np.zeros_like(values[0], dtype=complex)

        for i in range(Np):
            L_i = 1.0
            for j in range(Np):
                if j != i:
                    L_i *= (parameter_test - pts[j]) / (pts[i] - pts[j])
            result += L_i * values[i]

        return result

    def _lagrange_basis(self, parameter_test, pts):
        Np = len(pts)
        L = np.zeros(Np, dtype=float)

        for i in range(Np):
            L_i = 1.0
            for j in range(Np):
                if j != i:
                    L_i *= (parameter_test - pts[j]) / (pts[i] - pts[j])
            L[i] = L_i

        return L

    def _lagrange_polynomials(self, pts):
        Np = len(pts)
        polys = []

        for i in range(Np):
            poly = np.poly1d([1.0])
            denom = 1.0

            for j in range(Np):
                if j != i:
                    poly *= np.poly1d([1.0, -pts[j]])
                    denom *= (pts[i] - pts[j])

            polys.append(poly / denom)

        return polys

    def _interp_modes(self, pts):
        Re0 = pts[0]
        n_space, r = self.modes_aligned[Re0].shape
        Phi_interp = np.zeros((n_space, r), dtype=complex)

        for j in range(r):
            values = [self.modes_aligned[Re][:, j] for Re in pts]
            Phi_interp[:, j] = self._lagrange(self.parameter_test, pts, values)

        return Phi_interp

    def _interp_eigs(self, pts):
        Re0 = pts[0]
        r = self.eigs_aligned[Re0].shape[0]
        s_interp = np.zeros(r, dtype=complex)

        for j in range(r):
            values = [self.eigs_aligned[Re][j] for Re in pts]
            s_interp[j] = self._lagrange(self.parameter_test, pts, values)

        return s_interp

    def _interp_x_0(self, pts):
        
        values = [self.x_0_train[Re] for Re in pts]
        return self._lagrange(self.parameter_test, pts, values)

    def fit(self):
        self._build_stencils()

        for Np, pts in self.stencils.items():
            self.interp_modes[Np] = self._interp_modes(pts)
            self.interp_eigs[Np]  = self._interp_eigs(pts)
            self.interp_x_0[Np]   = self._interp_x_0(pts)

            self._basis_weights[Np] = self._lagrange_basis(self.parameter_test, pts)
            self._basis_polys[Np]   = self._lagrange_polynomials(pts)

        self._fitted = True
        return self

    @property
    def modes(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing interpolated modes.")
        return self.interp_modes

    @property
    def eigs(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing interpolated eigenvalues.")
        return self.interp_eigs

    @property
    def x_0(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing interpolated x_0.")
        return self.interp_x_0

    @property
    def basis(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing basis weights.")
        return self._basis_weights

    @property
    def polynomials(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing polynomials.")
        return self._basis_polys
