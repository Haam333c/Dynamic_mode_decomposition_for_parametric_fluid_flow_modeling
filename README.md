# Dynamic Mode Decomposition for Parametric Fluid Flow Modeling
### Spectrally‑Consistent Interpolation via Mode‑Realigned Pointwise Interpolation (MRPWI)

This repository accompanies the research work:

*Parametric Dynamic Mode Decomposition for Parametric Fluid Flow Modeling*

It provides a complete, reproducible workflow for constructing **parametric reduced‑order models (PROMs)** using Dynamic Mode Decomposition (DMD). The focus is on:

- Spectrally consistent preprocessing of DMD modes  
- **Mode‑Realigned Pointwise Interpolation (MRPWI)**  
- Lagrange interpolation of eigenvalues  
- Interpolation of initial conditions ($x_0$)  
- Evaluation against **PyDMD’s ParametricDMD** implementation  

The repository includes tools for both **laminar cylinder flow** and **transonic shock buffet** analysis.

---

### `MRPWI_algorithm/`
Core algorithms for parametric DMD:

- `mode_eigs_prep.py` — mode ranking, conjugate‑pair detection, phase alignment  
- `lagrange_interp.py` — Lagrange interpolation of aligned modes, eigenvalues, and x₀  

These scripts form the backbone of the MRPWI PROM pipeline.

---

### `notebooks/`
Interactive Jupyter notebooks demonstrating:

- Cylinder flow PROM construction
- **PyDMD’s ParametricDMD** used as the baseline comparison model 
- Preprocessing, alignment, and interpolation of spectral data
- MRPWI evaluation at unseen parameters   
- Flow reconstruction, forecasting, and error analysis  

This folder contains the full implementation of the evaluation workflow.

---

### `snapshot_loader/`
Utilities for loading CFD data:

- Masked field extraction  
- Coordinate and cell‑area loading  
- Snapshot matrix construction  
- `loader_buffet.py` for transonic buffet datasets  
- Cylinder loader for laminar wake analysis  

---

## Test Cases
Example CFD setups and data structures used for:

- 2D-Cylinder flow 
- Transonic buffet flow  

These cases allow users to reproduce the results shown in the notebooks.

---

## Methodological Basis

This work builds on and extends the interpolation methodology introduced in:

**L. Du, S. Zhang, R. Zhang, S. Zhang**  
*Interpolation‑based parametric reduced‑order models with dynamic mode decomposition*  
Eastern Institute for Advanced Study / Shanghai Jiao Tong University (2024)

Your implementation includes:

- FlowTorch‑style mode ranking  
- Conjugate‑pair reordering  
- Phase alignment across parameters  
- **MRPWI** for pointwise interpolation of complex DMD modes  
- Lagrange interpolation of eigenvalues  
- **Interpolation of initial conditions** (extension beyond the original paper)  

---


## Dependencies

- PyDMD ParametricDMD 
  - References:  
    Demo, N., Tezzele, M., & Rozza, G. (2018). *PyDMD: Python Dynamic Mode Decomposition.*  
    Journal of Open Source Software, 3(22), 530.  
    DOI: https://doi.org/10.21105/joss.00530  

    Ichinaga, D., Andreuzzi, F., Demo, N., Tezzele, M., Lapo, E., Rozza, G., Brunton, S. L., & Kutz, J. N. (2024).  
    *PyDMD: A Python Package for Robust Dynamic Mode Decomposition.*  
    Journal of Machine Learning Research.  
    DOI: https://www.jmlr.org/papers/v25/23-0872.html  
    arXiv: https://arxiv.org/abs/2307.07862

- FlowTorch DMD 
  - Reference:  
    Weiner, A., & Semaan, R. (2021). *flowTorch – a Python library for analysis and reduced‑order modeling of fluid flows*.  
    Journal of Open Source Software, 6(68), 3860.  
    https://doi.org/10.21105/joss.03860  
    Repository: https://github.com/AndreWeiner/flowtorch

- Cylinder case data  
  - Based on the open‑source CFD teaching repository:  
    https://github.com/AndreWeiner/ml-cfd-lecture

- Transonic buffet case  
  - Based on the open‑source buffet setup:  
    https://github.com/JanisGeise/buffet_oat15

- NumPy, SciPy, scikit‑learn  
- Matplotlib, Seaborn  
- OpenFOAM (for CFD data generation)

---

## Reproducibility

The workflow is designed to be:

- Deterministic  
- Reproducible  
- Modular  
- Solver‑agnostic  

---


