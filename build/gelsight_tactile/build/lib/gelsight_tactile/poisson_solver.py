"""
Poisson solver using 2D DCT with Neumann boundary conditions.

Adapted from GelSight Inc. gsrobotics (GPL-3.0):
https://github.com/gelsightinc/gsrobotics/blob/main/utilities/poisson_solver.py

Neumann BC enforces gradient continuity at boundaries, which is physically
appropriate for gel surface deformation (no zero-boundary assumption).
"""

import math
import numpy as np
from scipy import fftpack


def poisson_dct_neumann(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Integrate surface gradients into a depth map via 2D Poisson equation
    with non-homogeneous Neumann boundary conditions (DCT-based solver).

    Args:
        gx: Gradient in x direction (horizontal / column), shape (H, W).
        gy: Gradient in y direction (vertical  / row),    shape (H, W).

    Returns:
        depth: Reconstructed depth map, shape (H, W), zero-mean.
    """
    assert gx.shape == gy.shape, "gx and gy must have the same shape"

    num_rows, num_cols = gx.shape

    # ── x-direction divergence ────────────────────────────────────────────────
    next_col = np.r_[1:num_cols, num_cols - 1]
    prev_col = np.r_[0:num_cols - 1, num_cols - 2]
    gxx = gx[:, next_col] - gx[:, prev_col]

    # ── y-direction divergence ────────────────────────────────────────────────
    next_row = np.r_[1:num_rows, num_rows - 1]
    prev_row = np.r_[0:num_rows - 1, num_rows - 2]
    gyy = gy[next_row, :] - gy[prev_row, :]

    # ── Divergence (RHS of Poisson equation) ─────────────────────────────────
    div = gxx + gyy

    # ── Neumann boundary conditions ───────────────────────────────────────────
    b = np.zeros(gx.shape)
    b[0,  1:-1] = -gy[0,  1:-1]   # top
    b[-1, 1:-1] =  gy[-1, 1:-1]   # bottom
    b[1:-1,  0] = -gx[1:-1,  0]   # left
    b[1:-1, -1] =  gx[1:-1, -1]   # right

    f = 1.0 / math.sqrt(2)
    b[0,  0]  = f * (-gy[0,   0] - gx[0,   0])
    b[0,  -1] = f * (-gy[0,  -1] + gx[0,  -1])
    b[-1, 0]  = f * ( gy[-1,  0] - gx[-1,  0])
    b[-1, -1] = f * ( gy[-1, -1] + gx[-1, -1])

    div[0,  1:-1] -= b[0,  1:-1]
    div[-1, 1:-1] -= b[-1, 1:-1]
    div[1:-1,  0] -= b[1:-1,  0]
    div[1:-1, -1] -= b[1:-1, -1]
    div[0,  0]  -= math.sqrt(2) * b[0,  0]
    div[0,  -1] -= math.sqrt(2) * b[0,  -1]
    div[-1, 0]  -= math.sqrt(2) * b[-1, 0]
    div[-1, -1] -= math.sqrt(2) * b[-1, -1]

    # ── DCT → solve in frequency domain → IDCT ───────────────────────────────
    div_dct = fftpack.dct(fftpack.dct(div, norm='ortho').T, norm='ortho').T

    x, y = np.meshgrid(
        np.arange(1, num_cols + 1),
        np.arange(1, num_rows + 1),
    )
    denom = 4.0 * (
        np.sin(0.5 * math.pi * x / num_cols) ** 2
        + np.sin(0.5 * math.pi * y / num_rows) ** 2
    )
    denom[0, 0] = 1.0  # avoid divide-by-zero at DC; will be subtracted as mean

    depth_dct = -div_dct / denom
    depth = fftpack.idct(fftpack.idct(depth_dct, norm='ortho').T, norm='ortho').T

    depth -= depth.mean()
    return depth