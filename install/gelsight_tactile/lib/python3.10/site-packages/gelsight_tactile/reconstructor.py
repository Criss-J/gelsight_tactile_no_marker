"""
reconstructor.py  –  GelSight 3D surface reconstruction

Pipeline:
    frame (BGR, H×W×3)
    → normalize inputs with norm_stats.json
    → GelSightMLP: [R_norm, G_norm, B_norm, X/640, Y/480] → [nx, ny]
    → nz = sqrt(1 - nx² - ny²)
    → Gx = -nx/nz,  Gy = -ny/nz
    → poisson_dct_neumann(gx, gy) → depth_map
    → depth_map -= depth_map_zero  (first 50 frames average)
"""

import json
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from gelsight_tactile.poisson_solver import poisson_dct_neumann


# ── Model (must match train_mlp.py) ──────────────────────────────────────────

class GelSightMLP(nn.Module):
    """5 → 32 → 32 → 32 → 2  with tanh hidden activations."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Reconstruction class ──────────────────────────────────────────────────────

class Reconstruction3D:
    """
    Load a trained GelSightMLP and compute depth maps from BGR frames.

    Usage:
        rec = Reconstruction3D(image_width=640, image_height=480)
        rec.load_model('gelsight_mlp.pth', 'norm_stats.json')
        depth, gx, gy = rec.get_depthmap(bgr_frame)
    """

    ZERO_FRAMES = 50   # frames used to compute depth baseline

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        use_gpu: bool = False,
    ) -> None:
        self.W = image_width
        self.H = image_height
        self.device = (
            torch.device('cuda')
            if use_gpu and torch.cuda.is_available()
            else torch.device('cpu')
        )

        self.net: Optional[GelSightMLP] = None
        self._input_stats: Optional[dict] = None

        # depth baseline
        self._depth_zero = np.zeros((self.H, self.W), dtype=np.float64)
        self._zero_counter = 0

        # pre-build pixel coordinate feature columns (constant across frames)
        Y_grid, X_grid = np.mgrid[0:self.H, 0:self.W]
        self._x_norm = (X_grid.ravel() / self.W).astype(np.float32)   # X_px / 640
        self._y_norm = (Y_grid.ravel() / self.H).astype(np.float32)   # Y_px / 480

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, model_path: str, norm_stats_path: str) -> None:
        """Load .pth weights and norm_stats.json."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')
        if not os.path.isfile(norm_stats_path):
            raise FileNotFoundError(f'Norm stats not found: {norm_stats_path}')

        with open(norm_stats_path) as f:
            stats = json.load(f)
        self._input_stats = stats['input']

        net = GelSightMLP().float().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            net.load_state_dict(ckpt['state_dict'])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        self.net = net

    # ── Main inference ────────────────────────────────────────────────────────

    def get_depthmap(
        self,
        bgr_frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute depth map from a single BGR frame.

        Args:
            bgr_frame: (H, W, 3) uint8 BGR image from OpenCV.

        Returns:
            depth_map : (H, W) float32, baseline-subtracted depth.
            gx_map    : (H, W) float32, x-gradient.
            gy_map    : (H, W) float32, y-gradient.
        """
        if self.net is None:
            raise RuntimeError('Call load_model() before get_depthmap().')

        N = self.H * self.W

        # BGR → RGB, flatten, normalize
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).reshape(N, 3).astype(np.float32)
        rgb_mean = np.array(self._input_stats['rgb_mean'], dtype=np.float32)
        rgb_std  = np.array(self._input_stats['rgb_std'],  dtype=np.float32)
        rgb_norm = (rgb - rgb_mean) / rgb_std

        features = np.column_stack([
            rgb_norm,
            self._x_norm,
            self._y_norm,
        ])  # (N, 5): [R_norm, G_norm, B_norm, X/W, Y/H]

        tensor = torch.from_numpy(features).to(self.device)

        with torch.no_grad():
            out = self.net(tensor).cpu().numpy()  # (N, 2)

        # model outputs [nx, ny]; recover nz and convert to gradients
        nx = out[:, 0]
        ny = out[:, 1]
        nz2 = np.clip(1.0 - nx ** 2 - ny ** 2, 1e-6, None)   # numerical safety
        nz = np.sqrt(nz2)
        gx_flat = np.clip(-nx / nz, -3.0, 3.0).astype(np.float32)
        gy_flat = np.clip(-ny / nz, -3.0, 3.0).astype(np.float32)

        gx_map = gx_flat.reshape(self.H, self.W)
        gy_map = gy_flat.reshape(self.H, self.W)

        # Poisson integration
        depth_map = poisson_dct_neumann(gx=gx_map, gy=gy_map).astype(np.float32)

        # accumulate baseline for first ZERO_FRAMES frames
        if self._zero_counter < self.ZERO_FRAMES:
            self._depth_zero += depth_map
            if self._zero_counter == 0:
                print('[Reconstruction3D] Zeroing depth. Do not touch the sensor...')
            if self._zero_counter == self.ZERO_FRAMES - 1:
                self._depth_zero /= self.ZERO_FRAMES
                print('[Reconstruction3D] Sensor ready.')
        self._zero_counter += 1

        depth_map -= self._depth_zero.astype(np.float32)


        return depth_map, gx_map, gy_map