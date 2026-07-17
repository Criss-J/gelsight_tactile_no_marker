"""
visualizer3d.py  –  Open3D 기반 실시간 3D 포인트클라우드 시각화

gsrobotics(gelsightinc/gsrobotics) utilities/visualization.py 기반.
ROS 의존성 없음. numpy 배열만 받아 동작.

Usage:
    from gelsight_tactile.visualizer3d import Visualize3D

    vis = Visualize3D(width=640, height=480)
    vis.update(depth_map, gradient_x, gradient_y)   # 매 프레임 호출
    vis.destroy()
"""

import numpy as np
import open3d


class Visualize3D:
    """Open3D 포인트클라우드로 depth_map을 실시간 시각화."""

    def __init__(
        self,
        width: int,
        height: int,
        window_width: int = 640,
        window_height: int = 480,
    ) -> None:
        self._width = width
        self._height = height

        # XY 그리드 사전 할당 (불변)
        x_range = np.arange(width)
        y_range = np.arange(height)
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        n_points = width * height
        self._points = np.zeros((n_points, 3), dtype=np.float64)
        self._points[:, 0] = grid_x.flatten()
        self._points[:, 1] = grid_y.flatten()

        # Open3D 초기화
        self._pcd = open3d.geometry.PointCloud()
        self._pcd.points = open3d.utility.Vector3dVector(self._points)

        self._vis = open3d.visualization.Visualizer()
        self._vis.create_window(
            window_name='GelSight 3D',
            width=window_width,
            height=window_height,
        )
        self._vis.add_geometry(self._pcd)

        render_opt = self._vis.get_render_option()
        render_opt.background_color = np.array([0.05, 0.05, 0.05])
        render_opt.point_size = 1.0

        view = self._vis.get_view_control()
        view.set_front([0, 0, -1])
        view.set_up([0, -1, 0])
        view.set_lookat([width / 2, height / 2, 0])

    def update(
        self,
        depth_map: np.ndarray,
        gradient_x: np.ndarray | None = None,
        gradient_y: np.ndarray | None = None,
    ) -> None:
        """
        포인트클라우드를 새 depth_map으로 갱신.

        Args:
            depth_map:  (H, W) float, raw Poisson solver 출력 (단위 변환 없음).
            gradient_x: (H, W) float, x 방향 기울기. None이면 depth_map에서 자동 계산.
            gradient_y: (H, W) float, y 방향 기울기. None이면 depth_map에서 자동 계산.
        """
        # z 좌표 업데이트
        self._points[:, 2] = depth_map.flatten()

        # 기울기 기반 RGB 컬러
        if gradient_x is None or gradient_y is None:
            gradient_x, gradient_y = np.gradient(depth_map)

        cx = np.clip(0.5 * gradient_x + 0.5, 0.0, 1.0).flatten()
        cy = np.clip(0.5 * gradient_y + 0.5, 0.0, 1.0).flatten()

        colors = np.zeros((self._points.shape[0], 3), dtype=np.float64)
        colors[:, 0] = cx
        colors[:, 1] = cy
        colors[:, 2] = (cx + cy) / 2.0

        self._pcd.points = open3d.utility.Vector3dVector(self._points)
        self._pcd.colors = open3d.utility.Vector3dVector(colors)

        self._vis.update_geometry(self._pcd)
        self._vis.poll_events()
        self._vis.update_renderer()

    def destroy(self) -> None:
        """Open3D 창 닫기."""
        self._vis.destroy_window()