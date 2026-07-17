"""
reconstructor_node.py  –  ROS2 node for GelSight real-time 3D reconstruction

Subscribes : /camera/image_raw/compressed  (CompressedImage)
Publishes  : /gelsight/depth_map           (Image, 32FC1)

Usage:
    ros2 run gelsight_tactile reconstructor_node \\
      --ros-args \\
        -p model_path:=~/gelsight_ws/data/nx_ny/SESSION_TS/train_TRAIN_TS/gelsight_mlp_TRAIN_TS.pth \\
        -p norm_stats_path:=~/gelsight_ws/data/nx_ny/SESSION_TS/train_TRAIN_TS/norm_stats_TRAIN_TS.json \\
        -p px_per_mm:=<calib_SESSION_TS.json의 px_per_mm 값>
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

from gelsight_tactile.reconstructor import Reconstruction3D
from gelsight_tactile.visualizer3d import Visualize3D


class ReconstructorNode(Node):

    def __init__(self):
        super().__init__('reconstructor_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',      '')
        self.declare_parameter('norm_stats_path', '')
        self.declare_parameter('image_width',     640)
        self.declare_parameter('image_height',    480)
        self.declare_parameter('use_gpu',         False)
        self.declare_parameter('px_per_mm',       0.0)

        model_path      = self.get_parameter('model_path').value
        norm_stats_path = self.get_parameter('norm_stats_path').value
        image_width     = self.get_parameter('image_width').value
        image_height    = self.get_parameter('image_height').value
        use_gpu         = self.get_parameter('use_gpu').value
        px_per_mm       = self.get_parameter('px_per_mm').value

        if not model_path or not norm_stats_path:
            self.get_logger().error(
                'model_path and norm_stats_path parameters are required.\n'
                'Example:\n'
                '  ros2 run gelsight_tactile reconstructor_node \\\n'
                '    --ros-args \\\n'
                '      -p model_path:=/path/to/gelsight_mlp.pth \\\n'
                '      -p norm_stats_path:=/path/to/norm_stats.json \\\n'
                '      -p px_per_mm:=<calib_TIMESTAMP.json의 px_per_mm 값>'
            )
            raise SystemExit(1)

        if px_per_mm <= 0.0:
            self.get_logger().error(
                'px_per_mm 파라미터가 필요합니다.\n'
                'calib_TIMESTAMP.json 파일에서 px_per_mm 값을 확인하세요.\n'
                '  -p px_per_mm:=<값>'
            )
            raise SystemExit(1)

        self._px_per_mm = px_per_mm
        self.get_logger().info(f'px_per_mm = {px_per_mm:.2f}')

        # ── Reconstruction engine ──────────────────────────────────────────────
        self.rec = Reconstruction3D(
            image_width=image_width,
            image_height=image_height,
            use_gpu=use_gpu,
        )
        self.rec.load_model(model_path, norm_stats_path)
        self.get_logger().info(f'Model loaded: {model_path}')

        # ── ROS ───────────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self._callback,
            qos,
        )
        self.pub = self.create_publisher(Image, '/gelsight/depth_map', 10)

        cv2.namedWindow('GelSight Reconstruction', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('GelSight Reconstruction', 1280, 480)

        # ── 3D point cloud ─────────────────────────────────────────────────────
        self._vis3d = Visualize3D(
            width=image_width,
            height=image_height,
            window_width=image_width,
            window_height=image_height,
        )

        self.get_logger().info('ReconstructorNode started.')

    # ── Callback ──────────────────────────────────────────────────────────────

    def _callback(self, msg: CompressedImage) -> None:
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            depth_map, gx, gy = self.rec.get_depthmap(bgr)

            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')
            depth_msg.header = msg.header
            self.pub.publish(depth_msg)

            self._show(bgr, depth_map)
            self._vis3d.update(depth_map, gx, gy)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Reconstruction error: {e}')

    _DEPTH_RANGE_MM = 1.5   # 시각화 클램핑 범위 (mm)

    def _show(self, bgr: np.ndarray, depth_map: np.ndarray) -> None:
        # 클램핑 범위를 mm → 픽셀 단위로 변환
        depth_min_px = -self._DEPTH_RANGE_MM * self._px_per_mm
        d_clamp = np.clip(depth_map, depth_min_px, 0.0)
        d_norm = ((d_clamp - depth_min_px) / (0.0 - depth_min_px) * 255).astype(np.uint8)
        depth_gray = cv2.cvtColor(d_norm, cv2.COLOR_GRAY2BGR)

        # 픽셀 단위 깊이를 px_per_mm으로 나눠 실제 mm로 변환
        # max_depth_mm = float(np.min(depth_map)) / self._px_per_mm
        # label = f'max depth: {abs(max_depth_mm):.2f} mm'
        # cv2.putText(depth_gray, label, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 원본 | 깊이맵 나란히
        combined = np.hstack([bgr, depth_gray])
        cv2.imshow('GelSight Reconstruction', combined)


def main(args=None):
    rclpy.init(args=args)
    node = ReconstructorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node._vis3d.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()