import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            camera_qos
        )

        self.get_logger().info('CameraViewer 시작 — q 키로 종료')

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # RGB 채널 평균 계산 (BGR → RGB)
        B_mean = frame[:, :, 0].mean()
        G_mean = frame[:, :, 1].mean()
        R_mean = frame[:, :, 2].mean()

        # 화면에 표시
        disp = frame.copy()
        h, w = disp.shape[:2]
        text = f'R:{R_mean:5.1f}  G:{G_mean:5.1f}  B:{B_mean:5.1f}'
        cv2.putText(disp, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 각 채널을 개별 창으로 (흑백)
        cv2.imshow('R channel', frame[:, :, 2])  # BGR의 R
        cv2.imshow('G channel', frame[:, :, 1])
        cv2.imshow('B channel', frame[:, :, 0])
        cv2.imshow('GelSight Camera', disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()