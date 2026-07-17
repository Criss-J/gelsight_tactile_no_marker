"""
GelSight 촉각센서 학습 데이터 수집 노드 (Non-blocking 개선판)
------------------------------------------------------
[개선 사항]
1. 콜백 블로킹 방지: rclpy.spin이 멈추지 않도록 상태 머신(State Machine) 기반 비동기 설계
2. 타이머 기반 UI 루프: cv2.imshow 및 cv2.waitKey를 30Hz 타이머 콜백으로 분리
3. 안정성 향상: CLI 입력(input()) 시 rclpy 스레드가 무한 대기하는 현상 차단

키 조작:
  [b] 배경 이미지 캡처
  [p] px/mm 캘리브레이션 모드 진입 (두 점 클릭 -> Enter 확정 -> 터미널에 mm 입력)
  [s] 원 수동 조정 및 샘플 저장 모드 진입 (WASD로 중심 이동, M/N으로 반지름 조절 -> Enter 확정)
  [h] 도움말 출력
  [q] 데이터 저장 및 종료
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge

import cv2
import numpy as np
import os
import json
from datetime import datetime
import sys


class BallCalibrator(Node):
    def __init__(self):
        super().__init__('ball_calibrator')

        # ── ROS 파라미터 ──────────────────────────────────────────────────────
        self.declare_parameter('ball_radius_mm', 1.5)
        self.declare_parameter('bg_exclusion_multiplier', 2.8)
        _pkg_data = os.path.join(os.path.expanduser('~'), 'yang', 'data', 'nx_ny')
        self.declare_parameter('output_dir', _pkg_data)

        self.ball_radius_mm = self.get_parameter('ball_radius_mm').value
        self.bg_exclusion_multiplier = self.get_parameter('bg_exclusion_multiplier').value
        self.output_dir = self.get_parameter('output_dir').value
        os.makedirs(self.output_dir, exist_ok=True)

        # ── 상태 관리 (State Machine) ─────────────────────────────────────────
        # STATES: 'NORMAL', 'CALIPER_CLICK', 'CIRCLE_ADJUST'
        self.state = 'NORMAL'  

        # ── 이미지 및 데이터 저장소 ──────────────────────────────────────────
        self.raw_frame = None       # 최근 수신한 원본 프레임
        self.bg_frame = None        # 배경 프레임
        self.px_per_mm = None       # 픽셀/mm 비율
        self.collected_data = []    # 수집 데이터 목록
        self.frame_count = 0        # 저장된 프레임 수
        
        # ── Hough / 원 세팅 ──────────────────────────────────────────────────
        self.current_circle = None  # 자동 검출된 원 (cx, cy, r)
        self.adjust_circle = None   # 수동 조정 중인 원 [cx, cy, r]
        
        self.hough_param1 = 50
        self.hough_param2 = 30
        self.hough_min_r = 10
        self.hough_max_r = 200

        # ── 캘리퍼 캘리브레이션용 변수 ────────────────────────────────────────
        self.caliper_points = []

        # ── ROS 구독 및 타이머 ────────────────────────────────────────────────
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            qos)
        self.bridge = CvBridge()

        # UI 갱신 및 키 입력을 위한 비동기 타이머 (30Hz)
        self.ui_timer = self.create_timer(1.0 / 30.0, self.ui_loop_callback)

        # ── OpenCV 창 및 트랙바 설정 ──────────────────────────────────────────
        self.win_name = 'GelSight Calibrator'
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 800, 720)
        cv2.setMouseCallback(self.win_name, self._mouse_callback)
        self._create_trackbars()

        self.get_logger().info(f'BallCalibrator 시작됨 | 구슬 반지름={self.ball_radius_mm}mm')
        self.get_logger().info(f'저장 디렉토리: {self.output_dir}')
        self._print_help()

    def _create_trackbars(self):
        cv2.createTrackbar('Param1 (edge)', self.win_name, self.hough_param1, 300,
                           lambda v: setattr(self, 'hough_param1', max(v, 1)))
        cv2.createTrackbar('Param2 (accum)', self.win_name, self.hough_param2, 150,
                           lambda v: setattr(self, 'hough_param2', max(v, 1)))
        cv2.createTrackbar('Min Radius', self.win_name, self.hough_min_r, 300,
                           lambda v: setattr(self, 'hough_min_r', max(v, 1)))
        cv2.createTrackbar('Max Radius', self.win_name, self.hough_max_r, 400,
                           lambda v: setattr(self, 'hough_max_r', max(v, 1)))

    # ── 1. 이미지 수신 콜백 (Non-blocking) ──────────────────────────────────
    def listener_callback(self, msg):
        try:
            # 원본 프레임 업데이트만 수행하고 바로 리턴 (블로킹 제거)
            self.raw_frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            
            # 일반 모드일 때만 실시간 자동 원 탐지 구동
            if self.state == 'NORMAL':
                self.current_circle = self._detect_circle(self.raw_frame)
        except Exception as e:
            self.get_logger().error(f'이미지 수신 중 오류: {e}')

    # ── 2. UI 루프 및 상태 처리 타이머 콜백 (30Hz) ───────────────────────────
    def ui_loop_callback(self):
        if self.raw_frame is None:
            return

        display_frame = self.raw_frame.copy()
        h, w = display_frame.shape[:2]

        # 현재 상태 머신 분기에 맞춰 렌더링
        if self.state == 'NORMAL':
            display = self._make_normal_display(display_frame, h)
        elif self.state == 'CALIPER_CLICK':
            display = self._make_caliper_display(display_frame, h)
        elif self.state == 'CIRCLE_ADJUST':
            display = self._make_adjust_display(display_frame, h)
        else:
            display = display_frame

        cv2.imshow(self.win_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key != 0xFF:
            self._handle_key_event(key)

    # ── 3. 원 검출 로직 ──────────────────────────────────────────────────────
    def _detect_circle(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_min_r,
            maxRadius=self.hough_max_r,
        )
        if circles is not None:
            return np.round(circles[0, 0]).astype(int)
        return None

    # ── 4. 화면 디스플레이 생성 ────────────────────────────────────────────────
    def _make_normal_display(self, frame, h):
        display = cv2.copyMakeBorder(frame, 0, 170, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        if self.current_circle is not None:
            cx, cy, r = self.current_circle
            cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(display, f'Circle: ({cx},{cy}) r={r}px',
                        (10, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(display, 'Circle: Not detected',
                        (10, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 1)

        self._draw_status_text(display, h)
        return display

    def _make_caliper_display(self, frame, h):
        display = cv2.copyMakeBorder(frame, 0, 170, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        for pt in self.caliper_points:
            cv2.circle(display, pt, 6, (0, 255, 0), -1)

        if len(self.caliper_points) == 2:
            cv2.line(display, self.caliper_points[0], self.caliper_points[1], (0, 255, 0), 2)
            dist_px = np.hypot(self.caliper_points[1][0] - self.caliper_points[0][0],
                               self.caliper_points[1][1] - self.caliper_points[0][1])
            cv2.putText(display, f'dist={dist_px:.1f}px  Enter:Confirm  ESC:Reset',
                        (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(display, f'pt1={self.caliper_points[0]}  pt2={self.caliper_points[1]}',
                        (10, h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(display, f'Click point {len(self.caliper_points) + 1}/2 on image  ESC:Cancel',
                        (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
        return display

    def _make_adjust_display(self, frame, h):
        display = cv2.copyMakeBorder(frame, 0, 170, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        if self.adjust_circle is not None:
            cx, cy, r = self.adjust_circle
            cv2.circle(display, (cx, cy), r, (0, 255, 255), 2)  # 황색 조정원
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(display, '[ADJUST] WASD:move | MN:radius | Enter:Confirm | ESC:Cancel',
                        (10, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, f'center=({cx},{cy})  r={r}px',
                        (10, h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return display

    def _draw_status_text(self, display, h):
        bg_str = 'OK' if self.bg_frame is not None else 'None (press b)'
        px_str = f'{self.px_per_mm:.4f} px/mm' if self.px_per_mm else 'None (press p)'
        n_px = sum(len(d) for d in self.collected_data)
        data_str = f'{n_px} px / {self.frame_count} frames'

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, f'BG: {bg_str}',     (10, h + 60),  font, 0.55, (200, 200, 200), 1)
        cv2.putText(display, f'px/mm: {px_str}',  (10, h + 88),  font, 0.55, (200, 200, 200), 1)
        cv2.putText(display, f'Data: {data_str}', (10, h + 116), font, 0.55, (200, 200, 200), 1)
        cv2.putText(display, 'b:BG  p:px/mm  s:save sample  h:help  q:quit+save',
                    (10, h + 150), font, 0.45, (130, 130, 130), 1)

    # ── 5. 마우스 및 키 입력 핸들러 ───────────────────────────────────────────
    def _mouse_callback(self, event, x, y, flags, param):
        if self.state == 'CALIPER_CLICK' and event == cv2.EVENT_LBUTTONDOWN:
            if len(self.caliper_points) < 2:
                self.caliper_points.append((x, y))

    def _handle_key_event(self, key):
        # 1) 일반 상태 (NORMAL)
        if self.state == 'NORMAL':
            if key == ord('b'):
                if self.raw_frame is not None:
                    self.bg_frame = self.raw_frame.copy()
                    self.get_logger().info('배경 이미지 캡처 완료 📸')
            elif key == ord('p'):
                self.caliper_points.clear()
                self.state = 'CALIPER_CLICK'
                self.get_logger().info('캘리퍼 캘리브레이션 모드 진입')
            elif key == ord('s'):
                if self.current_circle is None:
                    self.get_logger().warn('Hough Circle이 원을 감지하지 못했습니다. 수동 타겟팅을 시작합니다.')
                    # 검출되지 않은 경우 화면 중심부에 임의의 시작원 배치
                    h, w = self.raw_frame.shape[:2]
                    self.adjust_circle = [w // 2, h // 2, 50]
                else:
                    self.adjust_circle = list(self.current_circle)
                self.state = 'CIRCLE_ADJUST'
                self.get_logger().info('원 미세 조정 모드 진입 (WASD/MN)')
            elif key == ord('h'):
                self._print_help()
            elif key == ord('q'):
                self._finish_and_save()
                sys.exit(0)

        # 2) 캘리퍼 포인트 선택 상태 (CALIPER_CLICK)
        elif self.state == 'CALIPER_CLICK':
            if key == 27:  # ESC
                if len(self.caliper_points) == 2:
                    self.caliper_points.clear()
                else:
                    self.state = 'NORMAL'
                    self.get_logger().info('캘리퍼 캘리브레이션 취소')
            elif key == 13 and len(self.caliper_points) == 2:  # Enter
                self._process_caliper_calculation()

        # 3) 원 조정 및 샘플링 상태 (CIRCLE_ADJUST)
        elif self.state == 'CIRCLE_ADJUST':
            if self.adjust_circle is not None:
                if key == ord('w'):   self.adjust_circle[1] -= 1
                elif key == ord('s'): self.adjust_circle[1] += 1
                elif key == ord('a'): self.adjust_circle[0] -= 1
                elif key == ord('d'): self.adjust_circle[0] += 1
                elif key == ord('m'): self.adjust_circle[2] += 1
                elif key == ord('n'): self.adjust_circle[2] = max(1, self.adjust_circle[2] - 1)
                elif key == 27:  # ESC
                    self.state = 'NORMAL'
                    self.adjust_circle = None
                    self.get_logger().info('샘플링 취소')
                elif key == 13:  # Enter
                    self._save_training_sample(self.raw_frame, self.adjust_circle)
                    self.adjust_circle = None
                    self.state = 'NORMAL'

    # ── 6. 데이터 계산 및 가공 로직 ──────────────────────────────────────────
    def _process_caliper_calculation(self):
        dist_px = np.hypot(self.caliper_points[1][0] - self.caliper_points[0][0],
                           self.caliper_points[1][1] - self.caliper_points[0][1])
        print(f'\n[입력 필요] 선택된 픽셀 거리: {dist_px:.2f} px')
        
        # input() 블로킹을 방지하기 위해 노드가 살아있는 상태에서 대기
        try:
            actual_mm = float(input('버니어 캘리퍼로 측정한 실제 거리(mm)를 입력하세요: '))
            if actual_mm <= 0:
                raise ValueError("mm 값은 0보다 커야 합니다.")
            self.px_per_mm = dist_px / actual_mm
            self.get_logger().info(f'px/mm 캘리브레이션 완료: {self.px_per_mm:.4f} px/mm')
        except ValueError as e:
            self.get_logger().error(f'잘못된 값 입력됨: {e}')
        
        self.state = 'NORMAL'

    def _save_training_sample(self, frame, circle):
        if circle is None:
            self.get_logger().warn('원이 검출되지 않았습니다.')
            return
        if self.px_per_mm is None:
            self.get_logger().warn('먼저 px/mm 캘리브레이션을 하세요 (p 키).')
            return

        cx, cy, r_px = circle
        R = self.ball_radius_mm
        ppm = self.px_per_mm
        h, w = frame.shape[:2]

        Y_grid, X_grid = np.mgrid[0:h, 0:w]

        dist_sq_px = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2
        contact_mask = dist_sq_px <= r_px ** 2

        x_mm = (X_grid - cx) / ppm
        y_mm = (Y_grid - cy) / ppm
        r_mm_sq = x_mm ** 2 + y_mm ** 2

        valid_mask = contact_mask & (r_mm_sq < R ** 2 - 1e-6)

        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            self.get_logger().warn(
                '유효 접촉 픽셀 없음. 구슬 반지름(ball_radius_mm)이나 '
                'Hough 파라미터를 확인하세요.')
            return

        denom_sphere = np.sqrt(np.maximum(R ** 2 - r_mm_sq[ys, xs], 1e-9))
        Gx_vals = x_mm[ys, xs] / denom_sphere
        Gy_vals = y_mm[ys, xs] / denom_sphere
        denom_normal = np.sqrt(Gx_vals ** 2 + Gy_vals ** 2 + 1.0)
        nx_vals = (-Gx_vals / denom_normal).astype(np.float32)
        ny_vals = (-Gy_vals / denom_normal).astype(np.float32)

        # raw RGB 그대로 (차분 없음)
        rgb = frame[ys, xs][:, ::-1].astype(np.float32)

        contact_data = np.column_stack([
            rgb,
            xs.astype(np.float32),
            ys.astype(np.float32),
            nx_vals,
            ny_vals,
        ])

        # 비접촉 픽셀 25% 추가
        exclusion_mask = dist_sq_px <= (r_px * self.bg_exclusion_multiplier) ** 2
        nc_ys, nc_xs = np.where(~exclusion_mask)
        n_nc = max(1, int(len(xs) * 0.25))
        idx = np.random.choice(len(nc_xs), size=min(n_nc, len(nc_xs)), replace=False)
        nc_rgb = frame[nc_ys[idx], nc_xs[idx]][:, ::-1].astype(np.float32)
        nc_data = np.column_stack([
            nc_rgb,
            nc_xs[idx].astype(np.float32),
            nc_ys[idx].astype(np.float32),
            np.zeros(len(idx), dtype=np.float32),
            np.zeros(len(idx), dtype=np.float32),
        ])

        frame_data = np.vstack([contact_data, nc_data])
        self.collected_data.append(frame_data)
        self.frame_count += 1
        n_total = sum(len(d) for d in self.collected_data)

        self.get_logger().info(
            f'프레임 {self.frame_count} 저장: '
            f'접촉 {len(xs)}px + 비접촉 {len(idx)}px → 누적 {n_total}px')
        print(f'[s] 프레임 {self.frame_count}: 접촉 {len(xs)}px + 비접촉 {len(idx)}px '
            f'(누적: {n_total}px)')
    # ── 7. 종료 및 원본 저장 ─────────────────────────────────────────────────
    def _finish_and_save(self):
        if not self.collected_data:
            self.get_logger().warn('저장할 데이터가 없어 종료합니다.')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        all_data = np.vstack(self.collected_data)
        np.random.shuffle(all_data)
        split = int(len(all_data) * 0.8)
        train_data = all_data[:split]
        val_data = all_data[split:]

        np.save(os.path.join(save_dir, f'train_{timestamp}.npy'), train_data)
        np.save(os.path.join(save_dir, f'val_{timestamp}.npy'), val_data)

        if self.bg_frame is not None:
            cv2.imwrite(os.path.join(save_dir, f'background_{timestamp}.png'), self.bg_frame)

        calib = {
            'px_per_mm': float(self.px_per_mm) if self.px_per_mm else None,
            'ball_radius_mm': self.ball_radius_mm,
            'timestamp': timestamp,
            'n_frames': self.frame_count,
            'n_train_samples': len(train_data),
            'n_val_samples': len(val_data),
            'column_order': ['R', 'G', 'B', 'X_px', 'Y_px', 'nx', 'ny'],
        }
        with open(os.path.join(save_dir, f'calib_{timestamp}.json'), 'w') as f:
            json.dump(calib, f, indent=2)

        print('\n=====================================')
        print(f'🎉 데이터 저장 완료 : {save_dir}')
        print(f'├─ 훈련 데이터: {len(train_data)} px')
        print(f'└─ 검증 데이터: {len(val_data)} px')
        print('=====================================\n')

    def _print_help(self):
        print('\n=== GelSight 데이터 수집 사용 가이드 ===')
        print('  [b] 배경 이미지 캡처 (물체 접촉이 없는 기본 젤 상태)')
        print('  [p] px/mm 캘리브레이션 (양 끝단 마우스 좌클릭 후 엔터 -> 실제mm 입력)')
        print('  [s] 훈련 샘플링 모드 진입')
        print('      -> 원이 발견되면 그 위치에서 미세조정 시작')
        print('      -> 원이 안보일 시 중앙에 강제 원 생성 후 수동 타겟팅 수행')
        print('         [W/S/A/D] 중심점 미세 이동 | [M/N] 반지름 크기 조절')
        print('         [Enter] 최종 샘플링 저장 | [ESC] 샘플링 취소')
        print('  [q] 최종 데이터 분리 저장 후 종료')
        print('========================================\n')


def main(args=None):
    rclpy.init(args=args)
    node = BallCalibrator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()