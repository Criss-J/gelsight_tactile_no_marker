#!/usr/bin/env python3
"""
GelSight MLP 기울기 시각화 진단 스크립트
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
def set_korean_font():
    font_candidates = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for path in font_candidates:
        if os.path.exists(path):
            font = fm.FontProperties(fname=path)
            matplotlib.rcParams['font.family'] = font.get_name()
            matplotlib.rcParams['axes.unicode_minus'] = False
            return
    # 못 찾으면 영어로 fallback
    matplotlib.rcParams['axes.unicode_minus'] = False

set_korean_font()


# ── 모델 정의 ──────────────────────────────────────────────────────────────────

class GelSightMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── 자동 탐색 ──────────────────────────────────────────────────────────────────

def find_latest_model(base_dir: str):
    pattern = os.path.join(base_dir, "*", "train_*", "gelsight_mlp_*.pth")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {pattern}")
    model_path = files[-1]
    stats_path = model_path.replace("gelsight_mlp_", "norm_stats_").replace(".pth", ".json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"norm_stats 파일 없음: {stats_path}")
    return model_path, stats_path


def find_latest_background(base_dir: str):
    pattern = os.path.join(base_dir, "*", "background_*.png")
    files = sorted(glob.glob(pattern))
    return cv2.imread(files[-1]) if files else None


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model(model_path: str, norm_stats_path: str, device):
    model = GelSightMLP().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    with open(norm_stats_path) as f:
        stats = json.load(f)

    rgb_mean = np.array(stats["input"]["rgb_mean"], dtype=np.float32)
    rgb_std  = np.array(stats["input"]["rgb_std"],  dtype=np.float32)
    return model, rgb_mean, rgb_std


# ── 추론 ───────────────────────────────────────────────────────────────────────

def predict(frame_bgr: np.ndarray, model, rgb_mean, rgb_std, device, bg_bgr=None):
    H, W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    xs = np.tile(np.arange(W), H).astype(np.float32)
    ys = np.repeat(np.arange(H), W).astype(np.float32)

    R = frame_rgb[:, :, 0].flatten()
    G = frame_rgb[:, :, 1].flatten()
    B = frame_rgb[:, :, 2].flatten()

    R_n = (R - rgb_mean[0]) / rgb_std[0]
    G_n = (G - rgb_mean[1]) / rgb_std[1]
    B_n = (B - rgb_mean[2]) / rgb_std[2]
    X_n = xs / 640.0
    Y_n = ys / 480.0

    inp = np.stack([R_n, G_n, B_n, X_n, Y_n], axis=1).astype(np.float32)
    inp_t = torch.from_numpy(inp).to(device)

    with torch.no_grad():
        out = model(inp_t).cpu().numpy()

    nx = out[:, 0].reshape(H, W)
    ny = out[:, 1].reshape(H, W)

    nz = np.sqrt(np.clip(1.0 - nx**2 - ny**2, 1e-8, 1.0))
    Gx = -nx / nz
    Gy = -ny / nz

    return nx, ny, Gx, Gy


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def visualize_once(frame_bgr, nx, ny, Gx, Gy):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    slope_deg = np.degrees(np.arctan(np.sqrt(Gx**2 + Gy**2)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("GelSight MLP Gradient Debugger", fontsize=14)

    panels = [
        (axes[0, 0], frame_rgb,  "Original image",   None,      None),
        (axes[0, 1], Gx,         "Gx (dZ/dx)",       "RdBu_r",  (-0.5, 0.5)),
        (axes[0, 2], Gy,         "Gy (dZ/dy)",       "RdBu_r",  (-0.5, 0.5)),
        (axes[1, 0], slope_deg,  "slope arctan(|G|) [deg]", "hot", (0, 90)),
        (axes[1, 1], nx,         "nx (normal x)",    "RdBu_r",  (-0.5, 0.5)),
        (axes[1, 2], ny,         "ny (normal y)",    "RdBu_r",  (-0.5, 0.5)),
    ]

    for ax, img, title, cmap, vlim in panels:
        if cmap is None:
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # min / max / mean 통계 타이틀에 표시
            title_with_stats = (
                f"{title}\n"
                f"min={img.min():.3f}  max={img.max():.3f}  mean={img.mean():.3f}"
            )
            ax.set_title(title_with_stats, fontsize=8)
        ax.axis("off")

    # slope 통계도 타이틀에 추가
    axes[1, 0].set_title(
        f"slope arctan(|G|) [deg]\n"
        f"max={slope_deg.max():.1f}°  99p={np.percentile(slope_deg,99):.1f}°  mean={slope_deg.mean():.1f}°",
        fontsize=8
    )

    print(f"\n[통계]")
    print(f"  nx  범위: [{nx.min():.3f}, {nx.max():.3f}]  mean={nx.mean():.4f}")
    print(f"  ny  범위: [{ny.min():.3f}, {ny.max():.3f}]  mean={ny.mean():.4f}")
    print(f"  Gx  범위: [{Gx.min():.3f}, {Gx.max():.3f}]")
    print(f"  Gy  범위: [{Gy.min():.3f}, {Gy.max():.3f}]")
    print(f"  경사각 최대: {slope_deg.max():.1f}°  평균: {slope_deg.mean():.1f}°")

    plt.tight_layout()
    plt.show()


# ── ROS2 라이브 모드 ───────────────────────────────────────────────────────────

def live_mode(model, rgb_mean, rgb_std, device, bg_bgr=None):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage
    except ImportError:
        print("[오류] ROS2 환경이 아닙니다. --image 옵션으로 이미지 파일을 지정하세요.")
        sys.exit(1)

    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("GelSight MLP Live Debugger (Ctrl+C: quit)", fontsize=14)
    initialized = [False]
    ims = [None] * 6

    panel_cfg = [
        (axes[0, 0], "Original image",         None,      None),
        (axes[0, 1], "Gx (dZ/dx)",             "RdBu_r",  (-0.5, 0.5)),
        (axes[0, 2], "Gy (dZ/dy)",             "RdBu_r",  (-0.5, 0.5)),
        (axes[1, 0], "slope arctan(|G|) [deg]","hot",     (0, 90)),
        (axes[1, 1], "nx (normal x)",          "RdBu_r",  (-0.5, 0.5)),
        (axes[1, 2], "ny (normal y)",          "RdBu_r",  (-0.5, 0.5)),
    ]
    for ax, title, _, _ in panel_cfg:
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    class DebugNode(Node):
        def __init__(self):
            super().__init__("debug_gradients")
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            qos = QoSProfile(depth=1)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.sub = self.create_subscription(
                CompressedImage, "/camera/image_raw/compressed",
                self.cb, qos)

        def cb(self, msg):
            arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            nx, ny, Gx, Gy = predict(frame, model, rgb_mean, rgb_std, device, bg_bgr=bg_bgr)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            slope_deg = np.degrees(np.arctan(np.sqrt(Gx**2 + Gy**2)))

            imgs = [frame_rgb, Gx, Gy, slope_deg, nx, ny]

            if not initialized[0]:
                for i, ((ax, title, cmap, vlim), img) in enumerate(zip(panel_cfg, imgs)):
                    if cmap is None:
                        ims[i] = ax.imshow(img)
                    else:
                        ims[i] = ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
                        plt.colorbar(ims[i], ax=ax, fraction=0.046, pad=0.04)
                initialized[0] = True
            else:
                for i, ((_ax, title, cmap, vlim), img) in enumerate(zip(panel_cfg, imgs)):
                    ims[i].set_data(img)
                    if cmap is not None and img.ndim == 2:
                        _ax.set_title(
                            f"{title}\nmin={img.min():.3f}  max={img.max():.3f}  mean={img.mean():.3f}",
                            fontsize=8
                        )

            fig.canvas.draw_idle()
            plt.pause(0.001)

    rclpy.init()
    node = DebugNode()
    print("[라이브 모드] /camera/image_raw/compressed 구독 중... (Ctrl+C 종료)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GelSight MLP 기울기 시각화 진단")
    parser.add_argument("--model",      default=None, help=".pth 모델 경로 (미지정 시 자동 탐색)")
    parser.add_argument("--norm-stats", default=None, help="norm_stats.json 경로")
    parser.add_argument("--image",      default=None, help="분석할 이미지 파일 경로")
    parser.add_argument("--background", default=None, help="배경 이미지 경로 (미지정 시 자동 탐색)")
    parser.add_argument("--live",       action="store_true", help="ROS2 라이브 카메라 모드")
    args = parser.parse_args()

    base_dir = os.path.expanduser("~/yang/data/nx_ny")

    if args.model and args.norm_stats:
        model_path = os.path.expanduser(args.model)
        stats_path = os.path.expanduser(args.norm_stats)
    else:
        print(f"[자동 탐색] {base_dir}")
        model_path, stats_path = find_latest_model(base_dir)

    print(f"[모델] {model_path}")
    print(f"[stats] {stats_path}")

    if args.background:
        bg = cv2.imread(os.path.expanduser(args.background))
    else:
        bg = find_latest_background(base_dir)

    if bg is not None:
        print(f"[배경] 배경 차분 적용됨")
    else:
        print(f"[경고] background 이미지 없음 — 차분 없이 추론")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, rgb_mean, rgb_std = load_model(model_path, stats_path, device)
    print(f"[디바이스] {device}")

    if args.live:
        live_mode(model, rgb_mean, rgb_std, device, bg_bgr=bg)
    elif args.image:
        img_path = os.path.expanduser(args.image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[오류] 이미지 로드 실패: {img_path}")
            sys.exit(1)
        print(f"[이미지] {img_path}  shape={frame.shape}")
        nx, ny, Gx, Gy = predict(frame, model, rgb_mean, rgb_std, device, bg_bgr=bg)
        visualize_once(frame, nx, ny, Gx, Gy)
    else:
        bg_pattern = os.path.join(base_dir, "*", "background_*.png")
        bgs = sorted(glob.glob(bg_pattern))
        if not bgs:
            print("[오류] --image 또는 --live 옵션을 지정하세요.")
            parser.print_help()
            sys.exit(1)
        bg_path = bgs[-1]
        print(f"[자동] background 이미지로 테스트: {bg_path}")
        frame = cv2.imread(bg_path)
        nx, ny, Gx, Gy = predict(frame, model, rgb_mean, rgb_std, device, bg_bgr=bg)
        visualize_once(frame, nx, ny, Gx, Gy)


if __name__ == "__main__":
    main()