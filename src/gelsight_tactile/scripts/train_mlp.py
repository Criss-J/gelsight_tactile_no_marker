#!/usr/bin/env python3
"""
GelSight MLP 학습 스크립트
입력: [R, G, B, X_px, Y_px] → 출력: [nx, ny]

사용법:
  # 가장 최근 세션 자동 선택
  python3 train_mlp.py --epochs 100

  # 특정 세션 지정
  python3 train_mlp.py \
    --train-file ~/yang/data/nx_ny/20260716_123329/train_20260716_123329.npy \
    --val-file   ~/yang/data/nx_ny/20260716_123329/val_20260716_123329.npy \
    --epochs 100
"""

import argparse
import json
import os
import glob
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ── 모델 정의 ─────────────────────────────────────────────────────────────────

class GelSightMLP(nn.Module):
    """논문 스펙: Linear(5,32)-Tanh x3 - Linear(32,2)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2),   # 출력: [nx, ny], 활성화 없음
        )

    def forward(self, x):
        return self.net(x)


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def find_latest_session(base_dir: str):
    """~/yang/data/nx_ny/ 에서 가장 최근 세션 자동 탐색"""
    pattern = os.path.join(base_dir, "*", "train_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"train_*.npy 파일을 찾을 수 없음: {pattern}")
    train_file = files[-1]
    val_file = train_file.replace("train_", "val_")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"val 파일 없음: {val_file}")
    return train_file, val_file


def load_data(train_file: str, val_file: str):
    """
    .npy 파일 로드. 컬럼 순서: [R, G, B, X_px, Y_px, nx, ny]
    """
    train = np.load(train_file).astype(np.float32)
    val   = np.load(val_file).astype(np.float32)
    print(f"  train: {train.shape}, val: {val.shape}")

    X_train, y_train = train[:, :5], train[:, 5:]
    X_val,   y_val   = val[:, :5],   val[:, 5:]
    return X_train, y_train, X_val, y_val


# ── 정규화 ─────────────────────────────────────────────────────────────────────

def normalize_inputs(X_train, X_val):
    """
    R,G,B: 채널별 표준화 (train 기준)
    X_px: /640, Y_px: /480
    """
    rgb_mean = X_train[:, :3].mean(axis=0)
    rgb_std  = X_train[:, :3].std(axis=0) + 1e-8

    def normalize(X):
        X = X.copy()
        X[:, :3] = (X[:, :3] - rgb_mean) / rgb_std
        X[:, 3] /= 640.0
        X[:, 4] /= 480.0
        return X

    return normalize(X_train), normalize(X_val), rgb_mean, rgb_std


# ── 학습 ───────────────────────────────────────────────────────────────────────

def train(args):
    # 1. 데이터 로드
    base_dir = os.path.expanduser("~/yang/data/nx_ny")
    if args.train_file and args.val_file:
        train_file = os.path.expanduser(args.train_file)
        val_file   = os.path.expanduser(args.val_file)
    else:
        print(f"[자동 탐색] {base_dir}")
        train_file, val_file = find_latest_session(base_dir)

    print(f"[데이터]")
    print(f"  train: {train_file}")
    print(f"  val  : {val_file}")
    X_train, y_train, X_val, y_val = load_data(train_file, val_file)

    # 2. 정규화
    X_train_n, X_val_n, rgb_mean, rgb_std = normalize_inputs(X_train, X_val)

    # 3. DataLoader
    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n),
        torch.from_numpy(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 4. 모델 / 옵티마이저
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}")

    model = GelSightMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 5. 학습 루프
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_ds)

        # Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                v_loss += criterion(pred, yb).item() * len(xb)
        v_loss /= len(val_ds)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train={t_loss:.6f}  val={v_loss:.6f}")

    # 6. 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.dirname(train_file)
    out_dir = os.path.join(session_dir, f"train_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # 모델
    model_path = os.path.join(out_dir, f"gelsight_mlp_{ts}.pth")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    print(f"\n[저장] 모델: {model_path}")

    # norm_stats
    norm_stats = {
        "output_mode": "normal",
        "input": {
            "rgb_mean": rgb_mean.tolist(),
            "rgb_std":  rgb_std.tolist(),
        },
        "output": {},
        "column_order": ["R", "G", "B", "X_px", "Y_px", "nx", "ny"],
    }
    stats_path = os.path.join(out_dir, f"norm_stats_{ts}.json")
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"[저장] norm_stats: {stats_path}")

    # loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("GelSight MLP Loss Curve")
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(out_dir, f"loss_curve_{ts}.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[저장] loss curve: {curve_path}")

    print(f"\n✅ 학습 완료! best val loss = {best_val_loss:.6f}")
    print(f"   출력 폴더: {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GelSight MLP 학습")
    parser.add_argument("--train-file", default=None,
                        help="train .npy 파일 경로 (미지정 시 최근 세션 자동 선택)")
    parser.add_argument("--val-file",   default=None,
                        help="val .npy 파일 경로")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=4096)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    train(args)