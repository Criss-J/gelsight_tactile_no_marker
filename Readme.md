# GelSight Tactile Sensor — Vision-Based 3D Reconstruction

GelSight Mini 제품과 [GelSight Wedge 논문](https://arxiv.org/abs/2106.08851)을 참고하여 자체 제작한 시각 기반 촉각센서 프로젝트입니다.
라즈베리파이 카메라로 엘라스토머(젤) 표면의 변형을 촬영하고, **MLP + Poisson Solver**를 통해 실시간으로 접촉면의 3D 형상(depth map)을 복원합니다.

<p align="center">
  <em>카메라 원본 | 실시간 Depth Map | 3D Point Cloud</em>
</p>

---

## 1. 하드웨어 구성

| 구성 요소 | 사양 |
|---|---|
| 카메라 | IMX219 (Raspberry Pi Camera Module v2), 640×480 |
| 컴퓨팅 | Raspberry Pi (Docker: Ubuntu 22.04 + ROS2 Humble) |
| 카메라 드라이버 | `camera_ros` 패키지 |
| 조명 | RGB LED, 각진 U자형 배치 (GelSight Wedge 스타일) |
| LED 밝기 조절 | 가변저항 |
| 엘라스토머 | 자체 제작 (실리콘 + 반사 코팅) |
| 네트워크 | 유선랜 권장 (WiFi는 3fps 제한) |

### 동작 원리

- 엘라스토머 표면에 물체가 닿으면 젤이 변형됨
- 서로 다른 방향에서 비추는 R/G/B LED가 변형된 표면에서 **방향에 따라 다른 색**으로 반사됨 (Photometric Stereo 원리)
- 카메라가 이 색 변화를 촬영 → 색으로부터 표면 기울기를 역산 → 기울기를 적분해 3D 형상 복원

---

## 2. 파이프라인

```
카메라 이미지 (640×480)
        ↓
[Step 1] MLP: [R, G, B, X_px, Y_px] → [nx, ny]   (단위 법선벡터)
        ↓
[Step 2] nz = √(1 − nx² − ny²),  Gx = −nx/nz,  Gy = −ny/nz
        ↓
[Step 3] Fast Poisson Solver (DCT + Neumann BC): [Gx, Gy] → depth map
        ↓
[Step 4] Baseline 보정 (초기 50프레임 평균 차감)
```

### 핵심 개념

**법선벡터(nx, ny)를 예측하는 이유**

초기에는 MLP가 기울기(Gx, Gy)를 직접 예측하도록 했으나, 구(sphere) 가장자리의 큰 기울기 값이 학습을 지배하여 전체적으로 기울기를 과예측하는 문제가 발생했습니다.
단위 법선벡터는 출력이 **[-1, 1]로 bounded** 되어 있어 이 문제가 해소됩니다.

**Poisson Solver (DCT + Neumann 경계조건)**

- MLP는 각 픽셀의 **기울기**만 예측하므로, 전체 형상을 얻으려면 기울기를 적분해야 함
- 2D 적분 문제를 Poisson 방정식으로 정식화: `∇²Z = ∂Gx/∂x + ∂Gy/∂y`
- DCT(이산 코사인 변환)로 주파수 영역에서 풀면 미분방정식이 단순 나눗셈으로 변환되어 실시간 처리 가능
- Neumann 경계조건: 이미지 경계에서 실측 기울기값을 사용 (젤 가장자리도 자유롭게 변형되므로 물리적으로 타당)

**Baseline Zeroing**

MLP 예측이 완벽하지 않아 비접촉 상태에서도 depth가 완전한 0이 아닐 수 있습니다.
노드 시작 후 초기 50프레임(비접촉 상태) 평균을 기준선으로 저장하고, 이후 모든 프레임에서 차감하여 "비접촉 = 0"을 보장합니다.

**Gx/Gy 클램핑**

강하게 누르면 접촉 가장자리에서 nx² + ny² → 1이 되어 nz → 0, 즉 Gx = −nx/nz가 발산합니다.
`np.clip(-nx/nz, -3.0, 3.0)`으로 클램핑하여 Poisson solver의 수치 폭발을 방지합니다.

---

## 3. 패키지 구조

```
src/gelsight_tactile/
├── gelsight_tactile/            # ROS2 노드 + 라이브러리
│   ├── camera_viewer.py          # LED 밝기 조절용 평균 RGB 모니터링 노드
│   ├── collect_calib_data.py       # 학습 데이터 수집 노드
│   ├── poisson_solver.py        # DCT Neumann Poisson solver (라이브러리)
│   ├── reconstructor.py         # Reconstruction3D 추론 엔진 (ROS 의존성 없음)
│   ├── reconstructor_node.py    # 실시간 depth map 퍼블리시 ROS2 노드
│   └── visualizer3d.py          # Open3D 실시간 3D 포인트클라우드 시각화
└── scripts/                     # 독립 실행 스크립트 (ROS 노드 아님)
    ├── train_mlp.py             # MLP 학습
    └── debug_gradients.py       # MLP 출력 시각화 진단
```

### 파일별 역할

| 파일 | 역할 |
|---|---|
| `collect_calib_data.py` | 반지름을 아는 교정 구슬을 눌러 `[R,G,B,X,Y] → [nx,ny]` 학습 데이터 수집 |
| `train_mlp.py` | 수집된 데이터로 MLP 학습, `.pth` + `norm_stats.json` 생성 |
| `debug_gradients.py` | 학습된 MLP가 배경에서 평평(≈0)하게 예측하는지 시각적 검증 |
| `poisson_solver.py` | 기울기 맵 → 깊이맵 적분 (gsrobotics 기반, GPL-3.0) |
| `reconstructor.py` | MLP 추론 + 기울기 변환 + Poisson 적분 + baseline 보정 |
| `reconstructor_node.py` | 카메라 구독 → depth map 퍼블리시 + OpenCV/Open3D 시각화 |
| `visualizer3d.py` | depth map을 3D 포인트클라우드로 실시간 렌더링 |

---

## 4. 데이터 폴더 구조

```
~/yang/data/nx_ny/
└── TIMESTAMP/                      # 수집 세션별 폴더
    ├── train_TIMESTAMP.npy         # (N, 7) [R,G,B,X_px,Y_px,nx,ny], 80%
    ├── val_TIMESTAMP.npy           # (N, 7), 20%
    ├── background_TIMESTAMP.png    # 비접촉 상태 배경 이미지
    ├── calib_TIMESTAMP.json        # px_per_mm, ball_radius_mm 등
    └── train_TRAIN_TS/             # 학습 결과 (train_mlp.py 실행 후 생성)
        ├── gelsight_mlp_TRAIN_TS.pth
        ├── norm_stats_TRAIN_TS.json
        └── loss_curve_TRAIN_TS.png
```

---

## 5. 사용 방법

### 5-0. 사전 준비

```bash
# PyTorch (CPU 버전)
pip3 install torch matplotlib --index-url https://download.pytorch.org/whl/cpu

# 빌드
cd ~/yang
colcon build --packages-select gelsight_tactile
source install/setup.bash
```

### 5-1. LED 밝기 조절 (최초 1회)

```bash
ros2 run gelsight_tactile camera_viewer
```
가변저항을 조절하며 R/G/B 평균값이 서로 비슷하고 과포화되지 않도록 맞춥니다.

### 5-2. 학습 데이터 수집

```bash
ros2 run gelsight_tactile collect_calib_data \
  --ros-args -p ball_radius_mm:=1.5 -p bg_exclusion_multiplier:=2.8
```

**수집 절차** (OpenCV 창에 포커스를 두고 키 입력):

1. `[b]` — 배경 캡처 (아무것도 닿지 않은 상태)
2. `[p]` — px/mm 캘리브레이션
   - 버니어 캘리퍼를 센서에 대고 양 끝 두 점을 클릭 → Enter → 터미널에 실측값(mm) 입력
   - 픽셀 좌표를 실제 물리 단위(mm)로 변환하기 위해 필수
3. `[s]` — 훈련 샘플 저장 (교정 구슬을 누른 상태에서, 여러 번 반복)
   - 자동 검출된 원을 `WASD`(이동) / `M,N`(반지름)으로 미세 조정 → Enter 확정
   - 구슬 위치를 화면 전체에 **골고루 분산**하여 30개 지점 이상 권장
4. `[q]` — 저장 및 종료

> **주의**: `ball_radius_mm`은 구슬의 **반지름**입니다. 직경 3mm 구슬 → `1.5`

### 5-3. MLP 학습

```bash
cd ~/yang
python3 src/gelsight_tactile/scripts/train_mlp.py --epochs 100
```
`~/yang/data/nx_ny/`에서 가장 최근 세션을 자동 선택하여 학습합니다.

### 5-4. 학습 결과 진단

```bash
python3 src/gelsight_tactile/scripts/debug_gradients.py
```
배경 이미지를 자동으로 불러와 MLP 예측을 시각화합니다.

**정상 기준** (배경 이미지 기준):
- Gx, Gy, nx, ny 맵이 전체적으로 0에 가까움
- 평균 경사각 5° 이하

### 5-5. 실시간 3D Reconstruction

**① 라즈베리파이에서 카메라 노드 실행**

```bash
# 라즈베리파이 SSH 접속
ssh <USER>@<RASPBERRY_PI_IP>

# Docker 컨테이너 시작 및 진입
docker start camera_ros
docker exec -it camera_ros /bin/bash

# ROS2 환경 설정
source /opt/ros/humble/setup.bash
source install/setup.bash

# 카메라 노드 실행 (수동 노출/화이트밸런스 고정)
ros2 run camera_ros camera_node --ros-args \
  -p sensor_mode:=1640:1232 \
  -p width:=640 \
  -p height:=480 \
  -p format:=RGB888 \
  -p role:=video \
  -p FrameDurationLimits:="[100000,100000]" \
  -p AeEnable:=false \
  -p AeExposureMode:=0 \
  -p AnalogueGain:=1.0 \
  -p ExposureTime:=33333 \
  -p AwbEnable:=false \
  -p ColourGains:="[1.0,1.0]"
```

> **자동 노출(AE)과 자동 화이트밸런스(AWB)를 반드시 꺼야 합니다.**
> 켜져 있으면 프레임마다 색이 변해 MLP 예측이 불안정해집니다.

**② 로컬 PC에서 카메라 토픽 수신 확인**

```bash
ros2 topic hz /camera/image_raw/compressed
```

**③ 로컬 PC에서 reconstructor_node 실행**

```bash
XDG_SESSION_TYPE=x11 GDK_BACKEND=x11 ros2 run gelsight_tactile reconstructor_node \
  --ros-args \
    -p model_path:=$(ls ~/yang/data/nx_ny/*/train_*/gelsight_mlp_*.pth | tail -1) \
    -p norm_stats_path:=$(ls ~/yang/data/nx_ny/*/train_*/norm_stats_*.json | tail -1) \
    -p px_per_mm:=12.1
```

- `model_path` / `norm_stats_path`: 위 명령어는 가장 최근 학습 결과를 자동 선택합니다. 특정 세션을 쓰려면 직접 경로 지정
- `px_per_mm`: 수집 세션의 `calib_TIMESTAMP.json`에서 확인
- Wayland 환경에서는 `XDG_SESSION_TYPE=x11 GDK_BACKEND=x11` 필수 (Open3D 호환성)
- 실행 후 **"Sensor ready."** 메시지가 뜰 때까지 센서를 건드리지 마세요 (baseline 계산 중)

**토픽**

| 방향 | 토픽 | 타입 |
|---|---|---|
| 구독 | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` |
| 퍼블리시 | `/gelsight/depth_map` | `sensor_msgs/Image` (32FC1, 픽셀 단위) |

> depth map의 단위는 **픽셀**입니다. 실제 mm 값이 필요하면 `px_per_mm`으로 나누세요.

---

## 6. MLP 상세

**아키텍처** (GelSight Wedge 논문 스펙):

```
Linear(5,32) → Tanh → Linear(32,32) → Tanh → Linear(32,32) → Tanh → Linear(32,2)
```

**입력 정규화**:
- R, G, B: 채널별 표준화 `(x − mean) / std` (train 세트 기준)
- X_px: `/640`, Y_px: `/480` → [0, 1]

**출력**: `[nx, ny]` — 정규화 불필요 (이미 [-1,1]로 bounded)

**학습 설정**: Adam (lr=1e-3), MSELoss, Batch=4096, Epochs=100

**레이블 생성 (구 기하학)**:

접촉원 내 각 픽셀에 대해, 반지름 `R`을 아는 구의 형상으로부터 해석적으로 계산:

```
x_mm = (px − cx) / px_per_mm
Gx = x_mm / √(R² − x_mm² − y_mm²)
n = [−Gx, −Gy, 1] / ‖·‖  →  (nx, ny)
```

**비접촉 샘플**: 접촉원 반지름의 2.8배(`bg_exclusion_multiplier`) 바깥 영역에서 접촉 픽셀 수의 25%를 `nx=ny=0`으로 샘플링 (데이터 균형 + 배경 왜곡 구역 제외)

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 배경인데 nx/ny에 큰 그라디언트 | 데이터 부족, 비접촉 샘플 비율 낮음 | 수집 지점 늘리기, 비접촉 비율 25% 확인 |
| 강하게 누르면 depth map 폭발 | nz→0으로 Gx/Gy 발산 | `reconstructor.py`의 clip(-3,3) 확인 |
| Open3D 창 안 뜸 (Wayland) | GLFW Wayland 비호환 | `XDG_SESSION_TYPE=x11 GDK_BACKEND=x11` 접두 |
| `KeyError: 'state_dict'` | 모델 저장 형식 불일치 | `reconstructor.py`의 load_model이 두 형식 모두 처리하는지 확인 |
| matplotlib 한글 깨짐 | 한글 폰트 미설치 | `sudo apt install fonts-nanum` |

---

## 8. 참고 자료

- **GelSight Wedge** — Wang et al., *GelSight Wedge: Measuring High-Resolution 3D Contact Geometry with a Compact Robot Finger* (ICRA 2021)
- **gsrobotics** — [gelsightinc/gsrobotics](https://github.com/gelsightinc/gsrobotics) (Poisson solver, 시각화 구조 참고, GPL-3.0)
- **Poisson Solver 이론** — DCT 기반 Neumann 경계조건 적분

---

## 9. 실행 화면 및 GUI 화면
<img width="631" height="479" alt="image" src="https://github.com/user-attachments/assets/ab24a242-cf73-45fb-8113-614b6244348c" />
<img width="1294" height="1001" alt="image" src="https://github.com/user-attachments/assets/58af6f91-fa44-4f2c-ad08-346a2f7b9034" />



## License

`poisson_solver.py`는 GPL-3.0 라이선스인 gsrobotics 코드를 기반으로 하므로, 본 저장소 전체에 GPL-3.0이 적용됩니다.
