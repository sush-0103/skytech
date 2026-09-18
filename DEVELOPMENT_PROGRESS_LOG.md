# SkyTech UAV Autonomous Navigation & Digital Twin Stack
## Development Progress & Milestone Audit Log

> **Repository**: [https://github.com/sush-0103/skytech.git](https://github.com/sush-0103/skytech.git)  
> **System Architecture**: Companion Computer (Learned Perception + Deterministic Planner) $\to$ 20 Hz MAVLink $\to$ Autopilot (PX4 / ArduPilot SITL)  
> **Target Hardware**: NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell `sm_120`, 8 GB VRAM) + Intel Core Ultra 9  
> **Primary Safety Rule**: Learned models propose observations and soft costs only; an independent non-learned Deterministic Safety Supervisor must approve all kinematic setpoints before transmission.

---

## 1. Executive Summary: What Has Been Built "Till Now"

| Component / Layer | Implementation Status | Benchmark Performance | Operational Role |
|---|---|---|---|
| **Dataset Sanitization & Audit** | Complete (Milestones 0 & 1) | 0% Data Leakage, 100% Palette Fixed | Quarantined corrupt labels, repaired Dubai palette defect |
| **Model 1: Terrain Segmenter** | Complete (Milestone 2) | **67.12% mIoU**, 71.56% Pixel Acc | Generates 2D costmaps, detects water hazards & safe highway landing strips |
| **Model 2: Tactical Detector** | Complete (Milestone 2) | **80.01% F1**, 78.74% Prec, 81.33% Rec | High-resolution P2–P5 detection of moving vehicles, UAVs, and people |
| **Model 3: 3D Kinematic A\* Planner** | Complete (Milestones 2 & 3) | **0.2938 Loss**, 60 Hz Replan | 27 kinodynamic motion primitives for reactive collision avoidance |
| **Model 4: OpenSky 4D Predictor** | Complete (Milestone 2) | **91.73% F1**, **100% Precision**, 97.22% Acc | 1D-ResNet + Multi-Head Self-Attention for ADS-B airspace conflict prediction |
| **Tactical Canvas Avionics UI** | Complete (Milestone 3) | 60 FPS, Strictly NO Circles | Single drone flight, military HUD, live fluctuating percentages, dynamic evasion |
| **Local Perception Daemon** | Complete (Port 5001 / 3000) | Concurrent Tensor Core FP16 | Multi-model live inference API proxy serving real-time perception vectors |

---

## 2. Milestone-by-Milestone Progress Archive

### Milestone 0: Workspace Audit, Defect Quarantine & Baseline Setup
* **Date**: September 18, 2026
* **Why**: The raw dataset bundle contained critical label defects that would corrupt model training if used naively:
  1. **Dubai Aerial Segmentation**: Five of six RGB class definitions in `classes.json` were mismatched against the actual masks. A standard loader would misclassify 99.49% of pixels as unmapped void.
  2. **VisDrone 2019-DET**: 3 boxes had non-positive dimensions ($w \le 0$ or $h \le 0$), and 14,200 score-zero rows had to be classified as ignore masks rather than false background negatives.
  3. **AU-AIR Multimodal UAV**: 54 non-positive boxes quarantined; inter-frame telemetry gaps up to 52.6s identified to prevent invalid interpolation.
  4. **OpenSky ADS-B**: Stale data points ($>15\text{s}$) isolated; 187 clean airborne aircraft tracks extracted across 20 temporal snapshots.
  5. **NASA C-MAPSS**: Piecewise-linear RUL clipping applied at 125 cycles; 4 zero-variance sensor columns dropped.
* **What was Done**:
  - Implemented `prepare_dubai.py`, `prepare_visdrone.py`, `prepare_auair.py`, `prepare_opensky.py`, and `prepare_cmapss.py`.
  - Generated audit report `Datasets/autonomous_navigation_system_audit_and_plan.md` establishing safety principles and strict prohibition of unverified pre-trained black-box checkpoints.

---

### Milestone 1: Data Ingestion Pipelines & Leakage Prevention
* **Date**: September 18, 2026
* **Why**: Random row-based splitting across video clips or satellite tiles causes severe perceptual and temporal data leakage, artificially inflating validation metrics while destroying real-world generalization.
* **What was Done**:
  - **Dubai Segmentation**: Grouped 72 satellite image slices by their 6 parent source tiles (Tile 1 through Tile 8) for strict leave-one-parent-tile-out cross-validation.
  - **VisDrone & AU-AIR**: Enforced split grouping strictly by video flight clips (`train`, `val`, `test-dev`).
  - **OpenSky ADS-B**: Partitioned by unique aircraft `icao24` transponder identifiers (128 train aircraft, 33 validation aircraft) ensuring zero trajectory temporal overlap.
  - **Smoke Test Suite**: Implemented `test_dataloaders.py` validating batch dimensions, tensor ranges, and label alignments.

---

### Milestone 2: Custom Neural Stack Training (Zero Pre-Trained Weights)
* **Date**: September 18, 2026
* **Why**: Real-time UAV flight computers require lightweight, latency-bounded neural networks without bloated generic backbones. All models were built from scratch and trained on the local **RTX 5070 Laptop GPU** (`sm_120`, Blackwell architecture).

#### Model 1: Strategic Terrain Segmenter
* **Architecture**: Depthwise-separable inverted residual stages (P2=64, P3=128, P4=192, P5=256) + multi-scale dilated context at P4 ($r=1, 2, 4, 8$) + auxiliary Sobel boundary edge prediction head.
* **Parameters**: 2,357,526 (2.36M weights).
* **Loss**: Class-balanced Cross-Entropy + Soft Multiclass Dice + Boundary BCE.
* **Trained Performance**:
  - **Mean IoU (mIoU)**: **67.12%**
  - **Pixel Accuracy**: **71.56%**
  - **Water Hazard F1**: **90.40%** (critical for preventing water ditching)
  - **Land F1**: **77.89%** | **Road F1**: **60.84%** | **Building F1**: **58.63%**
* **Runtime**: Static ONNX Opset 18 (`checkpoints/terrain_segmenter_512.onnx`), 11.8 ms inference.

#### Model 2: Tactical Aerial Object Detector
* **Architecture**: Anchor-free P2–P5 Bi-directional Feature Pyramid Network (BiFPN) featuring an explicit high-resolution P2 stride-4 head ($128 \times 128$) to reliably preserve tiny aerial objects (median area $0.046\%$).
* **Parameters**: 1,790,431 (1.79M weights).
* **Loss**: $\alpha$-Balanced Focal Loss + Complete IoU (CIoU) with $3 \times 3$ Peak NMS.
* **Trained Performance**:
  - **F1-Score**: **80.01%**
  - **Precision**: **78.74%** (98.2% reduction in false alarms via peak NMS)
  - **Recall**: **81.33%**
  - **Loss**: 0.4578
* **Runtime**: ONNX FP16 (`checkpoints/tactical_detector_best.pt`), 9.1 ms inference.

#### Model 3: 3D Kinodynamic Neural A* Planner
* **Architecture**: FiLM-conditioned spatial ConvNet + kinematic state feature fusion + heuristic cost decoder.
* **Parameters**: 842,109 (0.84M weights).
* **Kinematic Library**: 27 3D motion primitives obeying velocity ($15\text{ m/s}$), acceleration ($4.0\text{ m/s}^2$), vertical climb ($2.5\text{ m/s}$), and jerk limits ($8.0\text{ m/s}^3$).
* **Trained Performance**:
  - **Heuristic Loss**: Reduced from $2.34$ to **0.2938** ($87.4\%$ error reduction).
  - **Convergence**: Reached optimal primitive ranking across all flight vectors.
* **Runtime**: 1.2 ms on Tensor Cores; supports 60 Hz real-time reactive replanning.

#### Model 4: OpenSky 4D Airspace Traffic & Conflict Predictor
* **Architecture**: 1D Dilated Temporal Residual Convolutions ($d=1, 2$) + 4-Head Temporal Self-Attention ($d_{model}=128, d_k=32$) with dual heads (Conflict Classifier + 4-step future 3D trajectory horizon).
* **Parameters**: 265,546 (0.27M weights).
* **Dataset**: OpenSky ADS-B flight trajectories (1,568 train windows, 395 val windows).
* **Trained Performance (15 Epochs on RTX 5070 GPU)**:
  - **F1-Score**: **91.73%**
  - **Precision**: **100.00% (Zero False Alarms)**
  - **Recall**: **84.72%** (61/72 conflict scenarios caught)
  - **Overall Accuracy**: **97.22%**
  - **Optimal Decision Threshold**: $\tau^* = 0.48$
* **Runtime**: Exported ONNX Opset 18 (`checkpoints/airspace_predictor.onnx`), 0.8 ms inference.

---

### Milestone 3: Tactical Canvas Avionics & 60 FPS Dynamic Obstacle Avoidance
* **Date**: September 18, 2026
* **Why**: To deliver an operational Software-In-The-Loop (SITL) digital twin visual interface demonstrating the combined neural stack in real time.
* **UI Constraints Enforced**:
  1. **Strictly ZERO Circles**: All radar pings, aircraft symbols, bounding boxes, waypoints, reticles, and buttons use rectilinear, square, diamond ($45^\circ$), or polyline vector shapes.
  2. **Single Drone Flight**: Focuses on one tactical UAV navigating through an active airspace.
  3. **Live Fluctuating Metrics**: Confidence percentages, GPU loads, latencies, and clearance margins fluctuate dynamically.
* **What was Done**:
  - **Dynamic Collision Avoidance**: Continuously checks a forward collision cylinder ($180\text{px}$ lookahead, $62\text{px}$ lateral buffer) against roving vehicles, UAVs, and pedestrians.
  - **Kinematic Spline Generation**: Generates entry, apex, and exit waypoints with turn-rate damping ($4.0^\circ/\text{frame}$) and clear-path recovery.
  - **Cooperative Air Traffic Display**: Visualizes commercial flights (`IGO1477 [FL300]`, `AIC883 [FL350]`, `SEJ214 [FL279]`) using diamond transponder symbols, velocity vectors, and ATC callout boxes.
  - **AI HUD**: Displays status of all 4 models, active primitives, clearance, and hardware telemetry.
  - **AI Daemon (`src/inference/ai_server.py`)**: Runs on port 5001, providing live unified perception to the Node.js frontend (`server.js` on port 3000).

---

## 3. Active System State

* **Web UI**: `http://localhost:3000` (Node.js daemon task-1486)
* **AI Perception Service**: `http://localhost:5001/api/ai/live` (Python daemon task-2050)
* **Git Remote**: `https://github.com/sush-0103/skytech.git` (synchronized on `main` at commit `e38fc7f`)

---

## 4. Milestone Log: Continuous Appends (What We Did & Why)

---

### Milestone 4: ArduPilot / PX4 MAVLink SITL Closed-Loop Bridge & Safety Supervisor
* **Date**: September 18, 2026
* **Status**: Complete & Verified (100% Invariant & Streaming Tests Passed)

#### Why It Was Done:
1. **Safety Boundary of Separation**:
   - Per the audit mandate in `Datasets/autonomous_navigation_system_audit_and_plan.md`, learned neural networks (such as our 3D Kinematic A* Planner, Tactical Detector, and OpenSky Predictor) must operate strictly as **observation and candidate risk proposal layers**.
   - Under no circumstances may unverified neural weights directly drive flight actuators or declare unmapped space safe.
   - An independent, deterministic **Safety Supervisor** is required to intercept, validate, or reject every candidate command before transmission to the flight controller.
2. **Autopilot Offboard Compatibility**:
   - Autopilots (PX4 and ArduPilot) require continuous high-rate proof-of-life heartbeats (1 Hz) and tightly timed position/velocity setpoints (20 Hz, 50 ms period) over MAVLink. If setpoints drop below rate, the flight controller immediately disengages offboard mode and executes failsafes (RTL or land).

#### What Was Done:
1. **Deterministic Safety Supervisor (`src/interface/safety_supervisor.py`)**:
   - **Geofence Enforcement**: Rejects any candidate setpoint outside declared operational boundaries ($X \in [-500, 500]\text{m}$, $Y \in [-500, 500]\text{m}$, $Z \in [-120, -5]\text{m}$ in Local NED coordinates).
   - **Kinodynamic Envelope Verification**:
     - Maximum speed: $v_{max} = 15.0\text{ m/s}$
     - Maximum acceleration: $a_{max} = 4.0\text{ m/s}^2$
     - Maximum climb/descent rate: $v_{z,max} = 2.5\text{ m/s}$
     - Jerk limit: $j_{max} = 8.0\text{ m/s}^3$
   - **Obstacle Proximity Buffer**: Rejects any setpoint passing within $15.0\text{m}$ of detected tactical obstacles.
   - **Freshness Gate**: Rejects stale commands older than $50\text{ms}$ ($0.050\text{s}$).
   - **Multi-Tier Failsafe State Machine**:
     - `OFFBOARD_ACTIVE`: Nominal 20 Hz flight commands accepted.
     - `FAILSAFE_BRAKE`: Immediate zero-velocity hold setpoint emitted upon violation.
     - `FAILSAFE_HOLD`: Sustained position hold on repeated violations ($\ge 3$).
     - `EMERGENCY_LAND`: Controlled auto-descent ($v_z = +0.5\text{ m/s}$) on critical violations ($\ge 10$).
     - **Bumpless Transfer**: Internal state automatically synchronizes with the emitted failsafe setpoint to prevent acceleration spikes upon recovery.

2. **MAVLink 20 Hz SITL Telemetry Bridge (`src/interface/mavlink_bridge.py`)**:
   - Connects to local ArduPilot / PX4 SITL companion port over UDP (`udp://127.0.0.1:14550`).
   - Emits standard `SET_POSITION_TARGET_LOCAL_NED` (MAVLink Message #84) in `MAV_FRAME_LOCAL_NED` at **20 Hz**.
   - Emits standard `HEARTBEAT` (Message #0) with `MAV_TYPE_ONBOARD_CONTROLLER` at **1 Hz**.
   - Incorporates a continuous kinodynamic acceleration rate-limiter ($a \le 3.5\text{ m/s}^2$) ensuring smooth trajectory interpolation between planner replanning cycles.

3. **Automated Verification Test Suite (`tests/test_mavlink_bridge.py`)**:
   - Test 1: Verified all 8 safety supervisor invariants (nominal acceptance, excess speed rejection, geofence rejection, altitude ceiling rejection, climb rate rejection, staleness rejection, obstacle buffer rejection, and failsafe ladder transitions).
   - Test 2: Spawns mock SITL receiver on UDP port, decoding 31 setpoint packets and 2 heartbeat packets over 1.5 seconds with exact floating-point coordinate parity.
   - Result: **100% Tests Passed**.

4. **Avionics & Simulation Integration**:
   - `src/inference/ai_server.py`: Integrated `MAVLinkSITLBridge` streaming live 20 Hz commands from the 3D Kinematic A* Planner; added `/api/mavlink/status` route.
   - `index.html`: Added MAVLink 20 Hz SITL Telemetry Bridge card in Section 5 with live stream rate, packet count, supervisor state, and NED velocity.
   - `simulation.js`: Updated AI HUD banner with MAVLink status row (`MAVLink SITL Bridge: 20 Hz (UDP:14550) | Sup: OFFBOARD_ACTIVE`).

---
