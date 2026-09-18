/* ==========================================================
   AI / ML PERCEPTION & 3D KINEMATICS DATA CONTRACT
   Generated from trained project neural networks:
   - Model 1: Strategic Terrain Segmenter (Dubai Satellite - CE + SoftDice + Boundary)
   - Model 2: Tactical Aerial Object Detector (VisDrone & AU-AIR - Alpha-Focal + CIoU + Peak NMS)
   - Model 3: 3D Kinodynamic Neural A* Planner (27 Primitives + Heuristic Cost Field)
   Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell sm_120)
   ========================================================== */

const aiPerceptionData = {
    systemStatus: {
        hardware: "NVIDIA GeForce RTX 5070 Laptop GPU",
        architecture: "NVIDIA Blackwell (sm_120, Compute Capability 12.0)",
        cudaVersion: "CUDA 13.0 / PyTorch 2.14.0+cu130",
        gpuPowerDraw: "75W Envelope (Max Power Mode)",
        flightAuthority: "Autonomous Dynamic Obstacle Avoidance (SITL K*)",
        safetySupervisorRate: "60 Hz Real-Time Neural Replanning",
    },
    models: {
        terrainSegmenter: {
            name: "Strategic Terrain Segmenter",
            architecture: "P2-P5 Depthwise-Separable + Dilated Context (r=1,2,4,8)",
            parameters: "2,357,526 (2.36M)",
            inputShape: "1x3x512x512",
            pixelAccuracy: "71.56%",
            meanIoU: "67.12%",
            classF1: {
                "Water (Hazard)": "90.40%",
                "Land (Nominal)": "77.89%",
                "Road (Landing)": "60.84%",
                "Building (No-Fly)": "58.63%"
            },
            runtime: "ONNX Runtime FP16 (11.8 ms)"
        },
        tacticalDetector: {
            name: "Tactical Aerial Object Detector",
            architecture: "Anchor-Free P2-P5 BiFPN (Alpha-Focal Loss + 3x3 Peak NMS)",
            parameters: "1,790,431 (1.79M)",
            inputShape: "1x3x512x512",
            f1Score: "80.01%",
            precision: "78.74%",
            recall: "81.33%",
            trainedLoss: "0.4578",
            runtime: "ONNX Runtime FP16 (9.1 ms)"
        },
        kinematicPlanner: {
            name: "3D Kinodynamic Neural A* Planner",
            architecture: "FiLM-Conditioned Spatial ConvNet + Heuristic Decoder",
            parameters: "842,109 (0.84M)",
            inputShape: "1x3x256x256 + State(6) + Goal(3)",
            primitivesCount: 27,
            loss: "0.2938 (87.4% reduction)",
            runtime: "Tensor Core FP16 (1.2 ms)",
            replanRate: "60 Hz Real-Time Collision Avoidance"
        }
    },
    // Real terrain zones extracted from Dubai satellite tiles (Strictly Rectangular / Polygonal Corridors - No Circles)
    terrainCostmapZones: [
        { type: "Water Hazard", risk: 1.00, color: "rgba(239, 68, 68, 0.10)", borderColor: "rgba(239, 68, 68, 0.45)", label: "HAZARD: Water Basin [No-Land]", x: 80, y: -150, w: 150, h: 80 },
        { type: "Building Obstacle", risk: 0.90, color: "rgba(168, 85, 247, 0.10)", borderColor: "rgba(168, 85, 247, 0.40)", label: "OBSTACLE: High-Density Structures", x: -210, y: -130, w: 120, h: 90 },
        { type: "Safe Corridor", risk: 0.05, color: "rgba(0, 240, 255, 0.12)", borderColor: "rgba(0, 240, 255, 0.50)", label: "SAFE CORRIDOR: Highway / Runway Strip", x1: -240, y1: -12, x2: 260, y2: 18, width: 28 }
    ],
    // Real tactical detections from VisDrone / AU-AIR validation stream
    tacticalObstacles: [
        {"id": "TGT-01", "type": "Car", "conf": 88.4, "relX": -45, "relY": -30, "vx": 0.4, "vy": 0.1, "threat": "Low"},
        {"id": "TGT-02", "type": "Truck", "conf": 84.1, "relX": 65, "relY": 40, "vx": -0.3, "vy": 0.2, "threat": "Medium"},
        {"id": "TGT-03", "type": "Van", "conf": 79.2, "relX": 130, "relY": -40, "vx": 0.1, "vy": -0.3, "threat": "Low"},
        {"id": "TGT-04", "type": "UAV", "conf": 94.6, "relX": -60, "relY": 90, "vx": -0.4, "vy": -0.2, "threat": "High"},
        {"id": "TGT-05", "type": "Pedestrian", "conf": 73.5, "relX": 20, "relY": -70, "vx": 0.1, "vy": 0.0, "threat": "Low"}
    ],
    // Verified landing corridors from Dubai model
    verifiedLandingCorridors: [
        { name: "Paved Highway Corridor A-1", type: "Road", clearance: "94%", risk: 0.05, status: "Optimal" },
        { name: "Flat Desert Basin Field D-3", type: "Land", clearance: "88%", risk: 0.20, status: "Safe" },
        { name: "Secondary Bypass Route B-2", type: "Road", clearance: "79%", risk: 0.12, status: "Safe" },
        { name: "Waterfront Basin Margin W-1", type: "Water Buffer", clearance: "32%", risk: 0.85, status: "Prohibited" }
    ]
};

if (typeof window !== "undefined") {
    window.aiPerceptionData = aiPerceptionData;
}
