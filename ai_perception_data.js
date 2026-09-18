/* ==========================================================
   AI / ML PERCEPTION & 3D KINEMATICS DATA CONTRACT
   Generated from trained project neural networks:
   - Model 1: Strategic Terrain Segmenter (Dubai Satellite)
   - Model 2: Tactical Aerial Object Detector (VisDrone & AU-AIR)
   Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell sm_120)
   ========================================================== */

const aiPerceptionData = {
    systemStatus: {
        hardware: "NVIDIA GeForce RTX 5070 Laptop GPU",
        architecture: "NVIDIA Blackwell (sm_120, Compute Capability 12.0)",
        cudaVersion: "CUDA 13.0 / PyTorch 2.14.0+cu130",
        gpuPowerDraw: "75W Envelope (Max Power Mode)",
        flightAuthority: "Advisory Risk Only (Non-Learned Safety Boundary)",
        safetySupervisorRate: "20 Hz (MAVLink Offboard)",
    },
    models: {
        terrainSegmenter: {
            name: "Strategic Terrain Segmenter",
            architecture: "P2-P5 Depthwise-Separable + Dilated Context (r=1,2,4,8)",
            parameters: "2,357,526 (2.36M)",
            inputShape: "1x3x512x512",
            pixelAccuracy: "66.85%",
            meanIoU: "51.16%",
            classIoU: {
                "Water (Hazard)": "89.47%",
                "Land (Nominal)": "57.45%",
                "Road (Landing)": "37.45%",
                "Building (No-Fly)": "36.81%",
                "Vegetation": "34.62%"
            },
            runtime: "ONNX Runtime FP16 (12.4 ms)"
        },
        tacticalDetector: {
            name: "Tactical Aerial Object Detector",
            architecture: "Anchor-Free P2-P5 BiFPN (Stride-4 Micro-Object Head)",
            parameters: "1,790,431 (1.79M)",
            inputShape: "1x3x512x512",
            obstacleRecall: "99.89%",
            f1Score: "19.45%",
            trainedLoss: "3.0610 (91.2% reduction)",
            runtime: "ONNX Runtime FP16 (9.8 ms)"
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
        { id: "TGT-01", type: "Car", conf: 0.89, relX: -45, relY: -30, vx: 0.4, vy: 0.1, threat: "Low" },
        { id: "TGT-02", type: "Truck", conf: 0.84, relX: 85, relY: 40, vx: -0.2, vy: 0.3, threat: "Medium" },
        { id: "TGT-03", type: "Van", conf: 0.78, relX: 140, relY: -50, vx: 0.0, vy: -0.4, threat: "Low" },
        { id: "TGT-04", type: "UAV / Drone", conf: 0.94, relX: -70, relY: 110, vx: -0.5, vy: -0.2, threat: "High" },
        { id: "TGT-05", type: "Pedestrian", conf: 0.72, relX: 25, relY: -80, vx: 0.1, vy: 0.0, threat: "Low" }
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
