/* ===================================
   AUTONOMOUS DRONE SIMULATION
   Interactive UI & Canvas Engine
   =================================== */

// ─── State Management ───────────────────
const state = {
    simulation: {
        running: false,
        speed: 1,
        time: 0,
        fps: 60,
    },
    traffic: {
        radius: 5.0,
        level: 3,
        aircraftCount: 24,
    },
    memory: {
        allocation: 512,
        cpuCores: 4,
    },
    power: {
        batteryLevel: 78,
        consumption: 245,
        criticalThreshold: 15,
        emergencyAutoland: true,
    },
    canvas: {
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
        mouseX: 0,
        mouseY: 0,
        isDragging: false,
    },
    drones: [],
    waypoints: [],
    landingZones: [
        { name: 'Zone Alpha - Helipad A3', dist: 1.2, dir: 'NE', x: 0, y: 0 },
        { name: 'Zone Bravo - Open Field', dist: 3.4, dir: 'SW', x: 0, y: 0 },
        { name: 'Zone Charlie - Rooftop B7', dist: 5.1, dir: 'E', x: 0, y: 0 },
        { name: 'Zone Delta - Parking Lot', dist: 8.9, dir: 'N', x: 0, y: 0 },
        { name: 'Zone Echo - Base Station', dist: 14.2, dir: 'W', x: 0, y: 0 },
    ],
};

// ─── DOM References ─────────────────────
const canvas = document.getElementById('sim-canvas');
const ctx = canvas.getContext('2d');

// ─── Canvas Setup & Resize ─────────────
function resizeCanvas() {
    const viewport = document.getElementById('viewport');
    const rect = viewport.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = (rect.height - 28) * window.devicePixelRatio; // minus coord bar
    canvas.style.width = rect.width + 'px';
    canvas.style.height = (rect.height - 28) + 'px';
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

window.addEventListener('resize', () => {
    resizeCanvas();
});

// ─── Initialize Drones ─────────────────
function initDrones() {
    const cw = canvas.width / window.devicePixelRatio;
    const ch = canvas.height / window.devicePixelRatio;
    const cx = cw / 2;
    const cy = ch / 2;

    state.drones = [
        { id: 1, x: cx - 120, y: cy - 80, targetX: cx + 150, targetY: cy + 100, speed: 0.8, heading: 45, color: '#00f0ff', trail: [] },
        { id: 2, x: cx + 100, y: cy - 120, targetX: cx - 130, targetY: cy + 60, speed: 0.6, heading: 210, color: '#7b61ff', trail: [] },
        { id: 3, x: cx - 180, y: cy + 60, targetX: cx + 120, targetY: cy - 90, speed: 0.7, heading: 330, color: '#22c55e', trail: [] },
        { id: 4, x: cx + 60, y: cy + 130, targetX: cx - 80, targetY: cy - 130, speed: 0.5, heading: 120, color: '#eab308', trail: [] },
    ];

    // Initialize landing zone positions
    state.landingZones[0] = { ...state.landingZones[0], x: cx + 160, y: cy - 100 };
    state.landingZones[1] = { ...state.landingZones[1], x: cx - 170, y: cy + 120 };
    state.landingZones[2] = { ...state.landingZones[2], x: cx + 200, y: cy + 20 };
    state.landingZones[3] = { ...state.landingZones[3], x: cx - 40, y: cy - 180 };
    state.landingZones[4] = { ...state.landingZones[4], x: cx - 250, y: cy - 30 };
}

// ─── Draw Grid ──────────────────────────
function drawGrid() {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    const gridSize = 50 * state.canvas.zoom;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 0.5;

    const offsetX = (state.canvas.offsetX % gridSize);
    const offsetY = (state.canvas.offsetY % gridSize);

    for (let x = offsetX; x < w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }

    for (let y = offsetY; y < h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Major grid lines
    const majorGridSize = gridSize * 5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;

    for (let x = offsetX; x < w; x += majorGridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }

    for (let y = offsetY; y < h; y += majorGridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
}

// ─── Draw Radius Circle ─────────────────
function drawRadiusCircle() {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    const cx = w / 2 + state.canvas.offsetX;
    const cy = h / 2 + state.canvas.offsetY;
    const radiusPx = state.traffic.radius * 18 * state.canvas.zoom;

    // Outer radius
    ctx.beginPath();
    ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([8, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Radius fill
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radiusPx);
    gradient.addColorStop(0, 'rgba(0, 240, 255, 0.03)');
    gradient.addColorStop(0.7, 'rgba(0, 240, 255, 0.01)');
    gradient.addColorStop(1, 'transparent');
    ctx.fillStyle = gradient;
    ctx.fill();

    // Inner safe zone
    const safeRadius = radiusPx * 0.6;
    ctx.beginPath();
    ctx.arc(cx, cy, safeRadius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Center marker
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy);
    ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, cy + 10);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
}

// ─── Draw Traffic Dots ──────────────────
function drawTrafficDots(time) {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    const cx = w / 2 + state.canvas.offsetX;
    const cy = h / 2 + state.canvas.offsetY;
    const radiusPx = state.traffic.radius * 18 * state.canvas.zoom;
    const count = Math.floor(state.traffic.level * 6);

    for (let i = 0; i < count; i++) {
        const angle = (i / count) * Math.PI * 2 + time * 0.0002 * (i % 3 === 0 ? 1 : -1);
        const dist = radiusPx * (0.3 + Math.sin(i * 1.7 + time * 0.001) * 0.3);
        const x = cx + Math.cos(angle) * dist;
        const y = cy + Math.sin(angle) * dist;

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fillStyle = i % 4 === 0 ? 'rgba(239, 68, 68, 0.6)' :
                        i % 3 === 0 ? 'rgba(234, 179, 8, 0.5)' :
                        'rgba(100, 116, 139, 0.4)';
        ctx.fill();
    }
}

// ─── Draw Landing Zones ─────────────────
function drawLandingZones() {
    state.landingZones.forEach((zone, i) => {
        const x = zone.x + state.canvas.offsetX;
        const y = zone.y + state.canvas.offsetY;
        const reachable = zone.dist <= (state.power.batteryLevel / 100 * 15);
        const marginal = zone.dist <= (state.power.batteryLevel / 100 * 18);

        const color = reachable ? '#22c55e' : marginal ? '#eab308' : '#ef4444';
        const alpha = reachable ? 0.6 : marginal ? 0.4 : 0.2;

        // Landing pad
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.globalAlpha = 1;

        // "H" marker
        ctx.fillStyle = color;
        ctx.font = '8px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('H', x, y);

        // Label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.font = '8px Inter';
        ctx.fillText(zone.name.split(' - ')[0], x, y + 20);
    });
}

// ─── Draw Drone ─────────────────────────
function drawDrone(drone, time) {
    const x = drone.x + state.canvas.offsetX;
    const y = drone.y + state.canvas.offsetY;
    const headingRad = (drone.heading * Math.PI) / 180;

    // Trail
    if (drone.trail.length > 1) {
        ctx.beginPath();
        ctx.moveTo(drone.trail[0].x + state.canvas.offsetX, drone.trail[0].y + state.canvas.offsetY);
        for (let i = 1; i < drone.trail.length; i++) {
            ctx.lineTo(drone.trail[i].x + state.canvas.offsetX, drone.trail[i].y + state.canvas.offsetY);
        }
        ctx.strokeStyle = drone.color + '30';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    // Path projection
    const projLen = 60;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(headingRad) * projLen, y + Math.sin(headingRad) * projLen);
    ctx.strokeStyle = drone.color + '20';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Glow
    const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
    glowGradient.addColorStop(0, drone.color + '25');
    glowGradient.addColorStop(1, 'transparent');
    ctx.fillStyle = glowGradient;
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, Math.PI * 2);
    ctx.fill();

    // Drone body
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(headingRad);

    // Arms
    for (let i = 0; i < 4; i++) {
        const armAngle = (i * Math.PI) / 2 + Math.PI / 4;
        const armLen = 14;
        const ax = Math.cos(armAngle) * armLen;
        const ay = Math.sin(armAngle) * armLen;

        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(ax, ay);
        ctx.strokeStyle = drone.color;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Propeller
        const propRadius = 5 + Math.sin(time * 0.02 + i) * 0.5;
        ctx.beginPath();
        ctx.arc(ax, ay, propRadius, 0, Math.PI * 2);
        ctx.strokeStyle = drone.color + '60';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Spinning effect
        const spinAngle = time * 0.05 * (i % 2 === 0 ? 1 : -1);
        ctx.beginPath();
        ctx.arc(ax, ay, propRadius, spinAngle, spinAngle + Math.PI);
        ctx.strokeStyle = drone.color + '90';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    // Center body
    ctx.beginPath();
    ctx.arc(0, 0, 4, 0, Math.PI * 2);
    ctx.fillStyle = drone.color;
    ctx.fill();

    // Direction indicator
    ctx.beginPath();
    ctx.moveTo(6, 0);
    ctx.lineTo(3, -2);
    ctx.lineTo(3, 2);
    ctx.closePath();
    ctx.fillStyle = drone.color;
    ctx.fill();

    ctx.restore();

    // ID label
    ctx.fillStyle = drone.color;
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`D${drone.id}`, x, y - 22);
}

// ─── Update Drone Positions ─────────────
function updateDrones() {
    state.drones.forEach(drone => {
        const dx = drone.targetX - drone.x;
        const dy = drone.targetY - drone.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > 5) {
            drone.x += (dx / dist) * drone.speed;
            drone.y += (dy / dist) * drone.speed;
            drone.heading = (Math.atan2(dy, dx) * 180) / Math.PI;

            // Add trail point
            drone.trail.push({ x: drone.x, y: drone.y });
            if (drone.trail.length > 80) drone.trail.shift();
        } else {
            // Swap target
            const cw = canvas.width / window.devicePixelRatio;
            const ch = canvas.height / window.devicePixelRatio;
            drone.targetX = Math.random() * (cw - 200) + 100;
            drone.targetY = Math.random() * (ch - 200) + 100;
        }
    });
}

// ─── AI Perception & Kinematics Rendering ────────
function drawAITerrainCostmap() {
    const data = window.aiPerceptionData;
    if (!data || !data.terrainCostmapZones) return;

    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    const cx = w / 2 + state.canvas.offsetX;
    const cy = h / 2 + state.canvas.offsetY;
    const zoom = state.canvas.zoom;

    data.terrainCostmapZones.forEach(zone => {
        if (zone.radius) {
            const zx = cx + zone.cx * zoom;
            const zy = cy + zone.cy * zoom;
            const zr = zone.radius * zoom;

            ctx.beginPath();
            ctx.arc(zx, zy, zr, 0, Math.PI * 2);
            ctx.fillStyle = zone.color;
            ctx.fill();

            ctx.strokeStyle = zone.risk > 0.8 ? 'rgba(239, 68, 68, 0.4)' : 'rgba(123, 97, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.font = '9px JetBrains Mono';
            ctx.textAlign = 'center';
            ctx.fillText(zone.label, zx, zy - zr - 4);
        } else if (zone.x1 !== undefined) {
            // Road corridor
            const x1 = cx + zone.x1 * zoom;
            const y1 = cy + zone.y1 * zoom;
            const x2 = cx + zone.x2 * zoom;
            const y2 = cy + zone.y2 * zoom;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = zone.color;
            ctx.lineWidth = zone.width * zoom;
            ctx.lineCap = 'round';
            ctx.stroke();

            ctx.fillStyle = '#6ec1e4';
            ctx.font = '9px JetBrains Mono';
            ctx.textAlign = 'center';
            ctx.fillText(zone.label, (x1 + x2) / 2, (y1 + y2) / 2 - 16);
        }
    });
}

function drawAITacticalObstacles(time) {
    const data = window.aiPerceptionData;
    if (!data || !data.tacticalObstacles) return;

    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    const cx = w / 2 + state.canvas.offsetX;
    const cy = h / 2 + state.canvas.offsetY;
    const zoom = state.canvas.zoom;

    data.tacticalObstacles.forEach((obs, idx) => {
        // Slight dynamic drift for live simulation
        const wobbleX = Math.sin(time * 0.001 + idx) * 15;
        const wobbleY = Math.cos(time * 0.001 + idx) * 10;
        const ox = cx + (obs.relX + wobbleX) * zoom;
        const oy = cy + (obs.relY + wobbleY) * zoom;
        const bw = 24 * zoom;
        const bh = 18 * zoom;

        // Tactical Bounding Box (AI Object Detection)
        ctx.strokeStyle = obs.threat === 'High' ? '#ef4444' : '#22c55e';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(ox - bw / 2, oy - bh / 2, bw, bh);

        // Class tag
        ctx.fillStyle = 'rgba(10, 14, 26, 0.85)';
        ctx.fillRect(ox - bw / 2, oy - bh / 2 - 14, bw + 28, 12);
        ctx.fillStyle = obs.threat === 'High' ? '#ff6b6b' : '#4ade80';
        ctx.font = '8px JetBrains Mono';
        ctx.textAlign = 'left';
        ctx.fillText(`${obs.type} ${(obs.conf * 100).toFixed(0)}%`, ox - bw / 2 + 2, oy - bh / 2 - 5);

        // Velocity vector
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ox + obs.vx * 30 * zoom, oy + obs.vy * 30 * zoom);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}

function drawKinodynamicKStarTrajectory(drone) {
    const dx = drone.targetX - drone.x;
    const dy = drone.targetY - drone.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 5) return;

    // Kinodynamic jerk-limited polynomial curve avoiding building & water
    const midX = (drone.x + drone.targetX) / 2 + Math.sin(drone.id * 1.5) * 40;
    const midY = (drone.y + drone.targetY) / 2 - 50;

    const x0 = drone.x + state.canvas.offsetX;
    const y0 = drone.y + state.canvas.offsetY;
    const xm = midX + state.canvas.offsetX;
    const ym = midY + state.canvas.offsetY;
    const xt = drone.targetX + state.canvas.offsetX;
    const yt = drone.targetY + state.canvas.offsetY;

    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(xm, ym, xt, yt);
    ctx.strokeStyle = drone.color + '55';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Target waypoint marker
    ctx.beginPath();
    ctx.arc(xt, yt, 4, 0, Math.PI * 2);
    ctx.fillStyle = drone.color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawAIHUD(w, h) {
    // Top-left AI Perception & GPU Telemetry HUD
    ctx.fillStyle = 'rgba(15, 21, 36, 0.85)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.fillRect(16, 16, 270, 78);
    ctx.strokeRect(16, 16, 270, 78);

    ctx.fillStyle = '#22c55e';
    ctx.beginPath();
    ctx.arc(28, 30, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText('AI PERCEPTION STACK: ACTIVE', 38, 33);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px Inter';
    ctx.fillText('Tactical Detector: ONNX (99.89% Recall)', 28, 50);
    ctx.fillText('Terrain Segmenter: ONNX (51.16% mIoU)', 28, 65);
    ctx.fillText('Authority: Advisory (SITL EKF2 Sovereign)', 28, 80);
}

// ─── Draw Range Rings ───────────────────
function drawRangeRings() {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    const cx = w / 2 + state.canvas.offsetX;
    const cy = h / 2 + state.canvas.offsetY;

    // Max range ring
    const maxRange = (state.power.batteryLevel / 100) * 200 * state.canvas.zoom;
    ctx.beginPath();
    ctx.arc(cx, cy, maxRange, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.08)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Safe return ring
    const safeRange = maxRange * 0.77;
    ctx.beginPath();
    ctx.arc(cx, cy, safeRange, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(234, 179, 8, 0.06)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Critical ring
    const criticalRange = maxRange * 0.17;
    ctx.beginPath();
    ctx.arc(cx, cy, criticalRange, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.06)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
}

// ─── Main Render Loop ───────────────────
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsUpdateTime = 0;

function render(timestamp) {
    const dt = timestamp - lastFrameTime;
    lastFrameTime = timestamp;

    // FPS calculation
    frameCount++;
    if (timestamp - fpsUpdateTime > 1000) {
        state.simulation.fps = frameCount;
        document.getElementById('fps-counter').textContent = frameCount;
        frameCount = 0;
        fpsUpdateTime = timestamp;
    }

    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background
    const bgGrad = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) * 0.7);
    bgGrad.addColorStop(0, '#0d1220');
    bgGrad.addColorStop(1, '#0a0e1a');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Draw base layers
    drawGrid();
    drawRangeRings();
    drawRadiusCircle();
    drawTrafficDots(timestamp);

    // ─── AI Perception & Kinematics Layers ───
    drawAITerrainCostmap();
    drawLandingZones();
    drawAITacticalObstacles(timestamp);

    // Draw active drones and kinodynamic trajectories
    if (state.drones && state.drones.length > 0) {
        state.drones.forEach(drone => {
            drawKinodynamicKStarTrajectory(drone);
            drawDrone(drone, timestamp);
        });
        if (state.simulation.running) {
            updateDrones();
        }
    }

    // AI Perception HUD Banner
    drawAIHUD(w, h);

    // Update sim time
    if (state.simulation.running) {
        state.simulation.time += dt * state.simulation.speed;
    }

    // Update sim clock
    updateSimClock();

    requestAnimationFrame(render);
}

// ─── Simulation Clock ───────────────────
function updateSimClock() {
    const totalSeconds = Math.floor(state.simulation.time / 1000);
    const hours = Math.floor(totalSeconds / 3600).toString().padStart(2, '0');
    const minutes = Math.floor((totalSeconds % 3600) / 60).toString().padStart(2, '0');
    const seconds = (totalSeconds % 60).toString().padStart(2, '0');
    document.getElementById('sim-clock').textContent = `${hours}:${minutes}:${seconds}`;
}

// ─── Sparkline Drawing ──────────────────
function drawSparkline(containerId, color, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const sparkCanvas = document.createElement('canvas');
    sparkCanvas.width = container.offsetWidth * 2;
    sparkCanvas.height = 32;
    sparkCanvas.style.width = '100%';
    sparkCanvas.style.height = '16px';
    container.innerHTML = '';
    container.appendChild(sparkCanvas);

    const sctx = sparkCanvas.getContext('2d');
    const w = sparkCanvas.width;
    const h = sparkCanvas.height;
    const step = w / (data.length - 1);

    sctx.beginPath();
    sctx.moveTo(0, h - (data[0] / Math.max(...data)) * h);
    for (let i = 1; i < data.length; i++) {
        const x = i * step;
        const y = h - (data[i] / Math.max(...data)) * h;
        sctx.lineTo(x, y);
    }
    sctx.strokeStyle = color;
    sctx.lineWidth = 1.5;
    sctx.stroke();

    // Fill
    sctx.lineTo(w, h);
    sctx.lineTo(0, h);
    sctx.closePath();
    const fillGrad = sctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, color + '30');
    fillGrad.addColorStop(1, 'transparent');
    sctx.fillStyle = fillGrad;
    sctx.fill();
}

function initSparklines() {
    drawSparkline('spark-distance', '#00f0ff', [2.1, 2.5, 2.3, 2.8, 2.4, 2.6, 2.2, 2.9, 2.4]);
    drawSparkline('spark-path', '#7b61ff', [7.5, 8.2, 7.9, 8.5, 8.1, 7.8, 8.3, 8.0, 8.1]);
    drawSparkline('spark-collision', '#22c55e', [1.0, 1.3, 1.1, 1.5, 1.2, 1.4, 1.1, 1.3, 1.2]);
    drawSparkline('spark-frame', '#eab308', [16.2, 16.8, 16.5, 17.1, 16.7, 16.4, 16.9, 16.6, 16.7]);
}

// ─── Event Handlers ─────────────────────

// Simulation Controls
document.getElementById('btn-play').addEventListener('click', () => {
    state.simulation.running = true;
    document.querySelector('.status-icon').className = 'status-icon running';
    document.querySelector('.status-item').childNodes[1].textContent = ' Simulation Running';
});

document.getElementById('btn-pause').addEventListener('click', () => {
    state.simulation.running = false;
    document.querySelector('.status-icon').className = 'status-icon ready';
    document.querySelector('.status-item').childNodes[1].textContent = ' Simulation Paused';
});

document.getElementById('btn-stop').addEventListener('click', () => {
    state.simulation.running = false;
    state.simulation.time = 0;
    state.drones.forEach(d => { d.trail = []; });
    initDrones();
    document.querySelector('.status-icon').className = 'status-icon ready';
    document.querySelector('.status-item').childNodes[1].textContent = ' Simulation Idle';
});

document.getElementById('sim-speed').addEventListener('change', (e) => {
    state.simulation.speed = parseFloat(e.target.value);
});

// Canvas Tabs
document.querySelectorAll('.canvas-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.canvas-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
    });
});

// Tool Buttons
document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// Nav Buttons
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// Section Toggle
document.querySelectorAll('.section-header').forEach(header => {
    header.addEventListener('click', () => {
        const targetId = header.getAttribute('data-toggle');
        const body = document.getElementById(targetId);
        const chevron = header.querySelector('.section-chevron');

        body.classList.toggle('open');
        if (body.classList.contains('open')) {
            chevron.style.transform = 'rotate(0deg)';
        } else {
            chevron.style.transform = 'rotate(-90deg)';
        }
    });
});

// Panel Collapse
document.getElementById('panel-collapse').addEventListener('click', () => {
    const panel = document.getElementById('right-panel');
    panel.classList.toggle('collapsed');
    const btn = document.getElementById('panel-collapse');
    if (panel.classList.contains('collapsed')) {
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M5 3l5 4-5 4V3z"/></svg>';
    } else {
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M9 3l-5 4 5 4V3z"/></svg>';
    }
    // Resize canvas after panel animation
    setTimeout(resizeCanvas, 450);
});

// Traffic Radius Slider
document.getElementById('traffic-radius').addEventListener('input', (e) => {
    state.traffic.radius = parseFloat(e.target.value);
    document.getElementById('radius-display').textContent = `${state.traffic.radius.toFixed(1)} km`;
});

// Traffic Level Buttons
document.querySelectorAll('.traffic-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.traffic-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.traffic.level = parseInt(btn.dataset.level);

        // Update density fill
        const fillPercent = (state.traffic.level / 5) * 100;
        document.getElementById('density-fill').style.width = fillPercent + '%';

        // Update traffic count in state
        const baseCount = state.traffic.level * 8 + Math.floor(Math.random() * 5);
        state.traffic.aircraftCount = baseCount;
    });
});

// Memory Allocation Slider
document.getElementById('memory-alloc').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    state.memory.allocation = val;
    let display = val >= 1024 ? `${(val / 1024).toFixed(1)} GB` : `${val} MB`;
    document.getElementById('memory-display').textContent = display;
});

// CPU Cores Slider
document.getElementById('cpu-cores').addEventListener('input', (e) => {
    const cores = parseInt(e.target.value);
    state.memory.cpuCores = cores;
    document.getElementById('cpu-display').textContent = `${cores} Core${cores > 1 ? 's' : ''}`;

    // Update core visuals
    for (let i = 1; i <= 8; i++) {
        const coreEl = document.getElementById(`core-${i}`);
        if (i <= cores) {
            coreEl.classList.add('active');
            const usage = Math.floor(Math.random() * 40 + 30);
            coreEl.querySelector('.core-fill').style.height = usage + '%';
            coreEl.querySelector('.core-pct').textContent = usage + '%';
        } else {
            coreEl.classList.remove('active');
            coreEl.querySelector('.core-fill').style.height = '0%';
            coreEl.querySelector('.core-pct').textContent = '—';
        }
    }
});

// Battery Level Slider
document.getElementById('battery-level').addEventListener('input', (e) => {
    const level = parseInt(e.target.value);
    state.power.batteryLevel = level;
    document.getElementById('battery-display').textContent = level + '%';

    // Update battery fill
    const fill = document.getElementById('battery-fill');
    fill.style.width = level + '%';

    // Color transition
    if (level > 50) {
        fill.style.background = 'linear-gradient(90deg, #22c55e, #4ade80)';
    } else if (level > 25) {
        fill.style.background = 'linear-gradient(90deg, #eab308, #facc15)';
    } else if (level > 10) {
        fill.style.background = 'linear-gradient(90deg, #f97316, #fb923c)';
    } else {
        fill.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }

    // Update voltage
    const voltage = (level / 100 * 8.4 + 17).toFixed(1);
    document.querySelector('.battery-voltage').textContent = voltage + 'V';

    // Update range estimation
    const maxRange = (level / 100 * 16).toFixed(1);
    const safeRange = (maxRange * 0.77).toFixed(1);
    const criticalRange = (maxRange * 0.17).toFixed(1);
    document.getElementById('range-max').textContent = maxRange + ' km';
    document.getElementById('range-safe').textContent = safeRange + ' km';
    document.getElementById('range-critical').textContent = criticalRange + ' km';

    // Update consumption
    const consumption = Math.floor(150 + (100 - level) * 2 + Math.random() * 20);
    state.power.consumption = consumption;
    document.getElementById('consumption-text').textContent = consumption + 'W';
    const arcOffset = 141.37 * (1 - consumption / 500);
    document.getElementById('consumption-arc').setAttribute('stroke-dashoffset', arcOffset);

    // Update landing zones reachability
    updateLandingZones(maxRange);
});

// Critical Threshold Slider
document.getElementById('critical-threshold').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    state.power.criticalThreshold = val;
    document.getElementById('threshold-display').textContent = val + '%';
});

// Emergency Auto-Land Toggle
document.getElementById('emergency-autoland').addEventListener('change', (e) => {
    state.power.emergencyAutoland = e.target.checked;
});

// Update Landing Zones
function updateLandingZones(maxRange) {
    const zones = document.querySelectorAll('.landing-zone');
    zones.forEach((zone, i) => {
        const lz = state.landingZones[i];
        if (lz.dist <= maxRange * 0.77) {
            zone.className = 'landing-zone reachable';
            zone.querySelector('.zone-status').textContent = 'Reachable';
        } else if (lz.dist <= maxRange) {
            zone.className = 'landing-zone warning';
            zone.querySelector('.zone-status').textContent = 'Marginal';
        } else {
            zone.className = 'landing-zone unreachable';
            zone.querySelector('.zone-status').textContent = 'Unreachable';
        }
    });
}

// Canvas Mouse Events
const viewport = document.getElementById('viewport');

viewport.addEventListener('mousemove', (e) => {
    const rect = viewport.getBoundingClientRect();
    state.canvas.mouseX = e.clientX - rect.left;
    state.canvas.mouseY = e.clientY - rect.top;

    // Update coordinates
    const worldX = ((state.canvas.mouseX - rect.width / 2) / state.canvas.zoom).toFixed(3);
    const worldY = ((state.canvas.mouseY - rect.height / 2) / state.canvas.zoom).toFixed(3);
    document.getElementById('coord-x').textContent = worldX;
    document.getElementById('coord-y').textContent = worldY;

    if (state.canvas.isDragging) {
        state.canvas.offsetX += e.movementX;
        state.canvas.offsetY += e.movementY;
    }
});

viewport.addEventListener('mousedown', (e) => {
    if (e.button === 1 || e.button === 0) {
        state.canvas.isDragging = true;
        viewport.style.cursor = 'grabbing';
    }
});

viewport.addEventListener('mouseup', () => {
    state.canvas.isDragging = false;
    viewport.style.cursor = 'crosshair';
});

viewport.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.95 : 1.05;
    state.canvas.zoom = Math.max(0.3, Math.min(5, state.canvas.zoom * zoomFactor));
    document.getElementById('coord-zoom').textContent = Math.round(state.canvas.zoom * 100);
});

viewport.style.cursor = 'crosshair';

// ─── Live Metrics Update ────────────────
function updateMetrics() {
    // Simulate fluctuating metrics
    const distCalc = (2 + Math.random() * 0.8).toFixed(1);
    const pathOpt = (7.5 + Math.random() * 1.5).toFixed(1);
    const collDet = (0.8 + Math.random() * 0.8).toFixed(1);
    const frameTime = (16 + Math.random() * 2).toFixed(1);

    document.getElementById('metric-distance').textContent = distCalc + ' ms';
    document.getElementById('metric-path').textContent = pathOpt + ' ms';
    document.getElementById('metric-collision').textContent = collDet + ' ms';
    document.getElementById('metric-frame').textContent = frameTime + ' ms';

    // Memory usage in status bar
    const memBase = Math.floor(state.memory.allocation * 0.6);
    const memVar = Math.floor(Math.random() * 50);
    document.getElementById('mem-usage').textContent = (memBase + memVar) + ' MB';

    // GPU
    const gpuBase = 20 + state.traffic.level * 5;
    const gpuVar = Math.floor(Math.random() * 8);
    document.getElementById('gpu-usage').textContent = (gpuBase + gpuVar) + '%';


}

// ─── Initialize ─────────────────────────
function init() {
    resizeCanvas();
    initDrones();
    initSparklines();

    // Start render loop
    requestAnimationFrame(render);

    // Start metrics update
    setInterval(updateMetrics, 1000);

    // Initial landing zones update
    updateLandingZones(12.4);
}

// Start
window.addEventListener('DOMContentLoaded', init);
