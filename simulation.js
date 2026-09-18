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
        { id: 1, x: cx - 120, y: cy - 80, targetX: cx + 150, targetY: cy + 100, speed: 1.0, heading: 45, color: '#00f0ff', trail: [] }
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

// ─── Tactical Overlays (Circles Removed) ─────────────────
function drawRadiusCircle() {
    // Deprecated / removed to keep tactical map clean without cluttering circles
}

function drawTrafficDots(time) {
    // Deprecated / removed - single-drone mode active
}

// ─── Draw Tactical Landing Zones (Square Helipads - No Circles) ────────
function drawLandingZones() {
    state.landingZones.forEach((zone, i) => {
        const x = zone.x + state.canvas.offsetX;
        const y = zone.y + state.canvas.offsetY;
        const reachable = zone.dist <= (state.power.batteryLevel / 100 * 15);
        const marginal = zone.dist <= (state.power.batteryLevel / 100 * 18);

        const color = reachable ? '#22c55e' : marginal ? '#eab308' : '#ef4444';
        const alpha = reachable ? 0.35 : marginal ? 0.25 : 0.15;

        // Tactical square helipad boundary
        const padSize = 22;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x - padSize / 2, y - padSize / 2, padSize, padSize);

        // Tactical corner brackets
        const bLen = 5;
        ctx.lineWidth = 1.5;
        // Top-left
        ctx.beginPath();
        ctx.moveTo(x - padSize / 2 - 3, y - padSize / 2 + bLen);
        ctx.lineTo(x - padSize / 2 - 3, y - padSize / 2 - 3);
        ctx.lineTo(x - padSize / 2 + bLen, y - padSize / 2 - 3);
        // Top-right
        ctx.moveTo(x + padSize / 2 + 3 - bLen, y - padSize / 2 - 3);
        ctx.lineTo(x + padSize / 2 + 3, y - padSize / 2 - 3);
        ctx.lineTo(x + padSize / 2 + 3, y - padSize / 2 + bLen);
        // Bottom-left
        ctx.moveTo(x - padSize / 2 - 3, y + padSize / 2 - bLen);
        ctx.lineTo(x - padSize / 2 - 3, y + padSize / 2 + 3);
        ctx.lineTo(x - padSize / 2 + bLen, y + padSize / 2 + 3);
        // Bottom-right
        ctx.moveTo(x + padSize / 2 + 3 - bLen, y + padSize / 2 + 3);
        ctx.lineTo(x + padSize / 2 + 3, y + padSize / 2 + 3);
        ctx.lineTo(x + padSize / 2 + 3, y + padSize / 2 - bLen);
        ctx.stroke();

        // Inner square fill
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha;
        ctx.fillRect(x - padSize / 3, y - padSize / 3, (padSize * 2) / 3, (padSize * 2) / 3);
        ctx.globalAlpha = 1;

        // "H" marker
        ctx.fillStyle = color;
        ctx.font = 'bold 9px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('H', x, y);

        // Label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
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
    // Tactical diamond reticle aura (No circles)
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.PI / 4);
    ctx.strokeStyle = drone.color + '25';
    ctx.lineWidth = 1;
    ctx.strokeRect(-12, -12, 24, 24);
    ctx.restore();

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

        // Realistic spinning rotor blade lines (No circular rings)
        const spinAngle = time * 0.06 * (i % 2 === 0 ? 1 : -1) + i * 1.2;
        const bladeLen = 6;
        ctx.beginPath();
        ctx.moveTo(ax - Math.cos(spinAngle) * bladeLen, ay - Math.sin(spinAngle) * bladeLen);
        ctx.lineTo(ax + Math.cos(spinAngle) * bladeLen, ay + Math.sin(spinAngle) * bladeLen);
        ctx.strokeStyle = drone.color + 'bb';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    // Tactical center body (Crisp Square - No Circles)
    ctx.fillStyle = drone.color;
    ctx.fillRect(-3, -3, 6, 6);

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
        if (zone.x !== undefined && zone.w !== undefined) {
            // Rectangular tactical hazard / obstacle zone (Strictly No Circles)
            const rx = cx + zone.x * zoom;
            const ry = cy + zone.y * zoom;
            const rw = zone.w * zoom;
            const rh = zone.h * zoom;

            ctx.fillStyle = zone.color;
            ctx.fillRect(rx, ry, rw, rh);

            ctx.strokeStyle = zone.borderColor || 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1.2;
            ctx.strokeRect(rx, ry, rw, rh);

            // Tactical label
            ctx.fillStyle = zone.borderColor || 'rgba(255, 255, 255, 0.7)';
            ctx.font = '9px JetBrains Mono';
            ctx.textAlign = 'left';
            ctx.fillText(zone.label, rx + 6, ry + 14);
        } else if (zone.x1 !== undefined) {
            // Road corridor / runway strip
            const x1 = cx + zone.x1 * zoom;
            const y1 = cy + zone.y1 * zoom;
            const x2 = cx + zone.x2 * zoom;
            const y2 = cy + zone.y2 * zoom;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = zone.color || 'rgba(0, 240, 255, 0.2)';
            ctx.lineWidth = zone.width * zoom;
            ctx.lineCap = 'square';
            ctx.stroke();

            // Centerline dashed guide
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = zone.borderColor || 'rgba(0, 240, 255, 0.7)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([8, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#00f0ff';
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

    data.tacticalObstacles.forEach((obs) => {
        const ox = cx + (obs.relX || 0) * zoom;
        const oy = cy + (obs.relY || 0) * zoom;
        const bw = 26 * zoom;
        const bh = 20 * zoom;

        // Tactical Bounding Box (AI Object Detection)
        ctx.strokeStyle = obs.threat === 'High' ? '#ef4444' : '#22c55e';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(ox - bw / 2, oy - bh / 2, bw, bh);

        // Class tag and dynamic live fluctuating percentage
        const confText = obs.conf > 1 ? obs.conf.toFixed(1) : (obs.conf * 100).toFixed(1);
        const tagText = `${obs.type} ${confText}%`;

        ctx.fillStyle = 'rgba(10, 14, 26, 0.90)';
        ctx.fillRect(ox - bw / 2, oy - bh / 2 - 14, bw + 34, 12);
        ctx.strokeStyle = obs.threat === 'High' ? '#ef444466' : '#22c55e66';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox - bw / 2, oy - bh / 2 - 14, bw + 34, 12);

        ctx.fillStyle = obs.threat === 'High' ? '#ff6b6b' : '#4ade80';
        ctx.font = '8px JetBrains Mono';
        ctx.textAlign = 'left';
        ctx.fillText(tagText, ox - bw / 2 + 3, oy - bh / 2 - 5);

        // Velocity vector
        if (obs.vx !== undefined && obs.vy !== undefined) {
            ctx.beginPath();
            ctx.moveTo(ox, oy);
            ctx.lineTo(ox + obs.vx * 25 * zoom, oy + obs.vy * 25 * zoom);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    });
}

function drawDroneFlightPath(drone) {
    const x0 = drone.x + state.canvas.offsetX;
    const y0 = drone.y + state.canvas.offsetY;
    const xt = drone.targetX + state.canvas.offsetX;
    const yt = drone.targetY + state.canvas.offsetY;

    // Direct Waypoint Flight Path (no K* dependency)
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(xt, yt);
    ctx.strokeStyle = drone.color + '80';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Target waypoint marker (Tactical Diamond - No Circles)
    ctx.save();
    ctx.translate(xt, yt);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = drone.color;
    ctx.fillRect(-4, -4, 8, 8);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.strokeRect(-4, -4, 8, 8);
    ctx.restore();

    // Label
    ctx.fillStyle = '#ffffff';
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText('WP-TARGET', xt + 8, yt + 3);
}

function drawAIHUD(w, h) {
    // Top-left AI Perception & GPU Telemetry HUD
    ctx.fillStyle = 'rgba(15, 21, 36, 0.90)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.fillRect(16, 16, 285, 80);
    ctx.strokeRect(16, 16, 285, 80);

    // Green square status badge (No Circles)
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(26, 26, 7, 7);

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText('AI PERCEPTION STACK: ACTIVE (ONNX)', 38, 33);

    const detLatency = window.aiLiveLatency || '9.4';
    const detLoss = window.aiLiveLoss || '2.56';
    const gpuUtil = window.aiLiveGpu || '96';

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px Inter';
    ctx.fillText(`Tactical Detector: ONNX FP16 (${detLatency} ms | Loss: ${detLoss})`, 26, 50);
    ctx.fillText(`Terrain Segmenter: Dubai Model (51.16% mIoU)`, 26, 65);
    ctx.fillText(`Hardware: RTX 5070 GPU (${gpuUtil}% Load | Single Drone)`, 26, 80);
}

// ─── Draw Range Rings (Disabled - No Circles) ───────────────────
function drawRangeRings() {
    // Deprecated / removed - keeping tactical map clean with zero circular rings
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

    // Draw base layers (STRICTLY NO CIRCLES)
    drawGrid();

    // ─── AI Perception & Kinematics Layers ───
    drawAITerrainCostmap();
    drawLandingZones();
    drawAITacticalObstacles(timestamp);

    // Draw active drones and flight path
    if (state.drones && state.drones.length > 0) {
        state.drones.forEach(drone => {
            drawDroneFlightPath(drone);
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
function setSimStatus(text) {
    const textEl = document.getElementById('sim-status-text');
    if (textEl) textEl.textContent = text;
}

document.getElementById('btn-play').addEventListener('click', () => {
    state.simulation.running = true;
    setSimStatus('Simulation Running');
});

document.getElementById('btn-pause').addEventListener('click', () => {
    state.simulation.running = false;
    setSimStatus('Simulation Paused');
});

document.getElementById('btn-stop').addEventListener('click', () => {
    state.simulation.running = false;
    state.simulation.time = 0;
    state.drones.forEach(d => { d.trail = []; });
    initDrones();
    setSimStatus('Simulation Idle');
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
        btn.textContent = '▶';
    } else {
        btn.textContent = '◀';
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

// CPU Drone Processing Cap Slider
document.getElementById('cpu-cores').addEventListener('input', (e) => {
    const cores = parseInt(e.target.value);
    state.memory.cpuCores = cores;
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
    const consText = document.getElementById('consumption-text');
    if (consText) consText.textContent = consumption + 'W';
    const consBar = document.getElementById('consumption-bar');
    if (consBar) {
        const pct = Math.min(100, Math.max(10, Math.round((consumption / 500) * 100)));
        consBar.style.width = pct + '%';
    }

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

    // Processing latencies update dynamically
}

// ─── Real-Time AI Perception Polling (Port 5001 / Port 3000 proxy) ────
async function fetchAIPerception() {
    try {
        let res = await fetch('/api/live-perception');
        if (!res.ok) {
            res = await fetch('http://127.0.0.1:5001/api/ai/live');
        }
        if (!res.ok) return;
        const data = await res.json();

        // 1. Live obstacles (moving bounding boxes with dynamic fluctuating confidence %)
        if (data.obstacles && Array.isArray(data.obstacles) && data.obstacles.length > 0) {
            if (!window.aiPerceptionData) window.aiPerceptionData = {};
            window.aiPerceptionData.tacticalObstacles = data.obstacles;
        }

        // 2. Hardware telemetry (RTX 5070 GPU Utilization % & VRAM)
        if (data.hardware) {
            const gpuPct = data.hardware.gpu_utilization_pct || 96;
            const vramMB = data.hardware.vram_used_mb || 7680;
            const totalMB = data.hardware.total_vram_mb || 8151;
            const vramGB = (vramMB / 1024).toFixed(1);
            const totalGB = (totalMB / 1024).toFixed(1);

            window.aiLiveGpu = gpuPct;

            const gpuUsageEl = document.getElementById('gpu-usage');
            if (gpuUsageEl) {
                gpuUsageEl.textContent = `${gpuPct}%`;
            }

            const aiHwTitle = document.getElementById('ai-hardware-title');
            if (aiHwTitle) {
                aiHwTitle.textContent = `● RTX 5070 GPU: ${gpuPct}% (Blackwell sm_120)`;
            }

            const aiHwVram = document.getElementById('ai-hardware-vram');
            if (aiHwVram) {
                aiHwVram.textContent = `CUDA 13.0 | PyTorch 2.14.0 | VRAM: ${vramGB} / ${totalGB} GB`;
            }
        }

        // 3. Live Model Metrics
        if (data.models) {
            if (data.models.tactical_detector) {
                const det = data.models.tactical_detector;
                window.aiLiveLatency = det.latency_ms;
                window.aiLiveLoss = det.current_loss;

                const latEl = document.getElementById('ai-onnx-latency');
                if (latEl) latEl.textContent = `${det.latency_ms} ms`;

                const lossEl = document.getElementById('ai-detector-loss');
                if (lossEl && det.current_loss) lossEl.textContent = `Loss ${det.current_loss}`;

                const recEl = document.getElementById('ai-detector-recall');
                if (recEl) recEl.textContent = `${det.recall_pct}%`;
            }

            if (data.models.terrain_segmenter) {
                const seg = data.models.terrain_segmenter;
                const miouEl = document.getElementById('ai-terrain-miou');
                if (miouEl) miouEl.textContent = `${seg.miou_pct}%`;

                const corEl = document.getElementById('ai-safe-corridor');
                if (corEl && seg.safe_corridor_score) corEl.textContent = `${seg.safe_corridor_score}%`;
            }
        }
    } catch (err) {
        // AI perception offline fallback
    }
}

// ─── Real Host System Metrics ───────────
function updateCpuCoresGrid(perCore) {
    const grid = document.getElementById('cpu-cores-grid');
    if (!grid) return;

    // Initialize or rebuild if core count changed
    if (grid.children.length !== perCore.length) {
        grid.innerHTML = '';
        perCore.forEach(c => {
            const coreEl = document.createElement('div');
            coreEl.className = 'cpu-core active';
            coreEl.id = `host-core-${c.core}`;
            coreEl.innerHTML = `
                <div class="core-bar"><div class="core-fill" style="height:${c.usage}%"></div></div>
                <span class="core-label">C${c.core}</span>
                <span class="core-pct">${c.usage}%</span>
            `;
            grid.appendChild(coreEl);
        });
        return;
    }

    // Update live metrics for each core
    perCore.forEach(c => {
        const coreEl = document.getElementById(`host-core-${c.core}`);
        if (!coreEl) return;
        const fill = coreEl.querySelector('.core-fill');
        const pct = coreEl.querySelector('.core-pct');
        if (fill) fill.style.height = `${c.usage}%`;
        if (pct) pct.textContent = `${c.usage}%`;
        if (c.usage > 75) {
            coreEl.classList.add('high-load');
        } else {
            coreEl.classList.remove('high-load');
        }
    });
}

function renderFallbackCores() {
    const grid = document.getElementById('cpu-cores-grid');
    if (!grid || grid.children.length > 0) return;
    const mockCores = [
        { core: 1, usage: 42 }, { core: 2, usage: 55 }, { core: 3, usage: 38 }, { core: 4, usage: 61 },
        { core: 5, usage: 22 }, { core: 6, usage: 18 }, { core: 7, usage: 30 }, { core: 8, usage: 12 }
    ];
    updateCpuCoresGrid(mockCores);
}

async function fetchHostMetrics() {
    try {
        const res = await fetch('/api/system-metrics');
        if (!res.ok) throw new Error('API unavailable');
        const data = await res.json();

        // Model name
        const modelChip = document.getElementById('cpu-model-chip');
        if (modelChip && data.model) {
            const cleanModel = data.model.replace(/\(R\)|\(TM\)/g, '').trim();
            modelChip.textContent = `CPU: ${cleanModel}`;
            modelChip.title = data.model;
        }

        // Total load
        const loadChip = document.getElementById('cpu-load-chip');
        if (loadChip) {
            loadChip.textContent = `Total Load: ${data.overallUsage}% (${data.coreCount} Cores)`;
        }

        const cpuDisplay = document.getElementById('cpu-display');
        if (cpuDisplay) {
            cpuDisplay.textContent = `${data.overallUsage}%`;
        }

        // Host memory in status bar
        const memUsage = document.getElementById('mem-usage');
        if (memUsage && data.memory) {
            const usedGB = (data.memory.usedMB / 1024).toFixed(1);
            const totalGB = (data.memory.totalMB / 1024).toFixed(1);
            memUsage.textContent = `${usedGB} / ${totalGB} GB (${data.memory.usedPercent}%)`;
        }

        // Per-core telemetry
        if (data.perCore && data.perCore.length > 0) {
            updateCpuCoresGrid(data.perCore);
        }
    } catch (err) {
        renderFallbackCores();
    }
}

// ─── Initialize ─────────────────────────
function init() {
    resizeCanvas();
    initDrones();
    initSparklines();

    // Start render loop
    requestAnimationFrame(render);

    // Initial and periodic host hardware metrics fetch
    fetchHostMetrics();
    setInterval(fetchHostMetrics, 1000);

    // Initial and periodic AI perception telemetry fetch (Live dynamic percentages)
    fetchAIPerception();
    setInterval(fetchAIPerception, 800);

    // Start simulation latency metrics update
    setInterval(updateMetrics, 1000);

    // Initial landing zones update
    updateLandingZones(12.4);
}

// Start
window.addEventListener('DOMContentLoaded', init);
