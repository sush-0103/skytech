/**
 * IN-SILICO UAV VALIDATION ENGINE - 3D MISSION & TELEMETRY CONTROLLER
 * Renders worlds/dubai_osm/dubai_2_5d_visual.glb (exported from dubai_2_5d.blend),
 * real SITL flight route, 3D Vehicle Twin, AI perception HUD, and latency monitoring.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// =============================================================================
// GLOBAL STATE
// =============================================================================
const state = {
  // SITL Flight Trace Telemetry
  traceData: null,
  traceIndex: 0,
  isPlaying: true,
  playSpeed: 2,
  totalPoints: 501,

  // Mission Viewport (3D Dubai World)
  missionContainer: null,
  missionCanvas: null,
  missionScene: null,
  missionCamera: null,
  missionRenderer: null,
  missionControls: null,
  missionWorldGroup: null,
  missionDroneGroup: null,
  missionDroneModel: null,
  missionDroneProps: [],
  routeLine: null,

  // Synchronized simulated sensor renderers (same pose, separate representations)
  perceptionCanvas: null,
  perceptionRenderer: null,
  perceptionCamera: null,
  semanticCanvas: null,
  semanticRenderer: null,
  semanticCamera: null,
  semanticScene: null,
  semanticWorldGroup: null,
  semanticFrame: 0,
  lastSensorRenderMs: 0,
  sensorPose: null,

  // Vehicle Twin Viewport (Close-up 3D CAD Twin)
  twinContainer: null,
  twinCanvas: null,
  twinScene: null,
  twinCamera: null,
  twinRenderer: null,
  twinControls: null,
  twinDroneModel: null,
  twinPropellers: [],
  twinMixer: null,
  twinHoverAction: null,

  // Scenario Progress & Dynamic Mission Controls
  realFlightSpeed: 4.2,
  cruiseAltitude: 6.0,
  activeWaypointIndex: 0,

  // AI Telemetry: Air Traffic, Compute, and Battery
  trafficRadiusKm: 3.0,
  trafficDensityLevel: 'medium',
  airTrafficConflict: false,
  batterySocPct: 82,
  isDivertEngaged: false,
  trafficDome3D: null,
  emergency3DGroup: null,

  // Latency Chart
  latCanvas: null,
  latCtx: null,

  // Camera HUD & Reactive Obstacle Avoidance
  cameraCanvas: null,
  cameraCtx: null,
  avoidanceMode: 'auto',
  reactiveData: null,
  actualReactiveDecision: null,
  buildingColliders: [],
  collisionBuildings: [],
  actualObstacleHelper: null,
  avoidanceOffsetNed: { north: 0, east: 0 },
  avoidanceAltitudeBoost: 0,
  lastSafeFlightProgress: 0,
  lastSafePositionNed: null,
  collisionGuardTriggered: false,
  obstacleBox3D: null,
  evasionArc3D: null,

  // Checkpoints & Flight Path (Path A to Path B)
  startNed: [-360.0, -400.0, -6.0],
  destination: { name: 'Alpha (Northeast Hub)', ned: [300.0, 400.0, -6.0] },
  pendingTarget: null,
  pendingTargetBeacon3D: null,
  pendingDirectLine3D: null,
  flightProgress: 0.0,
  isHoveringAtGoal: false,
  totalRouteDistanceM: 1024.0,
  isPickingDestination: false,
  startBeacon3D: null,
  goalBeacon3D: null,
  directLine3D: null,
  currentRouteWaypoints: [],

  // AI & Filter
  aiData: null,
  eventFilter: 'all',

  // Map Display Modes, Layers & Camera Perspective
  mapMode: 'satellite', // 'standard' | 'satellite' | 'blender' | 'tactical'
  cameraPerspective: '3d', // '3d' | '2d'
  layers: {
    buildings: true,
    restricted: true,
    route: true,
    directLine: true,
    trafficDome: true,
    labels: true
  },
  buildingMeshes: [],
  roadMeshes: [],
  terrainMeshes: [],
  restrictedMeshes: [],
  standardGroundMesh: null,
  satelliteGroundMesh: null,
  mapLabelsGroup: null,
  ambientMissionLight: null,
  sunMissionLight: null,
  droneScale: 1.0
};
window.state = state;



// Planned waypoints from 3D Kinematic A* Planner in local NED (North x, East y, Down z)
const PLANNED_WAYPOINTS_NED = [
  [-360.0, -400.0, -6.0],
  [-351.1, -391.73, -6.0],
  [-319.1, -359.73, -6.0],
  [-319.1, -351.73, -6.0],
  [-311.1, -351.73, -6.0],
  [-311.1, -343.73, -6.0],
  [-303.1, -335.73, -6.0],
  [-303.1, -327.73, -6.0],
  [-287.1, -311.73, -6.0],
  [-279.1, -311.73, -6.0],
  [-255.1, -287.73, -6.0],
  [-255.1, -279.73, -6.0],
  [-247.1, -271.73, -6.0],
  [-247.1, -263.73, -6.0],
  [-239.1, -255.73, -6.0],
  [-239.1, -231.73, -6.0],
  [-231.1, -223.73, -6.0],
  [-231.1, -215.73, -6.0],
  [-223.1, -207.73, -6.0],
  [-223.1, -199.73, -6.0],
  [-55.1, -199.73, -6.0],
  [-47.1, -207.73, -6.0],
  [-15.1, -207.73, -6.0],
  [8.9, -231.73, -6.0],
  [32.9, -231.73, -6.0],
  [40.9, -223.73, -6.0],
  [48.9, -223.73, -6.0],
  [48.9, -215.73, -6.0],
  [56.9, -207.73, -6.0],
  [56.9, -199.73, -6.0],
  [64.9, -191.73, -6.0],
  [64.9, -183.73, -6.0],
  [88.9, -159.73, -6.0],
  [88.9, -119.73, -6.0],
  [112.9, -95.73, -6.0],
  [112.9, -31.73, -6.0],
  [288.9, 144.27, -6.0],
  [288.9, 392.27, -6.0],
  [300.0, 400.0, -6.0]
];

/**
 * Converts NED coordinates (North x, East y, Down z) to glTF 3D coordinates (Y-up)
 * glTF X = East (y)
 * glTF Y = Altitude (-z)
 * glTF Z = -North (-x)
 */
function nedToGltf(x, y, z) {
  return new THREE.Vector3(y, -z, -x);
}

// =============================================================================
// 1. DATA LOADERS
// =============================================================================
async function loadFlightTrace() {
  try {
    const res = await fetch('/api/flight-trace');
    if (res.ok) {
      const data = await res.json();
      if (data && data.points && data.points.length > 0) {
        state.traceData = data.points;
        state.totalPoints = data.points.length;
        const scrubber = document.getElementById('trace-scrubber');
        if (scrubber) scrubber.max = state.totalPoints - 1;
        return;
      }
    }
  } catch (e) {
    console.warn('Trace load error, using synthetic telemetry', e);
  }
  state.traceData = generateSyntheticTrajectory();
  state.totalPoints = state.traceData.length;
}

async function loadLivePerception() {
  try {
    const res = await fetch('/api/live-perception');
    if (res.ok) {
      state.aiData = await res.json();
      if (state.aiData && state.aiData.reactive_avoidance) {
        state.reactiveData = state.aiData.reactive_avoidance;
      }
      updatePerceptionHUD(state.aiData);
      updateReactivePerceptionHUD();
      updateBackendTelemetryHUD(state.aiData);
    }
  } catch (e) {}
}


function generateSyntheticTrajectory() {
  const pts = [];
  const wps = PLANNED_WAYPOINTS_NED;
  const segments = wps.length - 1;
  const stepsPerSeg = Math.floor(500 / segments);

  for (let s = 0; s < segments; s++) {
    const p1 = wps[s];
    const p2 = wps[s + 1];
    for (let step = 0; step < stepsPerSeg; step++) {
      const alpha = step / stepsPerSeg;
      const x = p1[0] + (p2[0] - p1[0]) * alpha;
      const y = p1[1] + (p2[1] - p1[1]) * alpha;
      const z = -6.0;
      pts.push({
        t: pts.length * 0.95,
        position: [x, y, z],
        velocity: [(p2[0] - p1[0]) * 0.05, (p2[1] - p1[1]) * 0.05, 0.0],
        cross_track_m: 0.15 + Math.random() * 0.1
      });
    }
  }
  return pts;
}

// =============================================================================
// 2. MISSION PLAN 3D VIEWPORT (dubai_2_5d_visual.glb)
// =============================================================================
function initMission3DViewport() {
  state.missionContainer = document.getElementById('mission-viewport');
  state.missionCanvas = document.getElementById('mission-canvas');
  if (!state.missionContainer || !state.missionCanvas) return;

  const w = state.missionContainer.clientWidth || 600;
  const h = state.missionContainer.clientHeight || 400;

  // 1. Scene & Camera
  state.missionScene = new THREE.Scene();
  state.missionScene.background = new THREE.Color(0x080d19);
  state.missionScene.fog = new THREE.FogExp2(0x0b1323, 0.00018);

  state.missionCamera = new THREE.PerspectiveCamera(42, w / h, 1, 10000);
  // Isometric perspective looking southwest-to-northeast across Dubai
  state.missionCamera.position.set(-650, 780, 750);
  state.missionCamera.lookAt(0, 0, 0);

  // 2. Renderer
  state.missionRenderer = new THREE.WebGLRenderer({
    canvas: state.missionCanvas,
    antialias: true,
    powerPreference: 'high-performance'
  });
  state.missionRenderer.setSize(w, h);
  state.missionRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  state.missionRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  state.missionRenderer.toneMappingExposure = 1.0;
  state.missionRenderer.outputColorSpace = THREE.SRGBColorSpace;
  state.missionRenderer.shadowMap.enabled = true;
  state.missionRenderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // 3. OrbitControls
  state.missionControls = new OrbitControls(state.missionCamera, state.missionCanvas);
  state.missionControls.target.set(0, 0, 0);
  state.missionControls.enableDamping = true;
  state.missionControls.dampingFactor = 0.06;
  state.missionControls.maxPolarAngle = Math.PI / 2.05; // Do not go below ground
  state.missionControls.minDistance = 50;
  state.missionControls.maxDistance = 3500;
  state.missionControls.update();

  // 4. Studio Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.8);
  state.missionScene.add(ambient);
  state.ambientMissionLight = ambient;

  const sunLight = new THREE.DirectionalLight(0xfffbeb, 1.5);
  sunLight.position.set(400, 900, 500);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(2048, 2048);
  sunLight.shadow.camera.left = -650;
  sunLight.shadow.camera.right = 650;
  sunLight.shadow.camera.top = 520;
  sunLight.shadow.camera.bottom = -520;
  state.missionScene.add(sunLight);
  state.sunMissionLight = sunLight;

  const skyHemisphere = new THREE.HemisphereLight(0x9fd5ef, 0x4b3b2b, 0.55);
  state.missionScene.add(skyHemisphere);

  initSynchronizedSensorRenderers();
  loadExactCollisionGeometry();

  // Procedural Ground Planes (Real Standard Map & Satellite)
  buildProceduralGroundPlanes();

  // 3D Landmark & Street Labels
  buildMapLabels();

  // 5. Load dubai_2_5d_visual.glb (from worlds/dubai_osm/dubai_2_5d.blend)
  loadDubaiGlbWorld();

  // 6. Draw 3D Flight Route & Start/Goal Beacons
  build3DFlightPath();

  // 7. Add 3D Quadrotor in City
  buildMissionDrone();

  // Real obstacle highlighting is created from loaded building colliders.

  window.addEventListener('resize', onMissionResize);

}

function onMissionResize() {
  if (!state.missionContainer || !state.missionCamera || !state.missionRenderer) return;
  const w = state.missionContainer.clientWidth || 600;
  const h = state.missionContainer.clientHeight || 400;
  state.missionCamera.aspect = w / h;
  state.missionCamera.updateProjectionMatrix();
  state.missionRenderer.setSize(w, h);
  resizeSynchronizedSensorRenderers();
}

function initSynchronizedSensorRenderers() {
  state.perceptionCanvas = document.getElementById('camera-scene-canvas');
  state.semanticCanvas = document.getElementById('semantic-canvas');
  if (!state.perceptionCanvas || !state.semanticCanvas) return;

  const cameraBox = state.perceptionCanvas.parentElement;
  const semanticBox = state.semanticCanvas.parentElement;
  const pw = cameraBox.clientWidth || 420;
  const ph = cameraBox.clientHeight || 170;
  const sw = semanticBox.clientWidth || 420;
  const sh = semanticBox.clientHeight || 140;

  state.perceptionRenderer = new THREE.WebGLRenderer({ canvas: state.perceptionCanvas, antialias: true });
  state.perceptionRenderer.setSize(pw, ph, false);
  state.perceptionRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  state.perceptionRenderer.outputColorSpace = THREE.SRGBColorSpace;
  state.perceptionRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  state.perceptionRenderer.toneMappingExposure = 1.15;
  state.perceptionCamera = new THREE.PerspectiveCamera(64, pw / ph, 0.25, 1400);

  state.semanticScene = new THREE.Scene();
  state.semanticScene.background = new THREE.Color(0x58b8e8);
  state.semanticRenderer = new THREE.WebGLRenderer({ canvas: state.semanticCanvas, antialias: false });
  state.semanticRenderer.setSize(sw, sh, false);
  state.semanticRenderer.setPixelRatio(1);
  state.semanticRenderer.outputColorSpace = THREE.SRGBColorSpace;
  state.semanticCamera = new THREE.PerspectiveCamera(64, sw / sh, 0.25, 1400);

  // Add initial class-labelled base plane (Vegetation/Terrain: #4fbf72)
  const base = new THREE.Mesh(
    new THREE.PlaneGeometry(1024, 814),
    new THREE.MeshBasicMaterial({ color: 0x4fbf72, side: THREE.DoubleSide })
  );
  base.rotation.x = -Math.PI / 2;
  base.position.y = -0.08;
  state.semanticScene.add(base);

  // Initialize sensorPose immediately using startNed so sensors render from frame 1
  const startPos = nedToGltf(state.startNed[0], state.startNed[1], state.startNed[2]);
  state.sensorPose = {
    position: startPos.clone(),
    direction: new THREE.Vector3(1, 0, 0),
    north: state.startNed[0],
    east: state.startNed[1],
    altitude: -state.startNed[2],
    headingDeg: 90
  };

  window.addEventListener('resize', resizeSynchronizedSensorRenderers);
}

function resizeSynchronizedSensorRenderers() {
  if (state.perceptionRenderer && state.perceptionCanvas?.parentElement) {
    const box = state.perceptionCanvas.parentElement;
    const w = box.clientWidth || 420;
    const h = box.clientHeight || 170;
    state.perceptionRenderer.setSize(w, h, false);
    state.perceptionCamera.aspect = w / h;
    state.perceptionCamera.updateProjectionMatrix();
  }
  if (state.semanticRenderer && state.semanticCanvas?.parentElement) {
    const box = state.semanticCanvas.parentElement;
    const w = box.clientWidth || 420;
    const h = box.clientHeight || 110;
    state.semanticRenderer.setSize(w, h, false);
    state.semanticCamera.aspect = w / h;
    state.semanticCamera.updateProjectionMatrix();
  }
  resizeCameraHUDCanvas();
}

function loadDubaiGlbWorld() {
  const loader = new GLTFLoader();
  const url = '/worlds/dubai_osm/dubai_2_5d_visual.glb';

  loader.load(
    url,
    (gltf) => {
      state.missionWorldGroup = gltf.scene;

      state.buildingMeshes = [];
      state.roadMeshes = [];
      state.terrainMeshes = [];
      state.restrictedMeshes = [];

      // Categorize meshes for instantaneous mode/layer swapping
      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          const name = (child.name || '').toLowerCase();
          child.castShadow = true;
          child.receiveShadow = true;
          
          if (name.includes('smoke') || name.includes('route') || name.includes('flight') || name.includes('trace') || name.includes('path') || name.includes('curve')) {
            child.visible = false;
          } else if (name.includes('restricted') || name.includes('demo_restricted')) {
            state.restrictedMeshes.push(child);
          } else if (name.includes('road')) {
            state.roadMeshes.push(child);
          } else if (name.includes('terrain') || name.includes('flat')) {
            state.terrainMeshes.push(child);
          } else {
            state.buildingMeshes.push(child);
          }
        }
      });

      state.missionScene.add(state.missionWorldGroup);
      gltf.scene.updateMatrixWorld(true);
      state.buildingColliders = state.buildingMeshes.map((mesh, index) => ({
        id: mesh.name || `BUILDING-MESH-${index + 1}`,
        mesh,
        box: new THREE.Box3().setFromObject(mesh)
      })).filter(item => !item.box.isEmpty());
      linkCollisionGeometryToMeshes();
      buildSemanticWorld(gltf.scene);
      addArchitecturalEdges();
      console.log(`Loaded 3D Dubai OSM World: ${state.buildingMeshes.length} buildings, ${state.roadMeshes.length} roads.`);

      // Apply initial selected map mode
      applyMapMode(state.mapMode || 'standard');
    },
    undefined,
    (err) => {
      console.warn('Failed to load dubai_2_5d_visual.glb, building fallback geometry:', err);
      buildFallbackGeometry();
    }
  );
}

function semanticMaterialForName(name) {
  const n = (name || '').toLowerCase();
  let color = 0xe39a43;
  if (n.includes('restricted') || n.includes('demo_restricted')) color = 0xef4444;
  else if (n.includes('road')) color = 0x7057d9;
  else if (n.includes('terrain') || n.includes('flat')) color = 0x4fbf72;
  return new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
}

function buildSemanticWorld(sourceWorld) {
  if (!state.semanticScene || !sourceWorld) return;
  if (state.semanticWorldGroup) state.semanticScene.remove(state.semanticWorldGroup);
  const semanticWorld = sourceWorld.clone(true);
  semanticWorld.traverse((child) => {
    if (!child.isMesh) return;
    const name = (child.name || '').toLowerCase();
    if (name.includes('smoke') || name.includes('route') || name.includes('flight') || name.includes('trace') || name.includes('path') || name.includes('curve')) {
      child.visible = false;
      return;
    }
    child.material = semanticMaterialForName(child.name);
    child.castShadow = false;
    child.receiveShadow = false;
    child.visible = true;
  });
  state.semanticWorldGroup = semanticWorld;
  state.semanticScene.add(semanticWorld);
  semanticWorld.updateMatrixWorld(true);

  // A class-labelled base guarantees a stable ground label even if a display mode hides the GLB terrain.
  const base = new THREE.Mesh(
    new THREE.PlaneGeometry(1024, 814),
    new THREE.MeshBasicMaterial({ color: 0x4fbf72, side: THREE.DoubleSide })
  );
  base.rotation.x = -Math.PI / 2;
  base.position.y = -0.08;
  state.semanticScene.add(base);
}

function addArchitecturalEdges() {
  if (!state.missionWorldGroup || state.missionWorldGroup.getObjectByName('ProfessionalBuildingEdges')) return;
  const edgeGroup = new THREE.Group();
  edgeGroup.name = 'ProfessionalBuildingEdges';
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.28 });
  state.buildingMeshes.forEach((mesh) => {
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 28), edgeMaterial);
    edges.applyMatrix4(mesh.matrixWorld);
    edgeGroup.add(edges);
  });
  state.missionScene.add(edgeGroup);
}

function buildFallbackGeometry() {
  // Ground
  const groundGeo = new THREE.PlaneGeometry(1024, 814);
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x141f36, roughness: 0.9 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  state.terrainMeshes = [ground];
  state.missionScene.add(ground);

  // Restricted Zone Box
  const rzGeo = new THREE.BoxGeometry(140, 20, 170);
  const rzMat = new THREE.MeshStandardMaterial({ color: 0xef4444, transparent: true, opacity: 0.45 });
  const rz = new THREE.Mesh(rzGeo, rzMat);
  rz.position.set(0, 10, 0);
  state.restrictedMeshes = [rz];
  state.missionScene.add(rz);

  applyMapMode(state.mapMode || 'standard');
}

// =============================================================================
// CARTOGRAPHIC MAP TEXTURE GENERATORS & DISPLAY MODES
// =============================================================================

function createStandardCartographicTexture() {
  const cvs = document.createElement('canvas');
  cvs.width = 2048;
  cvs.height = 1628;
  const ctx = cvs.getContext('2d');

  function toCvs(x, z) {
    const u = (x + 512) / 1024;
    const v = (z + 407) / 814;
    return [u * 2048, v * 1628];
  }

  // 1. Base OSM Parchment Land (#f1efe8)
  ctx.fillStyle = '#f1efe8';
  ctx.fillRect(0, 0, 2048, 1628);

  // 2. City Blocks / Urban Parcels Grid
  ctx.fillStyle = '#e8e5dc';
  ctx.strokeStyle = '#ded9cf';
  ctx.lineWidth = 2;
  for (let bx = -480; bx < 480; bx += 90) {
    for (let bz = -380; bz < 380; bz += 80) {
      const [px, py] = toCvs(bx, bz);
      ctx.fillRect(px + 4, py + 4, 160, 140);
      ctx.strokeRect(px + 4, py + 4, 160, 140);
    }
  }

  // 3. Green Parks & Natural Landscapes
  // Park 1: Al Safa Park Corridor (Southwest)
  ctx.fillStyle = '#cbe6a3';
  ctx.strokeStyle = '#a3d977';
  ctx.lineWidth = 3;
  let [p1x, p1y] = toCvs(-440, 140);
  ctx.beginPath();
  ctx.rect(p1x, p1y, 480, 420);
  ctx.fill();
  ctx.stroke();

  // Park 2: Northeast Botanical Gardens & Green Belt
  ctx.fillStyle = '#d4ecc5';
  ctx.strokeStyle = '#b8e2a4';
  let [p2x, p2y] = toCvs(160, -360);
  ctx.beginPath();
  ctx.rect(p2x, p2y, 460, 320);
  ctx.fill();
  ctx.stroke();

  // 4. Dubai Water Canal / Creek Corridor (Serene Blue)
  ctx.strokeStyle = '#93c5fd';
  ctx.lineWidth = 76;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let [w1x, w1y] = toCvs(-512, 60);
  let [w2x, w2y] = toCvs(-150, 10);
  let [w3x, w3y] = toCvs(180, -90);
  let [w4x, w4y] = toCvs(512, -140);
  ctx.moveTo(w1x, w1y);
  ctx.bezierCurveTo(w2x, w2y, w3x, w3y, w4x, w4y);
  ctx.stroke();

  // Canal inner water depth line
  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 54;
  ctx.stroke();

  // Canal shoreline borders
  ctx.strokeStyle = '#bfdbfe';
  ctx.lineWidth = 4;
  ctx.stroke();

  // 5. Secondary Arterial Boulevards (Financial Centre Rd, Al Safa St, Al Wasl Rd)
  const drawArterial = (pStart, pEnd, width, casingColor, fillColor) => {
    const [sx, sy] = toCvs(pStart[0], pStart[1]);
    const [ex, ey] = toCvs(pEnd[0], pEnd[1]);

    // Outer casing
    ctx.strokeStyle = casingColor;
    ctx.lineWidth = width + 6;
    ctx.lineCap = 'square';
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();

    // Inner fill
    ctx.strokeStyle = fillColor;
    ctx.lineWidth = width;
    ctx.stroke();
  };

  // Financial Centre Rd (East-West arterial)
  drawArterial([-512, -120], [512, -60], 22, '#94a3b8', '#ffffff');
  // Al Safa Street (Transverse commercial strip)
  drawArterial([-420, 220], [420, 280], 20, '#94a3b8', '#ffffff');
  // Al Wasl Road D92 (Coastal arterial)
  drawArterial([-260, -407], [-210, 407], 20, '#cbd5e1', '#ffffff');
  // Boulevard Cross-link
  drawArterial([20, -407], [80, 407], 16, '#cbd5e1', '#ffffff');

  // Local streets grid network
  for (let sx = -420; sx < 420; sx += 70) {
    drawArterial([sx, -360], [sx + 40, 360], 8, '#e2e8f0', '#ffffff');
  }
  for (let sz = -320; sz < 320; sz += 60) {
    drawArterial([-450, sz], [450, sz + 30], 8, '#e2e8f0', '#ffffff');
  }

  // 6. Primary Highway: Sheikh Zayed Road (E11)
  // Runs through Dubai with dual carriageway in classic OpenStreetMap amber gold
  const [h1x, h1y] = toCvs(-40, -407);
  const [h2x, h2y] = toCvs(20, 407);

  // Outer highway casing (amber brown)
  ctx.strokeStyle = '#d97706';
  ctx.lineWidth = 44;
  ctx.lineCap = 'square';
  ctx.beginPath();
  ctx.moveTo(h1x, h1y);
  ctx.lineTo(h2x, h2y);
  ctx.stroke();

  // Highway surface (OSM golden amber)
  ctx.strokeStyle = '#fbbf24';
  ctx.lineWidth = 36;
  ctx.stroke();

  // Center median divider
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 4;
  ctx.stroke();

  // Dashed white lane markings on both sides
  ctx.setLineDash([14, 14]);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;

  const offset = 10;
  ctx.beginPath();
  ctx.moveTo(h1x - offset, h1y);
  ctx.lineTo(h2x - offset, h2y);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(h1x + offset, h1y);
  ctx.lineTo(h2x + offset, h2y);
  ctx.stroke();
  ctx.setLineDash([]);

  // 7. Cartographic Street Typography & Map Labels (OpenStreetMap Style)
  const drawStreetLabel = (text, x, z, angle, size, isHighway) => {
    const [lx, ly] = toCvs(x, z);
    ctx.save();
    ctx.translate(lx, ly);
    ctx.rotate(angle);
    ctx.font = `bold ${size}px "Inter", -apple-system, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.strokeText(text, 0, 0);

    ctx.fillStyle = isHighway ? '#92400e' : '#334155';
    ctx.fillText(text, 0, 0);
    ctx.restore();
  };

  const szrAngle = Math.atan2(h2y - h1y, h2x - h1x) - Math.PI / 2;
  drawStreetLabel('SHEIKH ZAYED RD (E11)', -10, -220, szrAngle, 18, true);
  drawStreetLabel('SHEIKH ZAYED RD (E11)', 0, 180, szrAngle, 18, true);
  drawStreetLabel('FINANCIAL CENTRE RD', 0, -105, 0.05, 14, false);
  drawStreetLabel('AL SAFA STREET', -30, 240, 0.06, 14, false);
  drawStreetLabel('AL WASL RD (D92)', -235, 20, 0.08, 13, false);

  const drawAreaLabel = (text, sub, x, z, color) => {
    const [lx, ly] = toCvs(x, z);
    ctx.save();
    ctx.translate(lx, ly);
    ctx.font = 'bold 15px "Inter", sans-serif';
    ctx.textAlign = 'center';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.strokeText(text, 0, -4);
    ctx.fillStyle = color;
    ctx.fillText(text, 0, -4);

    if (sub) {
      ctx.font = '11px "Inter", sans-serif';
      ctx.fillStyle = '#64748b';
      ctx.strokeText(sub, 0, 12);
      ctx.fillText(sub, 0, 12);
    }
    ctx.restore();
  };

  drawAreaLabel('AL SAFA PARK', 'PUBLIC GREEN CORRIDOR', -280, 260, '#2d6a4f');
  drawAreaLabel('DUBAI WATER CANAL', 'MARITIME PASSAGE', -20, -30, '#1d4ed8');
  drawAreaLabel('BUSINESS BAY DISTRICT', 'COMMERCIAL TOWER ZONE', 260, -180, '#475569');

  const tex = new THREE.CanvasTexture(cvs);
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 8;
  return tex;
}

function createSatelliteTexture() {
  const cvs = document.createElement('canvas');
  cvs.width = 1024;
  cvs.height = 814;
  const ctx = cvs.getContext('2d');

  function toCvs(x, z) {
    const u = (x + 512) / 1024;
    const v = (z + 407) / 814;
    return [u * 1024, v * 814];
  }

  // Desert satellite earth base
  ctx.fillStyle = '#c7b99c';
  ctx.fillRect(0, 0, 1024, 814);

  // Subtle sand dunes texture
  for (let i = 0; i < 70; i++) {
    const gx = Math.random() * 1024;
    const gy = Math.random() * 814;
    const gw = 40 + Math.random() * 90;
    const gh = 20 + Math.random() * 40;
    ctx.fillStyle = i % 2 === 0 ? 'rgba(180, 160, 130, 0.35)' : 'rgba(215, 200, 175, 0.3)';
    ctx.fillRect(gx, gy, gw, gh);
  }

  // Deep Canal
  ctx.strokeStyle = '#1e3a5f';
  ctx.lineWidth = 36;
  ctx.lineCap = 'round';
  ctx.beginPath();
  let [w1x, w1y] = toCvs(-512, 60);
  let [w2x, w2y] = toCvs(-150, 10);
  let [w3x, w3y] = toCvs(180, -90);
  let [w4x, w4y] = toCvs(512, -140);
  ctx.moveTo(w1x, w1y);
  ctx.bezierCurveTo(w2x, w2y, w3x, w3y, w4x, w4y);
  ctx.stroke();

  // Dark asphalt highways & roads
  const [h1x, h1y] = toCvs(-40, -407);
  const [h2x, h2y] = toCvs(20, 407);
  ctx.strokeStyle = '#272b30';
  ctx.lineWidth = 22;
  ctx.beginPath();
  ctx.moveTo(h1x, h1y);
  ctx.lineTo(h2x, h2y);
  ctx.stroke();

  // Secondary roads
  const [f1x, f1y] = toCvs(-512, -120);
  const [f2x, f2y] = toCvs(512, -60);
  ctx.strokeStyle = '#33373d';
  ctx.lineWidth = 14;
  ctx.beginPath();
  ctx.moveTo(f1x, f1y);
  ctx.lineTo(f2x, f2y);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(cvs);
  tex.anisotropy = 4;
  return tex;
}

function buildProceduralGroundPlanes() {
  // 1. Real Standard Mode Ground Plane
  const standardTex = createStandardCartographicTexture();
  const standardGeo = new THREE.PlaneGeometry(1024, 814);
  const standardMat = new THREE.MeshStandardMaterial({
    map: standardTex,
    roughness: 0.85,
    metalness: 0.05
  });
  state.standardGroundMesh = new THREE.Mesh(standardGeo, standardMat);
  state.standardGroundMesh.rotation.x = -Math.PI / 2;
  state.standardGroundMesh.position.set(0, 0.1, 0);
  state.standardGroundMesh.visible = (state.mapMode === 'standard');
  state.missionScene.add(state.standardGroundMesh);

  // 2. Aerial Satellite Mode Ground Plane
  const satelliteTex = createSatelliteTexture();
  const satelliteGeo = new THREE.PlaneGeometry(1024, 814);
  const satelliteMat = new THREE.MeshStandardMaterial({
    map: satelliteTex,
    roughness: 0.8,
    metalness: 0.1
  });
  state.satelliteGroundMesh = new THREE.Mesh(satelliteGeo, satelliteMat);
  state.satelliteGroundMesh.rotation.x = -Math.PI / 2;
  state.satelliteGroundMesh.position.set(0, 0.12, 0);
  state.satelliteGroundMesh.visible = (state.mapMode === 'satellite');
  state.missionScene.add(state.satelliteGroundMesh);
}

function createMapLabelSprite(text, subText, colorHex) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');

  // Background tag (Strictly zero circles: rectilinear tag)
  ctx.fillStyle = 'rgba(11, 18, 32, 0.88)';
  ctx.fillRect(4, 4, 248, 56);
  ctx.strokeStyle = colorHex || '#38bdf8';
  ctx.lineWidth = 2;
  ctx.strokeRect(4, 4, 248, 56);

  // Diamond indicator
  ctx.save();
  ctx.translate(20, 32);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = colorHex || '#38bdf8';
  ctx.fillRect(-4, -4, 8, 8);
  ctx.restore();

  // Primary Text
  ctx.font = 'bold 15px "JetBrains Mono", monospace';
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'left';
  ctx.fillText(text, 34, 27);

  // Subtitle
  if (subText) {
    ctx.font = '10px "Inter", sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(subText, 34, 45);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(65, 16.25, 1);
  return sprite;
}

function buildMapLabels() {
  state.mapLabelsGroup = new THREE.Group();

  const labelsData = [
    { text: 'SHEIKH ZAYED RD', sub: 'E11 HIGHWAY CORRIDOR', pos: [0, 22, 0], color: '#f59e0b' },
    { text: 'BUSINESS BAY', sub: 'COMMERCIAL HUB', pos: [260, 22, -140], color: '#38bdf8' },
    { text: 'AL SAFA PARK', sub: 'PUBLIC GREEN BUFFER', pos: [-300, 22, 240], color: '#10b981' },
    { text: 'DUBAI WATER CANAL', sub: 'MARITIME PASSAGE', pos: [-60, 22, -60], color: '#60a5fa' },
    { text: 'FINANCIAL CENTRE', sub: 'DIFC ARTERIAL BOULEVARD', pos: [140, 22, 210], color: '#a855f7' }
  ];

  labelsData.forEach(item => {
    const sprite = createMapLabelSprite(item.text, item.sub, item.color);
    sprite.position.set(item.pos[0], item.pos[1], item.pos[2]);
    state.mapLabelsGroup.add(sprite);
  });

  state.mapLabelsGroup.visible = state.layers.labels;
  state.missionScene.add(state.mapLabelsGroup);
}

function applyMapMode(mode) {
  state.mapMode = mode;

  // 1. Sync UI buttons in mission toolbar
  const quickButtons = document.querySelectorAll('#map-mode-quick-switch .switch-btn[data-mode]');
  quickButtons.forEach(btn => {
    if (btn.dataset.mode === mode) btn.classList.add('active');
    else btn.classList.remove('active');
  });

  // 2. Sync mode cards in top-right dropdown
  const modeCards = document.querySelectorAll('#settings-menu-dropdown .mode-card[data-mode]');
  modeCards.forEach(card => {
    if (card.dataset.mode === mode) card.classList.add('active');
    else card.classList.remove('active');
  });

  // 3. Update top-right header badge
  const modeBadge = document.getElementById('header-active-mode-badge');
  if (modeBadge) {
    const badgeNames = {
      standard: 'STANDARD',
      satellite: 'SATELLITE',
      blender: 'BLENDER 2.5D',
      tactical: 'TACTICAL'
    };
    modeBadge.textContent = badgeNames[mode] || mode.toUpperCase();
  }

  // 4. Swap Materials & Lights based on mode
  if (mode === 'standard') {
    state.missionScene.background.setHex(0xb9d7e8);
    state.missionScene.fog.color.setHex(0xb9d7e8);
    if (state.standardGroundMesh) state.standardGroundMesh.visible = true;
    if (state.satelliteGroundMesh) state.satelliteGroundMesh.visible = false;

    // Hide GLB ground and road meshes so the detailed cartographic texture shows crisp
    state.terrainMeshes.forEach(m => { m.visible = false; });
    state.roadMeshes.forEach(m => { m.visible = false; });

    // Clean architectural limestone building prisms
    state.buildingMeshes.forEach(m => {
      m.material = new THREE.MeshStandardMaterial({
        color: 0xf8fafc,
        roughness: 0.8,
        metalness: 0.05
      });
    });

    state.restrictedMeshes.forEach(m => {
      m.material = new THREE.MeshStandardMaterial({
        color: 0xef4444,
        transparent: true,
        opacity: 0.5,
        roughness: 0.3
      });
    });

    // Warm natural daylight
    if (state.ambientMissionLight) {
      state.ambientMissionLight.color.setHex(0xffffff);
      state.ambientMissionLight.intensity = 0.8;
    }
    if (state.sunMissionLight) {
      state.sunMissionLight.color.setHex(0xfffbeb);
      state.sunMissionLight.intensity = 1.5;
    }
  } else if (mode === 'satellite') {
    state.missionScene.background.setHex(0x7896aa);
    state.missionScene.fog.color.setHex(0x7896aa);
    if (state.standardGroundMesh) state.standardGroundMesh.visible = false;
    if (state.satelliteGroundMesh) state.satelliteGroundMesh.visible = true;

    state.terrainMeshes.forEach(m => { m.visible = false; });
    state.roadMeshes.forEach(m => { m.visible = false; });

    // Semi-translucent glass/concrete building prisms
    const roofPalette = [0xd4c9b2, 0xb9c4c9, 0xe2ddd2, 0xafbac2, 0xc8bda7];
    state.buildingMeshes.forEach((m, index) => {
      m.material = new THREE.MeshStandardMaterial({
        color: roofPalette[index % roofPalette.length],
        roughness: 0.72,
        metalness: 0.08
      });
    });

    if (state.ambientMissionLight) {
      state.ambientMissionLight.color.setHex(0xdbeafe);
      state.ambientMissionLight.intensity = 0.65;
    }
    if (state.sunMissionLight) {
      state.sunMissionLight.intensity = 1.3;
    }
  } else if (mode === 'blender') {
    state.missionScene.background.setHex(0x17243a);
    state.missionScene.fog.color.setHex(0x17243a);
    if (state.standardGroundMesh) state.standardGroundMesh.visible = false;
    if (state.satelliteGroundMesh) state.satelliteGroundMesh.visible = false;

    state.terrainMeshes.forEach(m => {
      m.visible = true;
      m.material = new THREE.MeshStandardMaterial({ color: 0x141f36, roughness: 0.9, metalness: 0.1 });
    });

    state.roadMeshes.forEach(m => {
      m.visible = true;
      m.material = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8, metalness: 0.2 });
    });

    // Original architectural cyber gold
    state.buildingMeshes.forEach(m => {
      m.material = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.45, metalness: 0.15 });
    });

    if (state.ambientMissionLight) {
      state.ambientMissionLight.color.setHex(0xffffff);
      state.ambientMissionLight.intensity = 0.8;
    }
    if (state.sunMissionLight) {
      state.sunMissionLight.color.setHex(0xfffbeb);
      state.sunMissionLight.intensity = 1.8;
    }
  } else if (mode === 'tactical') {
    state.missionScene.background.setHex(0x040812);
    state.missionScene.fog.color.setHex(0x040812);
    if (state.standardGroundMesh) state.standardGroundMesh.visible = false;
    if (state.satelliteGroundMesh) state.satelliteGroundMesh.visible = false;

    state.terrainMeshes.forEach(m => {
      m.visible = true;
      m.material = new THREE.MeshStandardMaterial({ color: 0x060a12, roughness: 0.95, metalness: 0.2 });
    });

    state.roadMeshes.forEach(m => {
      m.visible = true;
      m.material = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.9 });
    });

    state.buildingMeshes.forEach(m => {
      m.material = new THREE.MeshStandardMaterial({
        color: 0x081326,
        roughness: 0.2,
        metalness: 0.8,
        transparent: true,
        opacity: 0.88
      });
    });

    if (state.ambientMissionLight) {
      state.ambientMissionLight.color.setHex(0x0284c7);
      state.ambientMissionLight.intensity = 0.45;
    }
    if (state.sunMissionLight) {
      state.sunMissionLight.color.setHex(0x38bdf8);
      state.sunMissionLight.intensity = 1.0;
    }
  }

  // Re-apply layer visibility to preserve user toggles across mode changes
  Object.keys(state.layers).forEach(k => {
    setLayerVisibility(k, state.layers[k]);
  });
}
window.applyMapMode = applyMapMode;

function setLayerVisibility(layerName, isVisible) {
  state.layers[layerName] = isVisible;

  const checkbox = document.getElementById(`toggle-layer-${layerName}`);
  if (checkbox) checkbox.checked = isVisible;

  if (layerName === 'buildings') {
    state.buildingMeshes.forEach(m => { m.visible = isVisible; });
  } else if (layerName === 'restricted') {
    state.restrictedMeshes.forEach(m => { m.visible = isVisible; });
  } else if (layerName === 'route') {
    if (state.routeLine) state.routeLine.visible = isVisible;
    if (state.startBeacon3D) state.startBeacon3D.visible = isVisible;
    if (state.goalBeacon3D) state.goalBeacon3D.visible = isVisible;
  } else if (layerName === 'directLine') {
    if (state.directLine3D) state.directLine3D.visible = isVisible;
  } else if (layerName === 'trafficDome') {
    if (state.trafficDome3D) state.trafficDome3D.visible = isVisible;
  } else if (layerName === 'labels') {
    if (state.mapLabelsGroup) state.mapLabelsGroup.visible = isVisible;
  }
}
window.setLayerVisibility = setLayerVisibility;

function setCameraPerspective(view) {
  state.cameraPerspective = view;

  const btn3d = document.getElementById('btn-cam-3d');
  const btn2d = document.getElementById('btn-cam-2d');

  if (view === '3d') {
    if (btn3d) btn3d.classList.add('active');
    if (btn2d) btn2d.classList.remove('active');
    if (state.missionCamera && state.missionControls) {
      state.missionCamera.position.set(-650, 780, 750);
      state.missionControls.target.set(0, 0, 0);
      state.missionControls.update();
    }
  } else {
    if (btn2d) btn2d.classList.add('active');
    if (btn3d) btn3d.classList.remove('active');
    if (state.missionCamera && state.missionControls) {
      state.missionCamera.position.set(0, 1150, 0);
      state.missionControls.target.set(0, 0, 0);
      state.missionControls.update();
    }
  }
}
window.setCameraPerspective = setCameraPerspective;

function build3DFlightPath() {
  const points = [];
  PLANNED_WAYPOINTS_NED.forEach(wp => {
    points.push(nedToGltf(wp[0], wp[1], wp[2]));
  });

  // 1. Glowing Emerald Route Polyline
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x10b981,
    linewidth: 3
  });
  state.routeLine = new THREE.Line(geometry, material);
  state.missionScene.add(state.routeLine);

  // 2. Start & Goal Beacons in 3D
  const startPos = nedToGltf(state.startNed[0], state.startNed[1], state.startNed[2]);
  const goalPos = nedToGltf(state.destination.ned[0], state.destination.ned[1], state.destination.ned[2]);

  // Start Beacon (Cyan Diamond)
  const startPin = create3DBeacon(0x38bdf8, 'START A');
  startPin.position.copy(startPos);
  state.startBeacon3D = startPin;
  state.missionScene.add(startPin);

  // Goal Beacon (Emerald/Amber Diamond)
  const goalPin = create3DBeacon(0x10b981, 'DESTINATION B');
  goalPin.position.copy(goalPos);
  state.goalBeacon3D = goalPin;
  state.missionScene.add(goalPin);

  // Blocked Direct Route (Dashed Line)
  const directPts = [startPos, goalPos];
  const directGeo = new THREE.BufferGeometry().setFromPoints(directPts);
  const directMat = new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 15,
    gapSize: 10
  });
  state.directLine3D = new THREE.Line(directGeo, directMat);
  state.directLine3D.computeLineDistances();
  state.missionScene.add(state.directLine3D);
}

function updateRoute3DVisuals(waypointsNED, startNED, goalNED) {
  if (!state.missionScene) return;

  const points3D = waypointsNED.map(wp => nedToGltf(wp[0], wp[1], wp[2]));
  const startPos = nedToGltf(startNED[0], startNED[1], startNED[2]);
  const goalPos = nedToGltf(goalNED[0], goalNED[1], goalNED[2]);

  // 1. Update Route Polyline
  if (state.routeLine) {
    state.routeLine.geometry.dispose();
    state.routeLine.geometry = new THREE.BufferGeometry().setFromPoints(points3D);
  }

  // 2. Reposition Start Beacon
  if (state.startBeacon3D) {
    state.startBeacon3D.position.copy(startPos);
  }

  // 3. Reposition Goal Beacon
  if (state.goalBeacon3D) {
    state.goalBeacon3D.position.copy(goalPos);
  }

  // 4. Update Direct Line
  if (state.directLine3D) {
    state.directLine3D.geometry.dispose();
    state.directLine3D.geometry = new THREE.BufferGeometry().setFromPoints([startPos, goalPos]);
    state.directLine3D.computeLineDistances();
  }
}

function build3DObstacleAvoidanceMarker() {
  const group = new THREE.Group();

  // 1. Glowing wireframe bounding cage for the approaching building obstacle
  const cageGeo = new THREE.BoxGeometry(42, 50, 36);
  const cageMat = new THREE.MeshBasicMaterial({
    color: 0xef4444,
    wireframe: true,
    transparent: true,
    opacity: 0.85
  });
  const cageMesh = new THREE.Mesh(cageGeo, cageMat);
  cageMesh.position.set(-185, 25, 260);
  group.add(cageMesh);

  // 2. Obstacle Proximity Warning Diamond at base (Zero circles)
  const ringGeo = new THREE.BoxGeometry(44, 1.2, 44);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xf59e0b,
    wireframe: true,
    transparent: true,
    opacity: 0.75
  });
  const ringMesh = new THREE.Mesh(ringGeo, ringMat);
  ringMesh.rotation.y = Math.PI / 4; // Diamond orientation
  ringMesh.position.set(-185, 1.5, 260);
  group.add(ringMesh);

  // 3. Dynamic 3D Evasion Bypass Arc (Glowing Emerald spline around building)
  const curvePoints = [
    new THREE.Vector3(-225, 6, 295),
    new THREE.Vector3(-205, 6, 280),
    new THREE.Vector3(-155, 6, 260),
    new THREE.Vector3(-140, 6, 235),
    new THREE.Vector3(-120, 6, 210)
  ];
  const curve = new THREE.CatmullRomCurve3(curvePoints);
  const curveGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(30));
  const curveMat = new THREE.LineBasicMaterial({
    color: 0x10b981,
    linewidth: 4
  });
  const bypassLine = new THREE.Line(curveGeo, curveMat);
  group.add(bypassLine);

  state.obstacleBox3D = group;
  state.missionScene.add(group);
}

function create3DBeacon(color, label) {
  const group = new THREE.Group();

  // Vertical light pillar: Square Rectilinear Beam (Zero circles)
  const pillarGeo = new THREE.BoxGeometry(1.2, 44, 1.2);
  const pillarMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.65 });
  const pillar = new THREE.Mesh(pillarGeo, pillarMat);
  pillar.position.y = 22;
  group.add(pillar);

  // Floating Diamond Beacon: Sharp 8-faceted Octahedron (Zero circles)
  const diamondGeo = new THREE.OctahedronGeometry(7.0, 0);
  const diamondMat = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.85,
    metalness: 0.2,
    roughness: 0.25
  });
  const diamond = new THREE.Mesh(diamondGeo, diamondMat);
  diamond.position.y = 44;
  diamond.name = 'beaconDiamond';
  group.add(diamond);

  return group;
}

function buildMissionDrone() {
  const group = new THREE.Group();
  const fallback = new THREE.Group();
  fallback.name = 'proceduralDroneFallback';
  group.add(fallback);

  // Central body
  const bodyGeo = new THREE.BoxGeometry(6.0, 1.4, 4.0);
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, metalness: 0.3, roughness: 0.2 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  fallback.add(body);

  // 4 Arms & Motor Pods
  const armMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
  const bladeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.8 });
  const armOffsets = [[4.5, 4.5], [-4.5, 4.5], [4.5, -4.5], [-4.5, -4.5]];

  armOffsets.forEach(([ax, az]) => {
    const armGeo = new THREE.CylinderGeometry(0.3, 0.3, 6.5);
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.position.set(ax * 0.5, 0, az * 0.5);
    arm.rotation.z = Math.PI / 4 * Math.sign(ax);
    fallback.add(arm);

    // Motor
    const motorGeo = new THREE.CylinderGeometry(0.8, 0.8, 1.2, 16);
    const motor = new THREE.Mesh(motorGeo, bodyMat);
    motor.position.set(ax, 0.5, az);
    fallback.add(motor);

    // Spinning Blade
    const bladeGeo = new THREE.BoxGeometry(5.0, 0.15, 0.6);
    const blade = new THREE.Mesh(bladeGeo, bladeMat);
    blade.position.set(ax, 1.2, az);
    state.missionDroneProps.push(blade);
    fallback.add(blade);
  });

  // Spotlight shining down from drone
  const spot = new THREE.SpotLight(0x38bdf8, 3.0, 120, Math.PI / 6, 0.3);
  spot.position.set(0, 0, 0);
  spot.target.position.set(0, -60, 0);
  group.add(spot);
  group.add(spot.target);

  state.missionDroneGroup = group;
  const initialScale = state.droneScale || 1.0;
  group.scale.set(initialScale, initialScale, initialScale);
  state.missionScene.add(group);

  // Use the supplied aircraft asset in the mission world; the procedural craft remains a load fallback.
  const loader = new GLTFLoader();
  loader.load('/thermal-simulation/gray_drone.glb', (gltf) => {
    const model = gltf.scene;
    model.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(model);
    const size = bounds.getSize(new THREE.Vector3());
    const center = bounds.getCenter(new THREE.Vector3());
    const wrapper = new THREE.Group();
    model.position.sub(center);
    wrapper.add(model);
    const maxDimension = Math.max(size.x, size.y, size.z) || 1;
    const scale = 8 / maxDimension;
    wrapper.scale.setScalar(scale);
    wrapper.rotation.y = Math.PI;
    model.traverse((child) => {
      if (!child.isMesh) return;
      child.castShadow = true;
      child.receiveShadow = true;
      const n = (child.name || '').toLowerCase();
      if (n.includes('prop') || n.includes('blade') || n.includes('rotor')) state.missionDroneProps.push(child);
    });
    fallback.visible = false;
    group.add(wrapper);
    state.missionDroneModel = wrapper;
  }, undefined, (error) => console.warn('Mission aircraft GLB unavailable; retaining procedural fallback.', error));
}

// =============================================================================
// 3. VEHICLE TWIN VIEWPORT (Close-up 3D CAD Twin in Right Panel)
// =============================================================================
function initThreeJsTwin() {
  state.twinContainer = document.getElementById('twin-viewport-box');
  state.twinCanvas = document.getElementById('twin-canvas');
  if (!state.twinContainer || !state.twinCanvas) return;

  const w = state.twinContainer.clientWidth || 280;
  const h = state.twinContainer.clientHeight || 140;

  state.twinScene = new THREE.Scene();
  state.twinCamera = new THREE.PerspectiveCamera(38, w / h, 0.1, 100);
  state.twinCamera.position.set(1.6, 1.2, 1.6);
  state.twinCamera.lookAt(0, 0, 0);

  state.twinRenderer = new THREE.WebGLRenderer({
    canvas: state.twinCanvas,
    antialias: true,
    alpha: true
  });
  state.twinRenderer.setSize(w, h);
  state.twinRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  state.twinControls = new OrbitControls(state.twinCamera, state.twinCanvas);
  state.twinControls.target.set(0, 0, 0);
  state.twinControls.enableDamping = true;
  state.twinControls.dampingFactor = 0.06;
  state.twinControls.maxDistance = 6;
  state.twinControls.minDistance = 0.5;
  state.twinControls.update();

  // Studio Lighting
  const amb = new THREE.AmbientLight(0xffffff, 2.2);
  state.twinScene.add(amb);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x1e293b, 1.8);
  state.twinScene.add(hemi);

  const dir1 = new THREE.DirectionalLight(0xffffff, 3.2);
  dir1.position.set(4, 7, 5);
  state.twinScene.add(dir1);

  const blueRim = new THREE.DirectionalLight(0x38bdf8, 2.2);
  blueRim.position.set(-4, 3, -4);
  state.twinScene.add(blueRim);

  const bottomFill = new THREE.DirectionalLight(0xffffff, 1.2);
  bottomFill.position.set(0, -4, 0);
  state.twinScene.add(bottomFill);

  // Subtle CAD Grid Helper under drone
  const cadGrid = new THREE.GridHelper(2.5, 8, 0x38bdf8, 0x1e293b);
  cadGrid.position.y = -0.22;
  state.twinScene.add(cadGrid);

  // Load Authentic thermal-simulation/gray_drone.glb
  const loader = new GLTFLoader();
  loader.load(
    '/thermal-simulation/gray_drone.glb',
    (gltf) => {
      if (state.twinDroneModel) {
        state.twinScene.remove(state.twinDroneModel);
      }
      state.twinPropellers = [];

      const model = gltf.scene;

      // Compute unscaled bounding box
      model.updateMatrixWorld(true);
      const bbox = new THREE.Box3().setFromObject(model);
      const size = bbox.getSize(new THREE.Vector3());
      const center = bbox.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const targetDim = 0.95;
      const scale = maxDim > 0 ? (targetDim / maxDim) : 1.0;

      // Offset model inside wrapper to align exactly at origin
      model.position.set(-center.x, -center.y, -center.z);

      const wrapper = new THREE.Group();
      wrapper.add(model);
      wrapper.scale.set(scale, scale, scale);
      wrapper.position.set(0, 0, 0);

      // Enhance mesh materials & extract propeller meshes
      model.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          if (child.material) {
            child.material.side = THREE.DoubleSide;
            child.material.metalness = Math.min(child.material.metalness || 0.2, 0.35);
            child.material.roughness = Math.max(child.material.roughness || 0.35, 0.3);
          }
          const cName = (child.name || '').toLowerCase();
          if (cName.includes('prop') || cName.includes('blade') || cName.includes('rotor')) {
            state.twinPropellers.push(child);
          }
        }
      });

      // Bind authentic hover animation clip to AnimationMixer
      if (gltf.animations && gltf.animations.length > 0) {
        state.twinMixer = new THREE.AnimationMixer(model);
        const hoverClip = gltf.animations.find(a => a.name.toLowerCase().includes('hover')) || gltf.animations[0];
        if (hoverClip) {
          state.twinHoverAction = state.twinMixer.clipAction(hoverClip);
          state.twinHoverAction.play();
          state.twinHoverAction.timeScale = 1.0;
        }
      }

      state.twinDroneModel = wrapper;
      state.twinScene.add(wrapper);
      console.log('Successfully loaded and centered gray_drone.glb in Vehicle Twin. Propeller parts:', state.twinPropellers.length);
    },
    undefined,
    (err) => {
      console.warn('Failed to load gray_drone.glb, fallback to procedural CAD twin:', err);
      buildTwinProceduralDrone();
    }
  );

  window.addEventListener('resize', () => {
    if (!state.twinContainer || !state.twinCamera || !state.twinRenderer) return;
    const tw = state.twinContainer.clientWidth || 280;
    const th = state.twinContainer.clientHeight || 140;
    state.twinCamera.aspect = tw / th;
    state.twinCamera.updateProjectionMatrix();
    state.twinRenderer.setSize(tw, th);
  });
}

function buildTwinProceduralDrone() {
  const group = new THREE.Group();

  // Bright satin metallic chassis
  const bodyGeo = new THREE.BoxGeometry(0.65, 0.12, 0.42);
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0xf1f5f9,
    metalness: 0.2,
    roughness: 0.25
  });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  group.add(body);

  // Dark avionics canopy
  const canopyGeo = new THREE.CylinderGeometry(0.14, 0.2, 0.08, 16);
  const canopyMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    metalness: 0.3,
    roughness: 0.3
  });
  const canopy = new THREE.Mesh(canopyGeo, canopyMat);
  canopy.position.y = 0.09;
  group.add(canopy);

  // 4 Carbon Arms & Motor Pods
  const armMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.3, roughness: 0.4 });
  const motorMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.4, roughness: 0.2 });
  const bladeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 });

  const armOffsets = [[0.45, 0.45], [-0.45, 0.45], [0.45, -0.45], [-0.45, -0.45]];

  armOffsets.forEach(([ax, az]) => {
    const armGeo = new THREE.CylinderGeometry(0.025, 0.025, 0.65);
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.position.set(ax * 0.5, 0, az * 0.5);
    arm.rotation.z = Math.PI / 4 * Math.sign(ax);
    group.add(arm);

    const motorGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.1, 16);
    const motor = new THREE.Mesh(motorGeo, motorMat);
    motor.position.set(ax, 0.04, az);
    group.add(motor);

    const bladeGeo = new THREE.BoxGeometry(0.5, 0.012, 0.05);
    const blade = new THREE.Mesh(bladeGeo, bladeMat);
    blade.position.set(ax, 0.1, az);
    state.twinPropellers.push(blade);
    group.add(blade);
  });

  // Navigation LEDs
  const greenLed = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
  greenLed.position.set(0.45, 0.04, 0.45);
  group.add(greenLed);

  const redLed = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), new THREE.MeshBasicMaterial({ color: 0xef4444 }));
  redLed.position.set(-0.45, 0.04, 0.45);
  group.add(redLed);

  state.twinDroneModel = group;
  state.twinScene.add(group);
}

// =============================================================================
// 4. LATENCY CHART CANVAS
// =============================================================================
function initLatencyChart() {
  state.latCanvas = document.getElementById('latency-canvas');
  if (!state.latCanvas) return;
  state.latCtx = state.latCanvas.getContext('2d');
  resizeLatencyCanvas();
  window.addEventListener('resize', resizeLatencyCanvas);
}

function resizeLatencyCanvas() {
  if (!state.latCanvas || !state.latCanvas.parentElement) return;
  state.latCanvas.width = state.latCanvas.parentElement.clientWidth || 340;
  state.latCanvas.height = state.latCanvas.parentElement.clientHeight || 115;
}

function renderLatencyChart() {
  const ctx = state.latCtx;
  const cvs = state.latCanvas;
  if (!ctx || !cvs) return;
  const w = cvs.width;
  const h = cvs.height;
  if (w === 0 || h === 0) return;

  ctx.clearRect(0, 0, w, h);

  // Background Grid Lines
  ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
  ctx.lineWidth = 1;
  const levels = [0, 50, 100, 150, 200];
  const startX = 28;
  const plotW = w - startX - 8;

  ctx.font = '8px "JetBrains Mono", monospace';
  ctx.fillStyle = '#64748b';
  ctx.textAlign = 'right';

  levels.forEach(val => {
    const y = h - 6 - (val / 200) * (h - 14);
    ctx.beginPath();
    ctx.moveTo(startX, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.fillText(String(val), startX - 5, y + 3);
  });

  const timeOffset = Date.now() * 0.0015;
  const count = 100;

  // 1. Perception (Cyan #38bdf8, mean ~14ms, peaks 24ms)
  drawCurve(ctx, plotW, h, startX, count, 26, 8, '#38bdf8', timeOffset, 1.4);

  // 2. Planning (Emerald #22c55e, mean ~8ms, peaks 16ms)
  drawCurve(ctx, plotW, h, startX, count, 14, 5, '#22c55e', timeOffset, 0.9);

  // 3. Control (Yellow #facc15, mean ~4ms, peaks 8ms)
  drawCurve(ctx, plotW, h, startX, count, 6, 2.5, '#facc15', timeOffset, 2.2);

  // Mean (28ms) guideline
  const meanY = h - 6 - (28 / 200) * (h - 14);
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.beginPath();
  ctx.moveTo(startX, meanY);
  ctx.lineTo(w, meanY);
  ctx.stroke();
  ctx.restore();
}

function drawCurve(ctx, plotW, h, startX, count, baseMs, jitterMs, color, timeOffset, freq) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();

  for (let i = 0; i < count; i++) {
    const x = startX + (i / (count - 1)) * plotW;
    const t = i * 0.2 + timeOffset * freq;
    const ms = baseMs + Math.sin(t * 1.8) * jitterMs * 0.6 + Math.cos(t * 3.4) * jitterMs * 0.4;
    const y = h - 6 - (ms / 200) * (h - 14);

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

// =============================================================================
// 5. PLAYBACK & HUD CONTROLLERS
// =============================================================================
function initPlaybackControls() {
  const btnPlay = document.getElementById('btn-play');
  const btnPause = document.getElementById('btn-pause');
  const btnRestart = document.getElementById('btn-restart');
  const scrubber = document.getElementById('trace-scrubber');
  const speedBtns = document.querySelectorAll('.speed-btn');

  if (btnPlay) btnPlay.addEventListener('click', () => {
    // A deliberate operator resume clears the latched hold; the geometry guard remains active.
    state.collisionGuardTriggered = false;
    state.isPlaying = true;
    state.isHoveringAtGoal = false;
  });
  if (btnPause) btnPause.addEventListener('click', () => {
    state.isPlaying = false;
  });
  if (btnRestart) {
    btnRestart.addEventListener('click', () => {
      state.traceIndex = 0;
      state.flightProgress = 0;
      state.isHoveringAtGoal = false;
      state.isPlaying = true;
      resetCollisionAvoidanceState();
      if (scrubber) scrubber.value = 0;
    });
  }
  if (scrubber) {
    scrubber.addEventListener('input', (e) => {
      state.traceIndex = parseInt(e.target.value, 10);
      state.flightProgress = state.traceIndex;
      state.isHoveringAtGoal = (state.traceIndex >= state.totalPoints - 1);
      resetCollisionAvoidanceState();
    });
  }
  speedBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      speedBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.playSpeed = parseFloat(btn.getAttribute('data-speed'));
    });
  });
}

function updatePlaybackHUD() {
  const scrubber = document.getElementById('trace-scrubber');
  const nedEl = document.getElementById('pb-ned');
  const speedEl = document.getElementById('pb-speed');
  const timeEl = document.getElementById('pb-time');
  const progressSteps = document.getElementById('score-steps');
  const progressFill = document.getElementById('score-prog-fill');
  const progressPct = document.getElementById('score-pct');
  const twinPos = document.getElementById('spec-init-pos');

  if (scrubber && document.activeElement !== scrubber) {
    scrubber.value = state.traceIndex;
  }

  const cur = (state.traceData && state.traceData[state.traceIndex]) || {
    position: [-360.0, -400.0, -6.0],
    velocity: [1.8, 2.1, 0.0],
    t: 0
  };

  const px = cur.position[0].toFixed(1);
  const py = cur.position[1].toFixed(1);
  const pz = (-Math.abs(state.cruiseAltitude || 6.0)).toFixed(1);
  if (nedEl) nedEl.textContent = `NED: ${px}, ${py}, ${pz}m`;

  const effSpeed = state.realFlightSpeed !== undefined ? state.realFlightSpeed : 4.2;
  if (speedEl) speedEl.textContent = `${effSpeed.toFixed(1)} m/s`;

  const t = cur.t ? cur.t.toFixed(1) : (state.traceIndex * 0.95).toFixed(1);
  if (timeEl) timeEl.textContent = `${t}s / 479s`;

  const pct = Math.min(100, Math.round((state.traceIndex / (state.totalPoints || 1)) * 100));
  const stepNum = Math.min(1000, Math.round((state.traceIndex / (state.totalPoints || 1)) * 1000));
  if (progressSteps) progressSteps.textContent = `${stepNum} / 1,000`;
  if (progressFill) progressFill.style.width = `${pct}%`;
  if (progressPct) progressPct.textContent = `${pct}%`;

  const lat = (25.1396 + (cur.position[0] + 360) / 111320).toFixed(4);
  const lon = (55.2543 + (cur.position[1] + 400) / 100800).toFixed(4);
  const alt = (state.cruiseAltitude || Math.abs(cur.position[2])).toFixed(1);
  if (twinPos) twinPos.textContent = `${lat}, ${lon}, ${alt} m (AGL)`;

  // Synchronize Active Waypoint Chip during playback
  const wpIndex = Math.min(10, Math.floor((state.traceIndex / (state.totalPoints || 1)) * 10));
  if (wpIndex !== state.activeWaypointIndex) {
    state.activeWaypointIndex = wpIndex;
    const wpChips = document.querySelectorAll('.wp-chip');
    wpChips.forEach((c, i) => {
      if (i === wpIndex) c.classList.add('active');
      else c.classList.remove('active');
    });
    const badge = document.getElementById('disp-active-wp');
    if (badge) {
      badge.textContent = wpIndex === 0 ? 'WP 1 (START)' : (wpIndex === 10 ? 'WP 10 (GOAL)' : `WP ${wpIndex + 1}`);
    }
  }

  // Update battery failsafe reachability
  updateBatteryPowerFailsafe();
}

function initClock() {
  const clockEl = document.getElementById('dubai-clock');
  function tick() {
    const now = new Date();
    const utc = now.getTime() + now.getTimezoneOffset() * 60000;
    const dubai = new Date(utc + 4 * 3600000);
    const yyyy = dubai.getFullYear();
    const mm = String(dubai.getMonth() + 1).padStart(2, '0');
    const dd = String(dubai.getDate()).padStart(2, '0');
    const hh = String(dubai.getHours()).padStart(2, '0');
    const min = String(dubai.getMinutes()).padStart(2, '0');
    const ss = String(dubai.getSeconds()).padStart(2, '0');
    if (clockEl) clockEl.textContent = `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
  }
  tick();
  setInterval(tick, 1000);
}

function updatePerceptionHUD(data) {
  // Static YOLO ground vehicles removed per user directive
}

// =============================================================================
// CAMERA HUD & REACTIVE OBSTACLE AVOIDANCE
// =============================================================================

function initCameraHUDCanvas() {
  state.cameraCanvas = document.getElementById('camera-hud-canvas');
  if (!state.cameraCanvas) return;
  state.cameraCtx = state.cameraCanvas.getContext('2d');
  resizeCameraHUDCanvas();
  window.addEventListener('resize', resizeCameraHUDCanvas);
}

function resizeCameraHUDCanvas() {
  if (!state.cameraCanvas || !state.cameraCanvas.parentElement) return;
  state.cameraCanvas.width = state.cameraCanvas.parentElement.clientWidth || 280;
  state.cameraCanvas.height = state.cameraCanvas.parentElement.clientHeight || 170;
}

function initAvoidanceScenarioButtons() {
  const chips = document.querySelectorAll('.avoid-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.avoidanceMode = chip.getAttribute('data-mode');
      updateReactivePerceptionHUD();
    });
  });
}

function getEffectiveReactiveDecision() {
  return state.actualReactiveDecision || {
    obstacle_detected: false,
    obstacle_type: 'None',
    obstacle_id: null,
    distance_m: Infinity,
    ttc_s: Infinity,
    threat_level: 'CLEAR',
    collision_course: false,
    bbox: null,
    candidate_paths: [
      { direction: 'LEFT', clearance_m: 20, status: 'FEASIBLE' },
      { direction: 'RIGHT', clearance_m: 20, status: 'FEASIBLE' },
      { direction: 'TOP', clearance_m: 20, vertical_offset_m: 0, status: 'FEASIBLE' }
    ],
    optimal_path: 'FORWARD',
    optimal_action: 'FORWARD CORRIDOR CLEAR',
    optimal_primitive: 'CRUISE_LEVEL',
    clearance_margin_m: Infinity
  };
}

function projectBoxToCamera(box, camera) {
  const xs = [box.min.x, box.max.x];
  const ys = [box.min.y, box.max.y];
  const zs = [box.min.z, box.max.z];
  let minX = 1, maxX = -1, minY = 1, maxY = -1, visibleCorners = 0;
  for (const x of xs) for (const y of ys) for (const z of zs) {
    const point = new THREE.Vector3(x, y, z);
    const view = point.clone().applyMatrix4(camera.matrixWorldInverse);
    if (view.z >= -camera.near) continue;
    point.project(camera);
    minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y); maxY = Math.max(maxY, point.y);
    visibleCorners += 1;
  }
  if (!visibleCorners || maxX < -1 || minX > 1 || maxY < -1 || minY > 1) return null;
  const left = THREE.MathUtils.clamp((minX * 0.5 + 0.5) * 100, 0, 100);
  const right = THREE.MathUtils.clamp((maxX * 0.5 + 0.5) * 100, 0, 100);
  const top = THREE.MathUtils.clamp((0.5 - maxY * 0.5) * 100, 0, 100);
  const bottom = THREE.MathUtils.clamp((0.5 - minY * 0.5) * 100, 0, 100);
  if (right - left < 1 || bottom - top < 1) return null;
  return { top, left, width: right - left, height: bottom - top };
}

function pointToSegmentDistance2D(point, a, b) {
  const abN = b[0] - a[0];
  const abE = b[1] - a[1];
  const lengthSq = abN * abN + abE * abE;
  if (lengthSq < 1e-9) return Math.hypot(point[0] - a[0], point[1] - a[1]);
  const t = THREE.MathUtils.clamp(
    ((point[0] - a[0]) * abN + (point[1] - a[1]) * abE) / lengthSq,
    0,
    1
  );
  return Math.hypot(point[0] - (a[0] + t * abN), point[1] - (a[1] + t * abE));
}

function pointInTriangle2D(point, triangle) {
  const [a, b, c] = triangle;
  const cross = (p1, p2, p3) =>
    (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
  const d1 = cross(a, b, point);
  const d2 = cross(b, c, point);
  const d3 = cross(c, a, point);
  const hasNegative = d1 < -1e-7 || d2 < -1e-7 || d3 < -1e-7;
  const hasPositive = d1 > 1e-7 || d2 > 1e-7 || d3 > 1e-7;
  return !(hasNegative && hasPositive);
}

function pointToTriangleDistance2D(point, triangle) {
  if (pointInTriangle2D(point, triangle)) return 0;
  return Math.min(
    pointToSegmentDistance2D(point, triangle[0], triangle[1]),
    pointToSegmentDistance2D(point, triangle[1], triangle[2]),
    pointToSegmentDistance2D(point, triangle[2], triangle[0])
  );
}

function pointToBoundsDistance2D(north, east, bounds) {
  const dNorth = north < bounds.minNorth ? bounds.minNorth - north
    : north > bounds.maxNorth ? north - bounds.maxNorth : 0;
  const dEast = east < bounds.minEast ? bounds.minEast - east
    : east > bounds.maxEast ? east - bounds.maxEast : 0;
  return Math.hypot(dNorth, dEast);
}

function exactBuildingClearance(building, north, east, altitude) {
  // Bounds are broad-phase only. Distance must always be measured against the
  // triangulated footprint; using the AABB distance near a corner creates a
  // false obstacle in empty space around rotated/concave buildings.
  let horizontalDistance = Infinity;
  const point = [north, east];
  for (const triangle of building.triangles) {
    horizontalDistance = Math.min(horizontalDistance, pointToTriangleDistance2D(point, triangle));
    if (horizontalDistance === 0) break;
  }
  const verticalDistance = altitude < 0 ? -altitude
    : altitude > building.height ? altitude - building.height : 0;
  if (horizontalDistance === 0) return verticalDistance;
  if (verticalDistance === 0) return horizontalDistance;
  return Math.hypot(horizontalDistance, verticalDistance);
}

function linkCollisionGeometryToMeshes() {
  if (!state.collisionBuildings.length) return;
  for (const building of state.collisionBuildings) {
    const featureName = building.id.toLowerCase();
    const matches = state.buildingColliders.filter(item => {
      const meshName = item.id.toLowerCase();
      return meshName === featureName || meshName.startsWith(`${featureName}_`);
    });
    if (matches.length) {
      building.meshes = matches.map(item => item.mesh);
      building.displayBox = matches[0].box.clone();
      for (let i = 1; i < matches.length; i += 1) building.displayBox.union(matches[i].box);
    }
  }
}

async function loadExactCollisionGeometry() {
  try {
    const response = await fetch('/worlds/dubai_osm/geometry.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const features = Array.isArray(payload) ? payload : (payload.features || []);
    state.collisionBuildings = features
      .filter(feature => feature.kind === 'building' && Array.isArray(feature.triangles) && feature.triangles.length)
      .map(feature => {
        // geometry.json stores [east, north]; flight state uses [north, east].
        const triangles = feature.triangles.map(triangle =>
          triangle.map(point => [Number(point[1]), Number(point[0])])
        );
        let minNorth = Infinity, maxNorth = -Infinity, minEast = Infinity, maxEast = -Infinity;
        for (const triangle of triangles) for (const [north, east] of triangle) {
          minNorth = Math.min(minNorth, north); maxNorth = Math.max(maxNorth, north);
          minEast = Math.min(minEast, east); maxEast = Math.max(maxEast, east);
        }
        const height = Math.max(1, Number(feature.height_m) || 8);
        return {
          id: String(feature.id),
          height,
          triangles,
          bounds: { minNorth, maxNorth, minEast, maxEast },
          // Exact footprint drives collision; this box is only a HUD projection/highlight aid.
          displayBox: new THREE.Box3(
            new THREE.Vector3(minEast, 0, -maxNorth),
            new THREE.Vector3(maxEast, height, -minNorth)
          ),
          meshes: []
        };
      });
    linkCollisionGeometryToMeshes();
    console.log(`Loaded ${state.collisionBuildings.length} exact OSM building collision footprints.`);
  } catch (error) {
    state.collisionBuildings = [];
    console.error('Exact collision geometry unavailable; obstacle avoidance is fail-closed.', error);
  }
}

function firstCorridorImpactDistance(building, north, east, dNorth, dEast, altitude, maxDistance, bodyRadius) {
  if (altitude > building.height + bodyRadius || altitude < -bodyRadius) return Infinity;
  const endNorth = north + dNorth * maxDistance;
  const endEast = east + dEast * maxDistance;
  const corridorBounds = {
    minNorth: Math.min(north, endNorth) - bodyRadius,
    maxNorth: Math.max(north, endNorth) + bodyRadius,
    minEast: Math.min(east, endEast) - bodyRadius,
    maxEast: Math.max(east, endEast) + bodyRadius
  };
  if (corridorBounds.maxNorth < building.bounds.minNorth || corridorBounds.minNorth > building.bounds.maxNorth ||
      corridorBounds.maxEast < building.bounds.minEast || corridorBounds.minEast > building.bounds.maxEast) return Infinity;

  const step = 1.0;
  let previous = 0;
  for (let distance = 0; distance <= maxDistance; distance += step) {
    const clearance = exactBuildingClearance(
      building,
      north + dNorth * distance,
      east + dEast * distance,
      altitude
    );
    if (clearance <= bodyRadius) {
      let low = previous;
      let high = distance;
      for (let iteration = 0; iteration < 6; iteration += 1) {
        const mid = (low + high) / 2;
        const midClearance = exactBuildingClearance(
          building,
          north + dNorth * mid,
          east + dEast * mid,
          altitude
        );
        if (midClearance <= bodyRadius) high = mid;
        else low = mid;
      }
      return high;
    }
    previous = distance;
  }
  return Infinity;
}

function plannedCorridorImpactDistance(building, north, east, altitude, speed, bodyRadius, fallbackDirection) {
  const predictionDistance = Math.min(100, Math.max(20, speed * 8));
  const points = [[north, east]];
  const firstFutureIndex = Math.min(state.totalPoints - 1, Math.floor(state.flightProgress) + 1);
  if (Array.isArray(state.traceData) && state.traceData.length > firstFutureIndex) {
    for (let index = firstFutureIndex; index < state.traceData.length && points.length < 90; index += 1) {
      const routePoint = state.traceData[index]?.position;
      if (!routePoint) continue;
      points.push([
        Number(routePoint[0]) + state.avoidanceOffsetNed.north,
        Number(routePoint[1]) + state.avoidanceOffsetNed.east
      ]);
      const previous = points[points.length - 2];
      const accumulatedEstimate = Math.hypot(points.at(-1)[0] - north, points.at(-1)[1] - east);
      if (accumulatedEstimate >= predictionDistance && points.length > 2) break;
      if (Math.hypot(points.at(-1)[0] - previous[0], points.at(-1)[1] - previous[1]) > predictionDistance) break;
    }
  }
  if (points.length === 1) {
    points.push([
      north + fallbackDirection[0] * predictionDistance,
      east + fallbackDirection[1] * predictionDistance
    ]);
  }

  let travelled = 0;
  for (let index = 1; index < points.length && travelled < predictionDistance; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    const deltaNorth = end[0] - start[0];
    const deltaEast = end[1] - start[1];
    const segmentLength = Math.hypot(deltaNorth, deltaEast);
    if (segmentLength < 1e-5) continue;
    const allowedLength = Math.min(segmentLength, predictionDistance - travelled);
    const impact = firstCorridorImpactDistance(
      building,
      start[0],
      start[1],
      deltaNorth / segmentLength,
      deltaEast / segmentLength,
      altitude,
      allowedLength,
      bodyRadius
    );
    if (Number.isFinite(impact)) return travelled + impact;
    travelled += allowedLength;
  }
  return Infinity;
}

function measureActualBuildingThreat(position, direction, speed) {
  if (!state.collisionBuildings.length || !state.perceptionCamera) return null;
  const bodyRadius = 3.5;
  const north = -position.z;
  const east = position.x;
  const altitude = position.y;
  const horizontalLength = Math.hypot(direction.z, direction.x);
  if (horizontalLength < 1e-6) return null;
  const dNorth = -direction.z / horizontalLength;
  const dEast = direction.x / horizontalLength;
  let bestCollision = null;
  let bestVisible = null;

  for (const building of state.collisionBuildings) {
    const broadphaseDistance = pointToBoundsDistance2D(north, east, building.bounds);
    if (broadphaseDistance > 120) continue;
    const centerNorth = (building.bounds.minNorth + building.bounds.maxNorth) / 2;
    const centerEast = (building.bounds.minEast + building.bounds.maxEast) / 2;
    const ahead = (centerNorth - north) * dNorth + (centerEast - east) * dEast;
    if (ahead < -bodyRadius || ahead > 120) continue;

    const surfaceDistance = exactBuildingClearance(building, north, east, altitude);
    const corridorDistance = plannedCorridorImpactDistance(
      building, north, east, altitude, speed, bodyRadius, [dNorth, dEast]
    );
    const bbox = projectBoxToCamera(building.displayBox, state.perceptionCamera);
    const candidate = { building, bbox, surfaceDistance, corridorDistance };
    if (Number.isFinite(corridorDistance) &&
        (!bestCollision || corridorDistance < bestCollision.corridorDistance)) bestCollision = candidate;
    if (bbox && (!bestVisible || surfaceDistance < bestVisible.surfaceDistance)) bestVisible = candidate;
  }
  const best = bestCollision || bestVisible;
  if (!best) return null;

  const boxCenter = best.bbox ? best.bbox.left + best.bbox.width / 2 : 50;
  let optimal = boxCenter < 50 ? 'RIGHT' : 'LEFT';
  if (state.avoidanceMode === 'right') optimal = 'RIGHT';
  if (state.avoidanceMode === 'left') optimal = 'LEFT';
  if (state.avoidanceMode === 'top') optimal = 'TOP';
  const collisionCourse = Number.isFinite(best.corridorDistance);
  const measuredDistance = collisionCourse ? best.corridorDistance : best.surfaceDistance;
  const ttc = collisionCourse && speed > 0.05 ? measuredDistance / speed : Infinity;
  const threat = collisionCourse && (measuredDistance < 7 || ttc < 2) ? 'CRITICAL'
    : collisionCourse && (measuredDistance < 30 || ttc < 8) ? 'CAUTION' : 'CLEAR';
  const buildingHeight = best.building.height;
  const topOffset = Math.max(0, buildingHeight + bodyRadius - position.y);
  const clearSide = Math.max(bodyRadius + 2, Math.min(20, measuredDistance + 8));
  const blockedSide = Math.max(0, measuredDistance - bodyRadius);

  return {
    obstacle_detected: true,
    obstacle_type: 'Building',
    obstacle_id: best.building.id,
    distance_m: measuredDistance,
    ttc_s: ttc,
    threat_level: threat,
    collision_course: collisionCourse,
    bbox: best.bbox,
    collider: best.building,
    candidate_paths: [
      { direction: 'LEFT', clearance_m: optimal === 'LEFT' ? clearSide : blockedSide, status: optimal === 'LEFT' ? 'OPTIMAL' : (collisionCourse ? 'BLOCKED' : 'FEASIBLE') },
      { direction: 'RIGHT', clearance_m: optimal === 'RIGHT' ? clearSide : blockedSide, status: optimal === 'RIGHT' ? 'OPTIMAL' : (collisionCourse ? 'BLOCKED' : 'FEASIBLE') },
      { direction: 'TOP', clearance_m: Math.max(0, 20 - topOffset), vertical_offset_m: topOffset, status: optimal === 'TOP' ? 'OPTIMAL' : (topOffset <= 8 ? 'FEASIBLE' : 'BLOCKED') }
    ],
    optimal_path: optimal,
    optimal_action: collisionCourse ? `COLLISION COURSE: EVADE ${optimal}` : 'BUILDING TRACKED · CORRIDOR CLEAR',
    optimal_primitive: optimal === 'RIGHT' ? 'BANK_RIGHT_EVADE' : (optimal === 'LEFT' ? 'BANK_LEFT_EVADE' : 'ASCEND_CLEAR'),
    clearance_margin_m: best.surfaceDistance
  };
}

function minimumBuildingClearance(position) {
  if (!state.collisionBuildings.length) return Infinity;
  const north = -position.z;
  const east = position.x;
  const altitude = position.y;
  let minimum = Infinity;
  for (const building of state.collisionBuildings) {
    const verticalDistance = altitude > building.height ? altitude - building.height : 0;
    const boundsDistance = pointToBoundsDistance2D(north, east, building.bounds);
    const lowerBound = Math.hypot(boundsDistance, verticalDistance);
    if (lowerBound >= minimum) continue;
    minimum = Math.min(minimum, exactBuildingClearance(building, north, east, altitude));
  }
  return minimum;
}

function resetCollisionAvoidanceState() {
  state.avoidanceOffsetNed = { north: 0, east: 0 };
  state.avoidanceAltitudeBoost = 0;
  state.lastSafeFlightProgress = state.flightProgress;
  state.lastSafePositionNed = null;
  state.collisionGuardTriggered = false;
  state.actualReactiveDecision = null;
}

function updateActualObstacleHelper(decision) {
  if (!state.missionScene) return;
  if (!decision?.collider) {
    if (state.actualObstacleHelper) state.actualObstacleHelper.visible = false;
    return;
  }
  if (!state.actualObstacleHelper) {
    state.actualObstacleHelper = new THREE.Box3Helper(decision.collider.displayBox, 0x10b981);
    state.missionScene.add(state.actualObstacleHelper);
  }
  state.actualObstacleHelper.box = decision.collider.displayBox;
  state.actualObstacleHelper.material.color.setHex(decision.threat_level === 'CRITICAL' ? 0xef4444 : (decision.threat_level === 'CAUTION' ? 0xf59e0b : 0x10b981));
  state.actualObstacleHelper.visible = true;
}

function updateLiveCollisionHUD(decision) {
  const alertEl = document.getElementById('traffic-alert-val');
  const clearanceEl = document.getElementById('val-clearance');
  const clearanceBadge = document.getElementById('badge-clearance');
  const supervisorBadge = document.getElementById('safety-supervisor-status');
  const emergency = state.collisionGuardTriggered;
  const caution = Boolean(decision?.collision_course);
  const clearance = Number(decision?.clearance_margin_m);

  if (alertEl) {
    alertEl.textContent = emergency ? 'EMERGENCY HOLD' : (caution ? 'TERRAIN CAUTION' : (state.airTrafficConflict ? 'TRAFFIC WARNING' : 'CLEAR'));
    alertEl.className = `t-stat-val ${emergency ? 'text-red' : ((caution || state.airTrafficConflict) ? 'text-amber' : 'text-emerald')}`;
  }
  if (clearanceEl && Number.isFinite(clearance)) {
    clearanceEl.textContent = `${clearance.toFixed(1)} m`;
    clearanceEl.className = `cell-number ${clearance < 3.5 ? 'text-red' : (clearance < 8 ? 'text-amber' : 'text-emerald')}`;
  }
  if (clearanceBadge) {
    clearanceBadge.textContent = emergency ? 'HOLD' : (caution ? 'MONITOR' : 'PASS');
    clearanceBadge.style.color = emergency ? 'var(--accent-red)' : (caution ? 'var(--accent-amber)' : 'var(--accent-emerald)');
    clearanceBadge.style.borderColor = emergency ? 'rgba(239,68,68,.6)' : (caution ? 'rgba(245,158,11,.55)' : 'rgba(16,185,129,.4)');
    clearanceBadge.style.background = emergency ? 'rgba(239,68,68,.14)' : (caution ? 'rgba(245,158,11,.12)' : 'rgba(16,185,129,.15)');
  }
  if (supervisorBadge) {
    supervisorBadge.textContent = emergency ? 'Collision hold active' : (caution ? 'Avoidance active' : 'All checks nominal');
    supervisorBadge.style.color = emergency ? 'var(--accent-red)' : (caution ? 'var(--accent-amber)' : 'var(--accent-emerald)');
  }
}

function updateReactivePerceptionHUD() {
  const dec = getEffectiveReactiveDecision();
  const box = document.getElementById('hud-building-box');
  const banner = document.getElementById('camera-evasion-banner');
  const bannerAction = document.getElementById('banner-action');
  const bannerPrim = document.getElementById('banner-primitive');
  const tagDist = document.getElementById('tag-building-dist');
  const tagType = document.getElementById('tag-building-type');
  const tagTtc = document.getElementById('tag-building-ttc');
  const tagRec = document.getElementById('tag-building-rec');
  const tagHazard = document.getElementById('tag-building-hazard');

  const mfLeft = document.getElementById('mf-left');
  const mfRight = document.getElementById('mf-right');
  const mfTop = document.getElementById('mf-top');

  if (box && dec && dec.obstacle_detected && dec.bbox) {
    box.style.display = 'flex';
    const b = dec.bbox || { top: 14, left: 16, width: 34, height: 60 };
    box.style.top = b.top + '%';
    box.style.left = b.left + '%';
    box.style.width = b.width + '%';
    box.style.height = b.height + '%';

    box.classList.remove('threat-caution', 'threat-critical', 'threat-clear');
    if (dec.threat_level === 'CRITICAL') {
      box.classList.add('threat-critical');
      if (banner) banner.classList.add('alert-critical');
    } else if (dec.threat_level === 'CAUTION') {
      box.classList.add('threat-caution');
      if (banner) banner.classList.remove('alert-critical');
    } else {
      box.classList.add('threat-clear');
      if (banner) banner.classList.remove('alert-critical');
    }

    if (tagType) tagType.textContent = `BUILDING ${Number(state.aiData?.models?.tactical_detector?.recall_pct || 96).toFixed(1)}%`;
    if (tagDist) tagDist.textContent = `DIST: ${Number.isFinite(dec.distance_m) ? dec.distance_m.toFixed(1) + 'm' : '—'}`;
    if (tagTtc) tagTtc.textContent = `TTC ${Number.isFinite(dec.ttc_s) ? dec.ttc_s.toFixed(1) + 's' : '∞'}`;
    if (tagRec) tagRec.textContent = dec.collision_course ? `EVADE: ${dec.optimal_path}` : 'TRACK ONLY';
    if (tagHazard) {
      tagHazard.textContent = dec.threat_level === 'CRITICAL' ? 'CRITICAL HAZARD' : (dec.threat_level === 'CAUTION' ? 'PROXIMITY HAZARD' : 'NOMINAL CLEAR');
    }

    if (bannerAction) bannerAction.textContent = dec.optimal_action || `OPTIMAL PATH: VEER ${dec.optimal_path || 'RIGHT'}`;
    if (bannerPrim) bannerPrim.textContent = `KINEMATIC PRIMITIVE: ${dec.optimal_primitive || 'BANK_RIGHT_EVADE'} (3.8 m/s)`;
  } else {
    if (box) box.style.display = 'none';
    if (banner) banner.classList.remove('alert-critical');
    if (bannerAction) bannerAction.textContent = 'FORWARD CORRIDOR CLEAR';
    if (bannerPrim) bannerPrim.textContent = 'GEOMETRY PROXIMITY · LIVE 20 Hz';
  }

  // Update Manifolds Status Bar
  if (dec && dec.candidate_paths) {
    dec.candidate_paths.forEach(cp => {
      let el = null;
      if (cp.direction === 'LEFT') el = mfLeft;
      else if (cp.direction === 'RIGHT') el = mfRight;
      else if (cp.direction === 'TOP') el = mfTop;

      if (el) {
        if (cp.status === 'OPTIMAL') el.className = 'manifold-item optimal';
        else el.className = 'manifold-item';
        const valEl = el.querySelector('.mf-val');
        const stEl = el.querySelector('.mf-status');
        if (valEl) valEl.textContent = cp.direction === 'TOP' ? `+${cp.vertical_offset_m || 8}m` : `${cp.clearance_m}m`;
        if (stEl) {
          stEl.className = 'mf-status ' + (cp.status === 'OPTIMAL' ? 'mf-optimal' : (cp.status === 'BLOCKED' ? 'mf-blocked' : 'mf-feasible'));
          stEl.textContent = cp.status === 'OPTIMAL' ? 'OPT' : (cp.status === 'BLOCKED' ? 'BLKD' : 'OK');
        }
      }
    });
  }
}

function renderCameraHUDOverlay() {
  const ctx = state.cameraCtx;
  const cvs = state.cameraCanvas;
  if (!ctx || !cvs) return;

  const w = cvs.width;
  const h = cvs.height;
  if (w === 0 || h === 0) return;

  ctx.clearRect(0, 0, w, h);

  const vpX = w * 0.50;
  const vpY = h * 0.50;
  const speed = state.realFlightSpeed !== undefined ? state.realFlightSpeed : 4.2;

  // 1. Dynamic Perspective Ground Scan & Optical Flow Grid
  const flowOffset = (Date.now() * 0.001 * Math.max(0.2, speed) * 0.5) % 1.0;
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.14)';
  ctx.lineWidth = 1;

  // Perspective longitudinal radials
  for (let r = -4; r <= 4; r++) {
    ctx.beginPath();
    ctx.moveTo(vpX, vpY);
    ctx.lineTo(vpX + r * (w * 0.18), h);
    ctx.stroke();
  }

  // Transverse terrain range markers moving towards drone
  for (let l = 0; l < 6; l++) {
    const prog = (l / 6 + flowOffset * (1 / 6)) % 1.0;
    const py = vpY + (prog * prog) * (h - vpY);
    const span = prog * (w * 0.65);
    ctx.beginPath();
    ctx.moveTo(vpX - span, py);
    ctx.lineTo(vpX + span, py);
    ctx.stroke();
  }
  ctx.restore();

  // 2. Artificial Horizon & Pitch Ladder Lines
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.28)';
  ctx.lineWidth = 1.2;
  // Left horizon wing
  ctx.beginPath();
  ctx.moveTo(w * 0.12, vpY);
  ctx.lineTo(w * 0.38, vpY);
  ctx.stroke();
  // Right horizon wing
  ctx.beginPath();
  ctx.moveTo(w * 0.62, vpY);
  ctx.lineTo(w * 0.88, vpY);
  ctx.stroke();

  // Center Flight Path Vector (FPV) reticle (Zero circles - Diamond)
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(vpX, vpY - 5);
  ctx.lineTo(vpX + 5, vpY);
  ctx.lineTo(vpX, vpY + 5);
  ctx.lineTo(vpX - 5, vpY);
  ctx.closePath();
  ctx.moveTo(vpX - 12, vpY); ctx.lineTo(vpX - 5, vpY);
  ctx.moveTo(vpX + 5, vpY); ctx.lineTo(vpX + 12, vpY);
  ctx.moveTo(vpX, vpY - 5); ctx.lineTo(vpX, vpY - 10);
  ctx.stroke();
  ctx.restore();

  const dec = getEffectiveReactiveDecision();
  if (!dec || !dec.obstacle_detected) return;

  // 3. 3D Perspective Wireframe Cage for the Approaching Building
  if (dec.bbox) {
    const b = dec.bbox;
    const bx = (b.left / 100) * w;
    const by = (b.top / 100) * h;
    const bw = (b.width / 100) * w;
    const bh = (b.height / 100) * h;

    const depthScale = 0.68;
    const rbx = vpX + (bx - vpX) * depthScale;
    const rby = vpY + (by - vpY) * depthScale;
    const rbw = bw * depthScale;
    const rbh = bh * depthScale;

    const isCrit = dec.threat_level === 'CRITICAL';
    const wireColor = isCrit ? 'rgba(239, 68, 68, 0.55)' : 'rgba(245, 158, 11, 0.45)';

    ctx.save();
    ctx.strokeStyle = wireColor;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    // Rear face of building
    ctx.strokeRect(rbx, rby, rbw, rbh);

    // 4 Corner Depth Perspective Lines connecting front face to rear face
    ctx.beginPath();
    ctx.moveTo(bx, by); ctx.lineTo(rbx, rby);
    ctx.moveTo(bx + bw, by); ctx.lineTo(rbx + rbw, rby);
    ctx.moveTo(bx, by + bh); ctx.lineTo(rbx, rby + rbh);
    ctx.moveTo(bx + bw, by + bh); ctx.lineTo(rbx + rbw, rby + rbh);
    ctx.stroke();
    ctx.restore();
  }

  // 4. Glowing Evasion Trajectory Curve with Flowing Chevrons
  const t = Date.now() * 0.0025;
  const opt = dec.optimal_path || 'RIGHT';

  let startX = w * 0.50;
  let startY = h * 0.98;
  let endX, endY, cpX, cpY;

  if (opt === 'RIGHT') {
    endX = w * 0.84;
    endY = h * 0.38;
    cpX = w * 0.62;
    cpY = h * 0.72;
  } else if (opt === 'LEFT') {
    endX = w * 0.16;
    endY = h * 0.38;
    cpX = w * 0.38;
    cpY = h * 0.72;
  } else {
    // TOP
    endX = w * 0.50;
    endY = h * 0.16;
    cpX = w * 0.50;
    cpY = h * 0.54;
  }

  // Glowing Evasion Trajectory
  ctx.save();
  ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.quadraticCurveTo(cpX, cpY, endX, endY);
  ctx.stroke();

  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.quadraticCurveTo(cpX, cpY, endX, endY);
  ctx.stroke();

  // Animated Flowing Chevrons along curve
  const chevronCount = 4;
  for (let c = 0; c < chevronCount; c++) {
    const frac = ((t * 0.8 + c / chevronCount) % 1.0);
    const u = frac;
    const px = (1 - u) * (1 - u) * startX + 2 * (1 - u) * u * cpX + u * u * endX;
    const py = (1 - u) * (1 - u) * startY + 2 * (1 - u) * u * cpY + u * u * endY;

    const dx = 2 * (1 - u) * (cpX - startX) + 2 * u * (endX - cpX);
    const dy = 2 * (1 - u) * (cpY - startY) + 2 * u * (endY - cpY);
    const angle = Math.atan2(dy, dx);

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(angle);
    ctx.fillStyle = '#f1f5f9';
    ctx.shadowColor = '#10b981';
    ctx.shadowBlur = 6;

    ctx.beginPath();
    ctx.moveTo(3.5, 0);
    ctx.lineTo(-3.5, -3);
    ctx.lineTo(-1.8, 0);
    ctx.lineTo(-3.5, 3);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Evasion Gateway Target Reticle (Zero circles - Rotating Diamond & Square)
  ctx.save();
  ctx.translate(endX, endY);
  ctx.rotate(t * 1.5);

  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 1.3;
  ctx.setLineDash([3, 2.5]);
  // Outer Diamond
  ctx.beginPath();
  ctx.moveTo(0, -9);
  ctx.lineTo(9, 0);
  ctx.lineTo(0, 9);
  ctx.lineTo(-9, 0);
  ctx.closePath();
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.1;
  ctx.strokeRect(-4, -4, 8, 8);

  ctx.fillStyle = '#10b981';
  ctx.fillRect(-1.5, -1.5, 3, 3);

  ctx.rotate(-t * 1.5); // restore rotation for text
  ctx.font = '7px "JetBrains Mono", monospace';
  ctx.fillStyle = '#10b981';
  ctx.textAlign = 'center';
  ctx.fillText('OPTIMAL ESCAPE', 0, -13);
  ctx.restore();

  ctx.restore();
}

function initEventLogTabs() {

  const tabs = document.querySelectorAll('.ev-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.eventFilter = tab.getAttribute('data-filter');
      filterEventTable();
    });
  });
}

function filterEventTable() {
  const rows = document.querySelectorAll('#event-table-body tr');
  let count = 0;
  rows.forEach(r => {
    const type = r.getAttribute('data-type');
    if (state.eventFilter === 'all' ||
       (state.eventFilter === 'warnings' && type === 'warning') ||
       (state.eventFilter === 'events' && type === 'info')) {
      r.style.display = '';
      count++;
    } else {
      r.style.display = 'none';
    }
  });
  const badge = document.getElementById('ev-counter');
  if (badge) badge.textContent = `${count} events`;
}

function initTelemetryAndMissionControls() {
  // 1. Telemetry Tabs (Traffic, Compute, Power)
  const tabs = document.querySelectorAll('.telem-tab');
  const panels = document.querySelectorAll('.telem-panel');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const targetPanel = document.getElementById(`panel-${tab.dataset.tab}`);
      if (targetPanel) targetPanel.classList.add('active');
    });
  });

  // 2. Air Traffic Radius Chips
  const radiusChips = document.querySelectorAll('#traffic-radius-chips .telem-mini-chip');
  radiusChips.forEach(chip => {
    chip.addEventListener('click', () => {
      radiusChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.trafficRadiusKm = parseFloat(chip.dataset.radius);
      const disp = document.getElementById('disp-traffic-radius');
      if (disp) disp.textContent = `${state.trafficRadiusKm.toFixed(1)} km`;
      updateAirTrafficMetrics();
      update3DTrafficDome();
    });
  });

  // 3. Air Traffic Density Chips
  const levelChips = document.querySelectorAll('#traffic-level-chips .telem-mini-chip');
  levelChips.forEach(chip => {
    chip.addEventListener('click', () => {
      levelChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.trafficDensityLevel = chip.dataset.level;
      const disp = document.getElementById('disp-traffic-level');
      const nameMap = {
        very_low: 'Very Low',
        low: 'Low',
        medium: 'Medium',
        high: 'High',
        very_high: 'Very High'
      };
      if (disp) disp.textContent = nameMap[state.trafficDensityLevel] || 'Medium';
      updateAirTrafficMetrics();
    });
  });

  // 4. Power / Battery Slider
  const batterySlider = document.getElementById('slider-battery');
  if (batterySlider) {
    batterySlider.addEventListener('input', (e) => {
      state.batterySocPct = parseInt(e.target.value, 10);
      updateBatteryPowerFailsafe();
    });
  }

  // 5. Waypoint Chips (WP 1 to WP 10, GOAL)
  const wpChips = document.querySelectorAll('.wp-chip');
  wpChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const wpIdx = parseInt(chip.dataset.wp, 10);
      jumpToWaypoint(wpIdx);
    });
  });

  // 6. Flight Speed Slider
  const speedSlider = document.getElementById('slider-flight-speed');
  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      state.realFlightSpeed = parseFloat(e.target.value);
      const disp = document.getElementById('disp-slider-speed');
      if (disp) {
        disp.textContent = state.realFlightSpeed <= 0.05 ? '0.0 m/s (Stop)' : `${state.realFlightSpeed.toFixed(1)} m/s`;
      }
    });
  }

  // 7. Cruise Altitude Slider
  const altSlider = document.getElementById('slider-flight-alt');
  if (altSlider) {
    altSlider.addEventListener('input', (e) => {
      state.cruiseAltitude = parseFloat(e.target.value);
      const disp = document.getElementById('disp-slider-alt');
      if (disp) disp.textContent = `${state.cruiseAltitude.toFixed(1)} m`;
    });
  }

  // 8. Checkpoint & Dynamic Destination Controls (Path A ➔ Path B)
  initDestinationNavigationControls();

  updateAirTrafficMetrics();
  updateBatteryPowerFailsafe();
}

// =============================================================================
// CHECKPOINT & DYNAMIC DESTINATION CONTROLLER (PATH A ➔ PATH B)
// =============================================================================
function initDestinationNavigationControls() {
  // 1. Preset Checkpoint Chips (Alpha, Bravo, Charlie, Echo, Foxtrot, Golf)
  const presetChips = document.querySelectorAll('.cp-preset-chip');
  presetChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      const gx = parseFloat(chip.dataset.x);
      const gy = parseFloat(chip.dataset.y);
      const name = chip.dataset.name || chip.textContent;
      const alt = state.cruiseAltitude || 6.0;
      selectTargetCandidate(gx, gy, alt, name, chip);
    });
  });

  // 2. Interactive 3D Canvas Raycasting with Drag/Click Distinction (OrbitControls NEVER locked)
  const canvas = state.missionCanvas || document.getElementById('mission-canvas');
  if (canvas) {
    state.missionCanvas = canvas;
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let pointerDownInfo = null;
    let lastPickTime = 0;

    function processCanvasPick(clientX, clientY) {
      if (!state.missionCamera || !canvas) return;
      if (performance.now() - lastPickTime < 200) return; // Debounce
      lastPickTime = performance.now();

      const rect = canvas.getBoundingClientRect();
      mouse.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, state.missionCamera);

      let hitPoint = null;
      if (state.missionWorldGroup) {
        const intersects = raycaster.intersectObjects(state.missionWorldGroup.children, true);
        if (intersects.length > 0) {
          hitPoint = intersects[0].point;
        }
      }

      if (!hitPoint) {
        const groundPlane1 = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
        const groundPlane2 = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
        const targetPt = new THREE.Vector3();
        if (raycaster.ray.intersectPlane(groundPlane1, targetPt) || raycaster.ray.intersectPlane(groundPlane2, targetPt)) {
          hitPoint = targetPt;
        }
      }

      if (hitPoint) {
        // glTF X = East (y) -> East = hitPoint.x
        // glTF Z = -North (-x) -> North = -hitPoint.z
        const northX = Math.round(-hitPoint.z);
        const eastY = Math.round(hitPoint.x);
        const clampedNorth = Math.max(-390, Math.min(390, northX));
        const clampedEast = Math.max(-490, Math.min(490, eastY));
        const altZ = state.cruiseAltitude || 6.0;

        selectTargetCandidate(clampedNorth, clampedEast, altZ, `Pinpoint [${clampedNorth}, ${clampedEast}]`, null);
      }
    }
    window.processCanvasPick = processCanvasPick;

    state.missionCanvas.addEventListener('pointerdown', (event) => {
      if (event.button === 0) {
        pointerDownInfo = {
          x: event.clientX,
          y: event.clientY,
          time: performance.now()
        };
      }
    });

    state.missionCanvas.addEventListener('pointerup', (event) => {
      if (!pointerDownInfo || event.button !== 0) return;

      const dx = event.clientX - pointerDownInfo.x;
      const dy = event.clientY - pointerDownInfo.y;
      const dt = performance.now() - pointerDownInfo.time;
      pointerDownInfo = null;

      // If user moved > 6px or held > 450ms, it is a camera orbit/pan drag - do not click
      if (Math.hypot(dx, dy) <= 6 && dt <= 450) {
        processCanvasPick(event.clientX, event.clientY);
      }
    });

    state.missionCanvas.addEventListener('click', (event) => {
      if (event.button === 0) {
        processCanvasPick(event.clientX, event.clientY);
      }
    });
  }

  // 3. Target Decision Card Action Buttons (FLY or CANCEL)
  const btnCancel = document.getElementById('btn-cancel-target');
  if (btnCancel) {
    btnCancel.addEventListener('click', (e) => {
      e.stopPropagation();
      const card = document.getElementById('target-decision-card');
      if (card) card.style.display = 'none';
      if (state.pendingTargetBeacon3D) state.pendingTargetBeacon3D.visible = false;
      if (state.pendingDirectLine3D) state.pendingDirectLine3D.visible = false;
      state.pendingTarget = null;
    });
  }

  const btnFly = document.getElementById('btn-confirm-fly');
  if (btnFly) {
    btnFly.addEventListener('click', (e) => {
      e.stopPropagation();
      const card = document.getElementById('target-decision-card');
      if (card) card.style.display = 'none';
      if (!state.pendingTarget) return;

      const tgt = state.pendingTarget;
      if (state.pendingTargetBeacon3D) state.pendingTargetBeacon3D.visible = false;
      if (state.pendingDirectLine3D) state.pendingDirectLine3D.visible = false;

      // Launch drone from CURRENT position
      const curPos = (state.traceData && state.traceData[state.traceIndex])
        ? state.traceData[state.traceIndex].position
        : state.startNed;

      planAndFlyRoute(
        curPos[0],
        curPos[1],
        tgt.ned[0],
        tgt.ned[1],
        -tgt.ned[2],
        tgt.name
      );

      state.pendingTarget = null;
    });
  }
}

function selectTargetCandidate(gx, gy, gz, name, chip) {
  state.pendingTarget = {
    ned: [gx, gy, -gz],
    name: name || `Target [${gx.toFixed(0)}, ${gy.toFixed(0)}]`
  };

  // 1. Highlight preset chip or deselect all if custom map click
  const presetChips = document.querySelectorAll('.cp-preset-chip');
  presetChips.forEach(c => {
    if (chip && c === chip) c.classList.add('active');
    else c.classList.remove('active');
  });

  // 2. Distance and Flight ETA from current drone location
  const curPos = (state.traceData && state.traceData[state.traceIndex])
    ? state.traceData[state.traceIndex].position
    : state.startNed;
  const distM = Math.hypot(gx - curPos[0], gy - curPos[1]);
  const speed = Math.max(0.5, state.realFlightSpeed || 4.0);
  const flightSec = distM / speed;
  const etaStr = flightSec >= 60 ? `~${(flightSec / 60).toFixed(1)} min` : `~${Math.round(flightSec)} s`;
  const distStr = distM >= 1000 ? `${(distM / 1000).toFixed(2)} km` : `${Math.round(distM)} m`;

  // 3. Populate Target Decision Card
  const card = document.getElementById('target-decision-card');
  const titleEl = document.getElementById('dec-title');
  const coordsEl = document.getElementById('dec-coords');
  const distEl = document.getElementById('dec-dist');
  const etaEl = document.getElementById('dec-eta');

  if (titleEl) titleEl.textContent = state.pendingTarget.name;
  if (coordsEl) coordsEl.textContent = `[N: ${gx.toFixed(1)}, E: ${gy.toFixed(1)}, Alt: ${gz.toFixed(1)}m]`;
  if (distEl) distEl.textContent = distStr;
  if (etaEl) etaEl.textContent = etaStr;

  if (card) {
    card.style.display = 'flex';
  }

  // 4. Show 3D Target Marker & Guideline
  showPendingTarget3D(gx, gy, gz);
}
window.selectTargetCandidate = selectTargetCandidate;

function showPendingTarget3D(gx, gy, gz) {
  if (!state.missionScene) return;

  const targetGltf = nedToGltf(gx, gy, -gz);

  if (!state.pendingTargetBeacon3D) {
    const group = new THREE.Group();

    // Vertical light pillar: Square Rectilinear Beam (Zero circles)
    const pillarGeo = new THREE.BoxGeometry(1.5, 55, 1.5);
    const pillarMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.7 });
    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
    pillar.position.y = 27.5;
    group.add(pillar);

    // Floating Diamond Beacon: Sharp 8-faceted Octahedron (Zero circles)
    const diamondGeo = new THREE.OctahedronGeometry(8.5, 0);
    const diamondMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xf59e0b,
      emissiveIntensity: 0.9,
      metalness: 0.2,
      roughness: 0.2
    });
    const diamond = new THREE.Mesh(diamondGeo, diamondMat);
    diamond.position.y = 55;
    diamond.name = 'pendingDiamond';
    group.add(diamond);

    state.pendingTargetBeacon3D = group;
    state.missionScene.add(group);
  }

  state.pendingTargetBeacon3D.position.copy(targetGltf);
  state.pendingTargetBeacon3D.visible = true;

  // Direct dashed guideline from current drone position to candidate
  const curPos = (state.traceData && state.traceData[state.traceIndex])
    ? state.traceData[state.traceIndex].position
    : state.startNed;
  const curGltf = nedToGltf(curPos[0], curPos[1], -gz);

  if (!state.pendingDirectLine3D) {
    const directGeo = new THREE.BufferGeometry().setFromPoints([curGltf, targetGltf]);
    const directMat = new THREE.LineDashedMaterial({
      color: 0xf59e0b,
      dashSize: 10,
      gapSize: 6
    });
    state.pendingDirectLine3D = new THREE.Line(directGeo, directMat);
    state.pendingDirectLine3D.computeLineDistances();
    state.missionScene.add(state.pendingDirectLine3D);
  } else {
    state.pendingDirectLine3D.geometry.dispose();
    state.pendingDirectLine3D.geometry = new THREE.BufferGeometry().setFromPoints([curGltf, targetGltf]);
    state.pendingDirectLine3D.computeLineDistances();
    state.pendingDirectLine3D.visible = true;
  }
}

async function planAndFlyRoute(startX, startY, goalX, goalY, alt, name) {
  // Update Destination state
  state.startNed = [startX, startY, -alt];
  state.destination = {
    name: name || `Target [${goalX.toFixed(0)}, ${goalY.toFixed(0)}]`,
    ned: [goalX, goalY, -alt]
  };

  const nameEl = document.getElementById('disp-dest-name');
  const coordsEl = document.getElementById('disp-dest-coords');
  if (nameEl) nameEl.textContent = state.destination.name;
  if (coordsEl) coordsEl.textContent = `[${goalX.toFixed(1)}, ${goalY.toFixed(1)}]`;

  try {
    const res = await fetch(`/api/plan-route?start_x=${startX}&start_y=${startY}&goal_x=${goalX}&goal_y=${goalY}&altitude=${alt}`);
    if (res.ok) {
      const data = await res.json();
      if (data && data.points && data.points.length > 0) {
        state.traceData = data.points;
        state.totalPoints = data.points.length;
        state.totalRouteDistanceM = data.total_distance_m || 1000.0;
        state.traceIndex = 0;
        state.flightProgress = 0.0;
        state.isHoveringAtGoal = false;
        state.isPlaying = true;
        resetCollisionAvoidanceState();
        state.currentRouteWaypoints = data.waypoints || [];

        // 1. Update 3D Polyline Route and Beacons
        updateRoute3DVisuals(state.currentRouteWaypoints, state.startNed, state.destination.ned);

        // 2. Update HUD metrics
        const distEl = document.getElementById('disp-dest-dist');
        const etaEl = document.getElementById('disp-dest-eta');
        const primEl = document.getElementById('disp-dest-primitive');

        const distStr = data.total_distance_m >= 1000
          ? `${(data.total_distance_m / 1000).toFixed(2)} km`
          : `${Math.round(data.total_distance_m)} m`;
        const etaStr = data.estimated_flight_time_s >= 60
          ? `~${(data.estimated_flight_time_s / 60).toFixed(1)} min`
          : `~${Math.round(data.estimated_flight_time_s)} s`;

        if (distEl) distEl.textContent = distStr;
        if (etaEl) etaEl.textContent = etaStr;
        if (primEl && data.points[0]) primEl.textContent = data.points[0].primitive;

        // 3. Rebuild bottom scorecard waypoint chips
        rebuildWaypointChips(state.currentRouteWaypoints);

        // 4. Update Scrubber
        const scrubber = document.getElementById('trace-scrubber');
        if (scrubber) {
          scrubber.max = state.totalPoints - 1;
          scrubber.value = 0;
        }

        // 5. Update Battery & Power status for new destination
        updateBatteryPowerFailsafe();

        console.log(`[AI Route Planner] Route computed to ${state.destination.name}: ${data.points.length} trajectory points, ${state.currentRouteWaypoints.length} waypoints, ${data.total_distance_m}m.`);
      }
    } else {
      const error = await res.json().catch(() => ({ error: 'Planner request failed' }));
      state.isPlaying = false;
      const primitive = document.getElementById('disp-dest-primitive');
      if (primitive) primitive.textContent = 'PLANNER_OFFLINE · HOLD';
      console.error('Route rejected fail-closed:', error.error || res.statusText);
    }
  } catch (err) {
    console.error('Error planning AI route:', err);
  }
}
window.planAndFlyRoute = planAndFlyRoute;

function rebuildWaypointChips(waypointsNED) {
  const container = document.getElementById('waypoint-chips-container');
  if (!container || !waypointsNED || waypointsNED.length === 0) return;

  container.innerHTML = '';
  waypointsNED.forEach((wp, idx) => {
    const btn = document.createElement('button');
    btn.className = `wp-chip font-mono ${idx === 0 ? 'active' : ''}`;
    btn.dataset.wp = idx;
    btn.textContent = idx === 0 ? 'START' : (idx === waypointsNED.length - 1 ? 'GOAL' : `WP ${idx + 1}`);
    btn.addEventListener('click', () => {
      jumpToWaypointDynamic(idx, waypointsNED.length);
    });
    container.appendChild(btn);
  });

  const badge = document.getElementById('disp-active-wp');
  if (badge) badge.textContent = 'WP 1 (START)';
}

function jumpToWaypointDynamic(wpIdx, totalWps) {
  state.activeWaypointIndex = wpIdx;
  const wpChips = document.querySelectorAll('.wp-chip');
  wpChips.forEach((c, i) => {
    if (i === wpIdx) {
      c.classList.add('active');
      c.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    } else {
      c.classList.remove('active');
    }
  });

  const badge = document.getElementById('disp-active-wp');
  if (badge) {
    badge.textContent = wpIdx === 0 ? 'WP 1 (START)' : (wpIdx === totalWps - 1 ? 'GOAL (DEST B)' : `WP ${wpIdx + 1}`);
  }

  if (state.totalPoints > 0) {
    let targetIdx = -1;
    if (Array.isArray(state.traceData)) {
      targetIdx = state.traceData.findIndex(p => p.waypoint_index === wpIdx);
    }
    if (targetIdx < 0) {
      const stepPerWp = Math.floor(state.totalPoints / Math.max(1, totalWps - 1));
      targetIdx = Math.min(state.totalPoints - 1, wpIdx * stepPerWp);
    }
    state.traceIndex = Math.min(state.totalPoints - 1, targetIdx);
    state.flightProgress = state.traceIndex;
    state.isHoveringAtGoal = (wpIdx === totalWps - 1);
    updatePlaybackHUD();
    updateBatteryPowerFailsafe();
  }
}

function updateAirTrafficMetrics() {
  const densityFactors = { very_low: 0.35, low: 0.65, medium: 1.0, high: 1.75, very_high: 2.8 };
  const factor = densityFactors[state.trafficDensityLevel] || 1.0;
  const count = Math.max(1, Math.round((state.trafficRadiusKm * 4.5) * factor));
  const minSep = Math.max(140, Math.round(1500 / Math.max(1, Math.sqrt(count) * 0.85)));
  const risk = Math.min(0.92, (count / (state.trafficRadiusKm * 18)).toFixed(2));

  const countEl = document.getElementById('traffic-count-val');
  const sepEl = document.getElementById('traffic-sep-val');
  const riskEl = document.getElementById('traffic-risk-val');
  const alertEl = document.getElementById('traffic-alert-val');

  if (countEl) countEl.textContent = `${count} acft`;
  if (sepEl) {
    sepEl.textContent = `${minSep} m`;
    sepEl.className = 't-stat-val ' + (minSep < 250 ? 'text-red' : (minSep < 500 ? 'text-amber' : 'text-emerald'));
  }
  if (riskEl) {
    const riskLabel = risk < 0.20 ? 'LOW' : (risk < 0.45 ? 'MODERATE' : 'CRITICAL');
    riskEl.textContent = `${riskLabel} (${risk})`;
    riskEl.className = 't-stat-val ' + (risk > 0.45 ? 'text-red' : (risk > 0.20 ? 'text-amber' : 'text-emerald'));
  }
  if (alertEl) {
    const isConflict = minSep < 300 || risk > 0.5;
    state.airTrafficConflict = isConflict;
    alertEl.textContent = isConflict ? 'WARNING' : 'CLEAR';
    state.missionScene.remove(state.trafficDome3D);
    state.trafficDome3D = null;
  }

  const domeRadius = Math.min(320, Math.max(35, state.trafficRadiusKm * 32));
  const domeGeo = new THREE.CylinderGeometry(domeRadius, domeRadius, 55, 32, 1, true);
  const domeMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    wireframe: true,
    transparent: true,
    opacity: 0.22
  });
  state.trafficDome3D = new THREE.Mesh(domeGeo, domeMat);
  state.trafficDome3D.position.set(0, 28, 0);
  state.missionScene.add(state.trafficDome3D);
}

function update3DTrafficDome() {
  updateAirTrafficMetrics();
}
window.update3DTrafficDome = update3DTrafficDome;

function setMissionDroneScale(multiplier) {
  const scale = Math.max(0.2, Math.min(10.0, parseFloat(multiplier) || 1.0));
  state.droneScale = scale;
  if (state.missionDroneGroup) {
    state.missionDroneGroup.scale.set(scale, scale, scale);
  }
  const slider = document.getElementById('slider-drone-scale');
  if (slider) slider.value = scale;
  const badge = document.getElementById('drone-scale-badge');
  if (badge) badge.textContent = `${scale.toFixed(1)}x`;
  const quickText = document.getElementById('quick-drone-size-text');
  if (quickText) quickText.textContent = `${scale.toFixed(1)}x`;

  const chips = document.querySelectorAll('.drone-scale-chip');
  chips.forEach(chip => {
    const s = parseFloat(chip.dataset.scale);
    if (Math.abs(s - scale) < 0.05) chip.classList.add('active');
    else chip.classList.remove('active');
  });
  try {
    localStorage.setItem('flytech_drone_scale', scale.toString());
  } catch (e) {}
}
window.setMissionDroneScale = setMissionDroneScale;

function updateBatteryPowerFailsafe() {
  const progressRatio = state.totalPoints > 0 ? (state.traceIndex / state.totalPoints) : 0;
  const totalDistKm = 2.45;
  const distRemainingKm = Math.max(0.08, totalDistKm * (1.0 - progressRatio));
  const maxRangeKm = (state.batterySocPct / 100) * 6.0; // 6 km range at 100%
  const requiredPct = Math.round((distRemainingKm / 6.0) * 100 + 12);

  const dispPct = document.getElementById('disp-battery-pct');
  if (dispPct) {
    dispPct.textContent = `${state.batterySocPct}%`;
    dispPct.style.color = state.batterySocPct > 35 ? '#10b981' : (state.batterySocPct > 20 ? '#f59e0b' : '#ef4444');
  }

  const isReachable = state.batterySocPct >= requiredPct && state.batterySocPct > 18;
  const banner = document.getElementById('power-status-banner');
  const bannerStatus = document.getElementById('power-banner-status');
  const bannerDetail = document.getElementById('power-banner-detail');
  const bannerIcon = document.getElementById('power-banner-icon');
  const siteTag = document.getElementById('disp-selected-site');
  const siteRow1 = document.getElementById('site-row-1');

  if (isReachable) {
    state.isDivertEngaged = false;
    if (banner) banner.className = 'power-status-banner safe font-mono';
    if (bannerIcon) bannerIcon.textContent = '⚡';
    if (bannerStatus) bannerStatus.textContent = 'SUFFICIENT POWER: GOAL REACHABLE';
    if (bannerDetail) bannerDetail.textContent = `Reserve: +${(maxRangeKm - distRemainingKm).toFixed(2)} km surplus to destination`;
    if (siteTag) {
      siteTag.textContent = 'STANDBY';
      siteTag.className = 's-hdr-rec font-mono text-cyan';
    }
    if (siteRow1) siteRow1.classList.remove('divert-engaged');
    hide3DEmergencyLanding();
  } else {
    state.isDivertEngaged = true;
    if (banner) banner.className = 'power-status-banner danger font-mono';
    if (bannerIcon) bannerIcon.textContent = '⚠️';
    if (bannerStatus) bannerStatus.textContent = 'EMERGENCY DIVERT: BATTERY CRITICAL';
    if (bannerDetail) bannerDetail.textContent = `Deficit: -${Math.abs(distRemainingKm - maxRangeKm).toFixed(2)} km. Diverting to Highway Strip A-1.`;
    if (siteTag) {
      siteTag.textContent = 'DIVERT ENGAGED ➔ STRIP A-1';
      siteTag.className = 's-hdr-rec font-mono text-red';
    }
    if (siteRow1) siteRow1.classList.add('divert-engaged');
    show3DEmergencyLanding();
  }
}

function show3DEmergencyLanding() {
  if (state.emergency3DGroup || !state.missionScene) return;
  const group = new THREE.Group();

  // 1. Landing Helipad Decal
  const padGeo = new THREE.CylinderGeometry(20, 20, 1.2, 32);
  const padMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0xef4444,
    emissiveIntensity: 0.45,
    roughness: 0.5
  });
  const pad = new THREE.Mesh(padGeo, padMat);
  const padPos = nedToGltf(-280, -250, 0);
  pad.position.set(padPos.x, 2, padPos.z);
  group.add(pad);

  // 2. High-intensity Emergency Strobe Beacon
  const beacon = create3DBeacon(0xef4444, 'DIVERT');
  beacon.position.set(padPos.x, 0, padPos.z);
  group.add(beacon);

  // 3. Glowing Amber/Red Diversion Path Line from current drone position to pad
  const curPos = state.missionDroneGroup ? state.missionDroneGroup.position : new THREE.Vector3(-180, 20, 200);
  const divertCurve = new THREE.CatmullRomCurve3([
    curPos.clone(),
    new THREE.Vector3((curPos.x + padPos.x) * 0.5, 30, (curPos.z + padPos.z) * 0.5),
    new THREE.Vector3(padPos.x, 2, padPos.z)
  ]);
  const divertGeo = new THREE.BufferGeometry().setFromPoints(divertCurve.getPoints(25));
  const divertMat = new THREE.LineDashedMaterial({
    color: 0xef4444,
    dashSize: 8,
    gapSize: 4
  });
  const divertLine = new THREE.Line(divertGeo, divertMat);
  divertLine.computeLineDistances();
  group.add(divertLine);

  state.emergency3DGroup = group;
  state.missionScene.add(group);
}

function hide3DEmergencyLanding() {
  if (state.emergency3DGroup && state.missionScene) {
    state.missionScene.remove(state.emergency3DGroup);
    state.emergency3DGroup = null;
  }
}

function jumpToWaypoint(wpIdx) {
  state.activeWaypointIndex = wpIdx;
  const wpChips = document.querySelectorAll('.wp-chip');
  wpChips.forEach((c, i) => {
    if (i === wpIdx) c.classList.add('active');
    else c.classList.remove('active');
  });

  const badge = document.getElementById('disp-active-wp');
  if (badge) {
    badge.textContent = wpIdx === 0 ? 'WP 1 (START)' : (wpIdx === 10 ? 'WP 10 (GOAL)' : `WP ${wpIdx + 1}`);
  }

  if (state.totalPoints > 0) {
    const stepPerWp = Math.floor(state.totalPoints / 10);
    state.traceIndex = Math.min(state.totalPoints - 1, wpIdx * stepPerWp);
    updatePlaybackHUD();
    updateBatteryPowerFailsafe();
  }
}

function updateBackendTelemetryHUD(aiData) {
  if (!aiData) return;
  if (aiData.compute_memory) {
    const mem = aiData.compute_memory;
    const ramEl = document.getElementById('compute-ram-val');
    const cpuEl = document.getElementById('compute-cpu-val');
    const nodesEl = document.getElementById('compute-nodes-val');
    const replanEl = document.getElementById('compute-replan-val');
    const cacheEl = document.getElementById('compute-cache-pct');
    const fillEl = document.getElementById('compute-cache-fill');
    const trtEl = document.getElementById('compute-trt-val');

    if (ramEl) ramEl.textContent = `${mem.ram_footprint_mb.toFixed(1)} MB`;
    if (cpuEl) cpuEl.textContent = `${mem.cpu_search_load_pct.toFixed(1)}%`;
    if (nodesEl) nodesEl.textContent = `${mem.a_star_nodes_expanded_per_sec.toLocaleString()} /s`;
    if (replanEl) replanEl.textContent = `${mem.replan_latency_ms} ms`;
    if (cacheEl) cacheEl.textContent = `${mem.frontier_cache_pct.toFixed(0)}%`;
    if (fillEl) fillEl.style.width = `${mem.frontier_cache_pct}%`;
    if (trtEl) trtEl.textContent = `${mem.tensorrt_fps.toFixed(1)} FPS`;
  }
}

function initMapModeAndSettingsControls() {
  // 1. Header Settings Trigger Button & Floating Dropdown
  const btnSettings = document.getElementById('btn-settings-menu');
  const dropdown = document.getElementById('settings-menu-dropdown');
  const btnClose = document.getElementById('btn-close-settings');
  const btnQuick = document.getElementById('btn-quick-settings');

  function toggleSettingsDropdown(e) {
    if (e) e.stopPropagation();
    if (!dropdown) return;
    const isHidden = (dropdown.style.display === 'none' || !dropdown.style.display);
    dropdown.style.display = isHidden ? 'flex' : 'none';
  }

  function closeSettingsDropdown() {
    if (dropdown) dropdown.style.display = 'none';
  }

  if (btnSettings) {
    btnSettings.addEventListener('click', toggleSettingsDropdown);
  }
  if (btnClose) {
    btnClose.addEventListener('click', (e) => {
      e.stopPropagation();
      closeSettingsDropdown();
    });
  }
  if (btnQuick) {
    btnQuick.addEventListener('click', toggleSettingsDropdown);
  }

  // Close dropdown when clicking outside
  window.addEventListener('click', (e) => {
    if (!dropdown || dropdown.style.display === 'none') return;
    const container = document.querySelector('.settings-menu-container');
    if (container && !container.contains(e.target) && e.target !== btnQuick) {
      closeSettingsDropdown();
    }
  });

  // 2. Quick Mode Buttons in Mission Header
  const quickModeBtns = document.querySelectorAll('#map-mode-quick-switch .switch-btn[data-mode]');
  quickModeBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const mode = btn.dataset.mode;
      if (mode) applyMapMode(mode);
    });
  });

  // 3. Mode Cards in Top-Right Settings Dropdown
  const modeCards = document.querySelectorAll('#settings-menu-dropdown .mode-card[data-mode]');
  modeCards.forEach(card => {
    card.addEventListener('click', (e) => {
      e.stopPropagation();
      const mode = card.dataset.mode;
      if (mode) applyMapMode(mode);
    });
  });

  // 4. Map Layer Toggles (ON / OFF switches)
  const layerIds = ['buildings', 'restricted', 'route', 'directline', 'trafficdome', 'labels'];
  const layerKeyMap = {
    buildings: 'buildings',
    restricted: 'restricted',
    route: 'route',
    directline: 'directLine',
    trafficdome: 'trafficDome',
    labels: 'labels'
  };

  layerIds.forEach(id => {
    const input = document.getElementById(`toggle-layer-${id}`);
    if (input) {
      const layerKey = layerKeyMap[id];
      input.addEventListener('change', (e) => {
        setLayerVisibility(layerKey, e.target.checked);
      });
    }
  });

  // 5. Camera Perspective Buttons (3D Orbit vs 2D Nadir)
  const btnCam3d = document.getElementById('btn-cam-3d');
  const btnCam2d = document.getElementById('btn-cam-2d');
  if (btnCam3d) {
    btnCam3d.addEventListener('click', (e) => {
      e.stopPropagation();
      setCameraPerspective('3d');
    });
  }
  if (btnCam2d) {
    btnCam2d.addEventListener('click', (e) => {
      e.stopPropagation();
      setCameraPerspective('2d');
    });
  }

  // 6. Drone Visual Scale Controls (Settings Slider + Presets + Quick Steppers)
  const scaleSlider = document.getElementById('slider-drone-scale');
  if (scaleSlider) {
    scaleSlider.addEventListener('input', (e) => {
      setMissionDroneScale(e.target.value);
    });
  }

  const scaleChips = document.querySelectorAll('.drone-scale-chip');
  scaleChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      setMissionDroneScale(chip.dataset.scale);
    });
  });

  const btnScaleDown = document.getElementById('btn-drone-size-down');
  const btnScaleUp = document.getElementById('btn-drone-size-up');
  if (btnScaleDown) {
    btnScaleDown.addEventListener('click', (e) => {
      e.stopPropagation();
      setMissionDroneScale(Math.max(0.5, (state.droneScale || 1.0) - 0.5));
    });
  }
  if (btnScaleUp) {
    btnScaleUp.addEventListener('click', (e) => {
      e.stopPropagation();
      setMissionDroneScale(Math.min(5.0, (state.droneScale || 1.0) + 0.5));
    });
  }

  try {
    const savedScale = parseFloat(localStorage.getItem('flytech_drone_scale'));
    if (!isNaN(savedScale) && savedScale > 0) {
      setMissionDroneScale(savedScale);
    }
  } catch (e) {}
}

function renderSynchronizedSensors(timestamp) {
  if (!state.sensorPose || !state.perceptionRenderer || !state.perceptionCamera ||
      !state.semanticRenderer || !state.semanticCamera || !state.semanticScene) return;
  // The simulated sensor pipeline is intentionally fixed at 20 Hz, matching the companion setpoint rate.
  if (timestamp - state.lastSensorRenderMs < 50) return;
  state.lastSensorRenderMs = timestamp;

  const { position, direction, north, east, altitude, headingDeg } = state.sensorPose;
  const cameraPosition = position.clone().addScaledVector(direction, 0.8);
  cameraPosition.y += 0.45;
  const lookTarget = cameraPosition.clone().addScaledVector(direction, 90);
  lookTarget.y -= 2.2;
  state.perceptionCamera.position.copy(cameraPosition);
  state.perceptionCamera.up.set(0, 1, 0);
  state.perceptionCamera.lookAt(lookTarget);
  state.perceptionCamera.updateMatrixWorld(true);

  state.semanticCamera.position.copy(state.perceptionCamera.position);
  state.semanticCamera.quaternion.copy(state.perceptionCamera.quaternion);
  state.semanticCamera.updateMatrixWorld(true);

  const measuredSpeed = state.traceData?.[state.traceIndex]?.velocity
    ? Math.hypot(state.traceData[state.traceIndex].velocity[0], state.traceData[state.traceIndex].velocity[1])
    : state.realFlightSpeed;
  const measuredDecision = measureActualBuildingThreat(position, direction, measuredSpeed);
  state.actualReactiveDecision = state.collisionGuardTriggered ? {
    ...(measuredDecision || state.actualReactiveDecision || {}),
    obstacle_detected: true,
    collision_course: true,
    distance_m: Math.min(3.5, measuredDecision?.distance_m ?? 3.5),
    ttc_s: 0,
    threat_level: 'CRITICAL',
    optimal_action: 'EMERGENCY BRAKE · HOLD POSITION',
    optimal_primitive: 'BRAKE_HOLD_COLLISION_GUARD'
  } : measuredDecision;
  updateLiveCollisionHUD(state.actualReactiveDecision);
  updateActualObstacleHelper(state.actualReactiveDecision);

  // Mission-only symbology is excluded from the synthetic camera image.
  const hidden = [state.missionDroneGroup, state.routeLine, state.directLine3D, state.startBeacon3D,
    state.goalBeacon3D, state.mapLabelsGroup, state.trafficDome3D, state.obstacleBox3D, state.actualObstacleHelper,
    state.pendingTargetBeacon3D, state.pendingDirectLine3D]
    .filter(Boolean).map(object => [object, object.visible]);
  hidden.forEach(([object]) => { object.visible = false; });
  state.perceptionRenderer.render(state.missionScene, state.perceptionCamera);
  hidden.forEach(([object, wasVisible]) => { object.visible = wasVisible; });

  state.semanticRenderer.render(state.semanticScene, state.semanticCamera);
  state.semanticFrame += 1;
  const pose = document.getElementById('camera-pose-readout');
  if (pose) pose.textContent = `N ${north.toFixed(1)} · E ${east.toFixed(1)} · AGL ${altitude.toFixed(1)}m · HDG ${headingDeg.toFixed(0)}°`;
  const status = document.getElementById('semantic-frame-status');
  if (status) status.textContent = 'CLASS RASTER · 20 Hz';
  const meta = document.getElementById('semantic-frame-meta');
  if (meta) meta.textContent = `FRAME ${String(state.semanticFrame).padStart(4, '0')} · SAME POSE · SIMULATION LABELS`;
}

// =============================================================================
// 6. MAIN ANIMATION LOOP
// =============================================================================
let lastTs = performance.now();

function animationLoop(timestamp) {
  try {
    const dt = Math.min(0.05, (timestamp - lastTs) / 1000);
    lastTs = timestamp;

    // 1. Advance Flight Progress smoothly based on real physical flight speed (kilometers simulation)
    if (state.isPlaying && state.totalPoints > 1 && !state.isHoveringAtGoal) {
      if (state.realFlightSpeed > 0.05) {
        // Physical distance per trajectory step
        const avgStepM = Math.max(1.0, (state.totalRouteDistanceM || 1000) / Math.max(1, state.totalPoints - 1));
        // Distance traveled in this dt frame at realFlightSpeed
        const distTraveledM = state.realFlightSpeed * dt * (state.playSpeed || 1.0);
        const progressDelta = distTraveledM / avgStepM;

        state.flightProgress += progressDelta;

        if (state.flightProgress >= state.totalPoints - 1) {
          state.flightProgress = state.totalPoints - 1;
          state.traceIndex = state.totalPoints - 1;
          state.isHoveringAtGoal = true;
        } else {
          state.traceIndex = Math.floor(state.flightProgress);
        }

        // Keep active waypoint index in sync with flight progress
        if (Array.isArray(state.traceData) && state.traceData[state.traceIndex]) {
          const curWp = state.traceData[state.traceIndex].waypoint_index;
          if (curWp !== undefined && curWp !== state.activeWaypointIndex) {
            state.activeWaypointIndex = curWp;
            const wpChips = document.querySelectorAll('.wp-chip');
            wpChips.forEach((c, i) => {
              if (i === curWp) {
                c.classList.add('active');
                c.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
              } else {
                c.classList.remove('active');
              }
            });
            const badge = document.getElementById('disp-active-wp');
            if (badge) {
              const totalWps = state.currentRouteWaypoints?.length || 0;
              badge.textContent = curWp === 0 ? 'WP 1 (START)' : (curWp === totalWps - 1 ? 'GOAL (DEST B)' : `WP ${curWp + 1}`);
            }
          }
        }
      }
    }

    // 2. Update Drone in 3D City (Mission Viewport) with smooth fractional sub-point interpolation
    if (state.traceData && state.missionDroneGroup && state.totalPoints > 0) {
      const pFloor = Math.min(state.totalPoints - 1, Math.floor(state.flightProgress));
      const pCeil = Math.min(state.totalPoints - 1, pFloor + 1);
      const alpha = Math.max(0, Math.min(1, state.flightProgress - pFloor));

      const ptA = state.traceData[pFloor] || state.traceData[0];
      const ptB = state.traceData[pCeil] || ptA;

      let curX = THREE.MathUtils.lerp(ptA.position[0], ptB.position[0], alpha);
      let curY = THREE.MathUtils.lerp(ptA.position[1], ptB.position[1], alpha);

      let vx = THREE.MathUtils.lerp(ptA.velocity ? ptA.velocity[0] : 0, ptB.velocity ? ptB.velocity[0] : 0, alpha);
      let vy = THREE.MathUtils.lerp(ptA.velocity ? ptA.velocity[1] : 0, ptB.velocity ? ptB.velocity[1] : 0, alpha);
      const speed = Math.hypot(vx, vy);
      let activePrimitive = ptA.primitive || 'CRUISE_LEVEL';

      // Apply a smooth local avoidance offset only when measured geometry is on the flight corridor.
      const measuredThreat = state.actualReactiveDecision;
      let desiredNorth = 0;
      let desiredEast = 0;
      let desiredAltitudeBoost = 0;
      if (measuredThreat?.collision_course && measuredThreat.distance_m < 24 && speed > 0.05) {
        const dNorth = vx / speed;
        const dEast = vy / speed;
        const magnitude = Math.min(16, Math.max(0, 25 - measuredThreat.distance_m) * 0.9);
        if (measuredThreat.optimal_path === 'TOP') {
          desiredAltitudeBoost = Math.min(10, measuredThreat.candidate_paths?.[2]?.vertical_offset_m || 6);
        } else {
          const sign = measuredThreat.optimal_path === 'LEFT' ? -1 : 1;
          desiredNorth = -dEast * magnitude * sign;
          desiredEast = dNorth * magnitude * sign;
        }
        activePrimitive = measuredThreat.optimal_primitive;
      }
      const response = 1 - Math.exp(-3.5 * dt);
      state.avoidanceOffsetNed.north = THREE.MathUtils.lerp(state.avoidanceOffsetNed.north, desiredNorth, response);
      state.avoidanceOffsetNed.east = THREE.MathUtils.lerp(state.avoidanceOffsetNed.east, desiredEast, response);
      state.avoidanceAltitudeBoost = THREE.MathUtils.lerp(state.avoidanceAltitudeBoost, desiredAltitudeBoost, response);
      curX += state.avoidanceOffsetNed.north;
      curY += state.avoidanceOffsetNed.east;
      let altNED = -(Math.max(1.0, state.cruiseAltitude || 6.0) + state.avoidanceAltitudeBoost);

      if (state.isHoveringAtGoal) {
        // Station-keeping gentle hover oscillation at destination
        const hTime = timestamp * 0.0015;
        curX += Math.sin(hTime * 1.2) * 0.25;
        curY += Math.cos(hTime * 1.0) * 0.25;
        vx = 0.0;
        vy = 0.0;
        activePrimitive = 'HOLD_HOVER_3D';
      }

      let pos = nedToGltf(curX, curY, altNED);
      const bodyClearance = minimumBuildingClearance(pos);
      if (bodyClearance < 3.5) {
        // Never visually tunnel through a building: hold the last verified pose.
        state.flightProgress = state.lastSafeFlightProgress;
        state.traceIndex = Math.floor(state.lastSafeFlightProgress);
        state.isPlaying = false;
        state.collisionGuardTriggered = true;
        activePrimitive = 'BRAKE_HOLD_COLLISION_GUARD';
        vx = 0;
        vy = 0;
        if (state.lastSafePositionNed) {
          [curX, curY, altNED] = state.lastSafePositionNed;
          pos = nedToGltf(curX, curY, altNED);
        }
        state.actualReactiveDecision = {
          ...(state.actualReactiveDecision || {}),
          obstacle_detected: true,
          collision_course: true,
          distance_m: bodyClearance,
          ttc_s: 0,
          threat_level: 'CRITICAL',
          optimal_action: 'EMERGENCY BRAKE · HOLD POSITION',
          optimal_primitive: activePrimitive
        };
      } else {
        state.lastSafeFlightProgress = state.flightProgress;
        state.lastSafePositionNed = [curX, curY, altNED];
        if (state.isPlaying) state.collisionGuardTriggered = false;
      }
      state.missionDroneGroup.position.copy(pos);

      // Heading & Banking tilt into turns driven by lateral velocity & kinematic primitives
      const commandedSpeed = Math.hypot(vx, vy);
      if (commandedSpeed > 0.05) {
        const heading = Math.atan2(-vx, vy);
        state.missionDroneGroup.rotation.y = heading;

        const bankAngle = THREE.MathUtils.clamp(-vy * 0.04, -0.28, 0.28);
        state.missionDroneGroup.rotation.z = THREE.MathUtils.lerp(state.missionDroneGroup.rotation.z, bankAngle, 0.1);
      }

      let sensorDirection = state.sensorPose?.direction?.clone() || new THREE.Vector3(0, 0, -1);
      if (commandedSpeed > 0.05) sensorDirection.set(vy / commandedSpeed, 0, -vx / commandedSpeed).normalize();
      const headingDeg = (THREE.MathUtils.radToDeg(Math.atan2(vy, vx)) + 360) % 360;
      state.sensorPose = {
        position: pos.clone(),
        direction: sensorDirection,
        north: curX,
        east: curY,
        altitude: -altNED,
        headingDeg
      };

      // Live Dynamic Destination Telemetry Update
      if (state.destination && state.destination.ned) {
        const distToB = Math.hypot(curX - state.destination.ned[0], curY - state.destination.ned[1]);
        const distEl = document.getElementById('disp-dest-dist');
        const etaEl = document.getElementById('disp-dest-eta');
        const primEl = document.getElementById('disp-dest-primitive');

        if (distEl) {
          distEl.textContent = distToB >= 1000 ? `${(distToB / 1000).toFixed(2)} km` : `${Math.round(distToB)} m`;
        }
        if (etaEl) {
          if (state.isHoveringAtGoal) {
            etaEl.textContent = 'REACHED (HOVER)';
          } else {
            const spd = Math.max(0.5, state.realFlightSpeed);
            const sec = distToB / spd;
            etaEl.textContent = sec >= 60 ? `~${(sec / 60).toFixed(1)} min` : `${Math.round(sec)} s`;
          }
        }
        if (primEl) {
          primEl.textContent = activePrimitive;
        }
      }

      // Spin rotors based on real speed
      if (state.isPlaying && state.realFlightSpeed > 0.05) {
        const propSpeed = state.isHoveringAtGoal ? 0.35 : (state.realFlightSpeed / 4.0) * 0.65;
        state.missionDroneProps.forEach(prop => {
          prop.rotation.y += propSpeed;
        });
      }

      // Center 3D traffic dome on drone
      if (state.trafficDome3D) {
        state.trafficDome3D.position.x = pos.x;
        state.trafficDome3D.position.z = pos.z;
      }
    }

    // Animate Pending Target Diamond Beacon if active
    if (state.pendingTargetBeacon3D && state.pendingTargetBeacon3D.visible) {
      const d = state.pendingTargetBeacon3D.getObjectByName('pendingDiamond');
      if (d) {
        d.rotation.y += 0.035;
        d.position.y = 55 + Math.sin(timestamp * 0.004) * 2.5;
      }
    }

    // 3. Render Mission 3D Scene
    if (state.missionRenderer && state.missionScene && state.missionCamera) {
      if (state.missionControls) state.missionControls.update();
      state.missionRenderer.render(state.missionScene, state.missionCamera);
    }

    renderSynchronizedSensors(timestamp);

    // 4. Update Vehicle Twin (gray_drone.glb CAD Viewport)
    if (state.twinDroneModel) {
      const isMoving = state.isPlaying && state.realFlightSpeed > 0.05;
      const speedRatio = state.realFlightSpeed / 4.0;

      if (isMoving) {
        // Built-in hover animation clip
        if (state.twinMixer && state.twinHoverAction) {
          state.twinHoverAction.timeScale = state.isHoveringAtGoal ? 0.45 : Math.max(0.15, speedRatio * 1.5);
          state.twinMixer.update(dt);
        }
        // Direct propeller node rotation
        state.twinPropellers.forEach(p => {
          p.rotation.y += 0.85 * (state.isHoveringAtGoal ? 0.45 : speedRatio);
          p.rotation.z += 0.85 * (state.isHoveringAtGoal ? 0.45 : speedRatio);
        });
      } else {
        // Stop wings/propellers completely when speed is 0 or stopped!
        if (state.twinHoverAction) {
          state.twinHoverAction.timeScale = 0;
        }
      }

      if (state.traceData && state.traceData[state.traceIndex]) {
        const cur = state.traceData[state.traceIndex];
        const vx = cur.velocity ? cur.velocity[0] : 0;
        const vy = cur.velocity ? cur.velocity[1] : 0;
        const tiltFactor = Math.min(1.5, Math.max(0, speedRatio));
        state.twinDroneModel.rotation.x = THREE.MathUtils.lerp(state.twinDroneModel.rotation.x, -vx * 0.04 * tiltFactor, 0.1);
        state.twinDroneModel.rotation.z = THREE.MathUtils.lerp(state.twinDroneModel.rotation.z, -vy * 0.04 * tiltFactor, 0.1);
      }
    }

    // 4b. Dynamic Evasion Banking & Pitch
    const reactive = getEffectiveReactiveDecision();
    if (reactive && reactive.threat_level !== 'CLEAR') {
      if (state.missionDroneGroup) {
        if (reactive.optimal_path === 'RIGHT') {
          state.missionDroneGroup.rotation.z = THREE.MathUtils.lerp(state.missionDroneGroup.rotation.z, -0.22, 0.08);
        } else if (reactive.optimal_path === 'LEFT') {
          state.missionDroneGroup.rotation.z = THREE.MathUtils.lerp(state.missionDroneGroup.rotation.z, 0.22, 0.08);
        } else if (reactive.optimal_path === 'TOP') {
          state.missionDroneGroup.rotation.x = THREE.MathUtils.lerp(state.missionDroneGroup.rotation.x, -0.16, 0.08);
        }
      }
      if (state.twinDroneModel) {
        if (reactive.optimal_path === 'RIGHT') {
          state.twinDroneModel.rotation.z = THREE.MathUtils.lerp(state.twinDroneModel.rotation.z, -0.32, 0.1);
        } else if (reactive.optimal_path === 'LEFT') {
          state.twinDroneModel.rotation.z = THREE.MathUtils.lerp(state.twinDroneModel.rotation.z, 0.32, 0.1);
        } else if (reactive.optimal_path === 'TOP') {
          state.twinDroneModel.rotation.x = THREE.MathUtils.lerp(state.twinDroneModel.rotation.x, -0.22, 0.1);
        }
      }
    }

    // 5. Render Twin Viewport
    if (state.twinRenderer && state.twinScene && state.twinCamera) {
      if (state.twinControls) state.twinControls.update();
      state.twinRenderer.render(state.twinScene, state.twinCamera);
    }

    // 6. Render Latency Chart, Telemetry HUD, and Camera HUD Overlay
    renderLatencyChart();
    updatePlaybackHUD();
    updateReactivePerceptionHUD();
    renderCameraHUDOverlay();

  } catch (err) {
    console.error('Animation error guarded:', err);
  }

  requestAnimationFrame(animationLoop);
}

// =============================================================================
// APP START
// =============================================================================
function startApp() {
  initClock();
  initEventLogTabs();
  initPlaybackControls();
  initMapModeAndSettingsControls();

  // Initialize Camera HUD & Obstacle Scenario Buttons
  initCameraHUDCanvas();
  initAvoidanceScenarioButtons();

  // Initialize both 3D viewports first so canvases and scenes are active
  initMission3DViewport();
  initThreeJsTwin();
  initLatencyChart();

  // Initialize Real-time Telemetry & Mission Toolbar
  initTelemetryAndMissionControls();

  loadLivePerception();

  // Compute initial flight plan using trained AI models
  planAndFlyRoute(state.startNed[0], state.startNed[1], state.destination.ned[0], state.destination.ned[1], -state.destination.ned[2], state.destination.name);

  update3DTrafficDome();
  updateReactivePerceptionHUD();

  requestAnimationFrame(animationLoop);

  setInterval(loadLivePerception, 1200);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', startApp);
} else {
  startApp();
}
