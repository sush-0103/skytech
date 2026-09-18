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
  missionDroneProps: [],
  routeLine: null,

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
  obstacleBox3D: null,
  evasionArc3D: null,

  // AI & Filter
  aiData: null,
  eventFilter: 'all'
};
window.state = state;



// Planned waypoints from flight_result.json in local NED (North x, East y)
const PLANNED_WAYPOINTS_NED = [
  [-360.0, -400.0, -6.0],
  [-303.1, -335.7, -6.0],
  [-303.1, -311.7, -6.0],
  [-383.1, -263.7, -6.0],
  [-407.1, -183.7, -6.0],
  [-407.1, -111.7, -6.0],
  [-71.1, -191.7, -6.0],
  [0.9, -231.7, -6.0],
  [48.9, -223.7, -6.0],
  [136.9, -127.7, -6.0],
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
  state.missionRenderer.toneMappingExposure = 1.3;

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
  const ambient = new THREE.AmbientLight(0xffffff, 1.4);
  state.missionScene.add(ambient);

  const sunLight = new THREE.DirectionalLight(0xfffbeb, 2.5);
  sunLight.position.set(400, 900, 500);
  state.missionScene.add(sunLight);

  const skyHemisphere = new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 1.0);
  state.missionScene.add(skyHemisphere);

  // 5. Load dubai_2_5d_visual.glb (from worlds/dubai_osm/dubai_2_5d.blend)
  loadDubaiGlbWorld();

  // 6. Draw 3D Flight Route & Start/Goal Beacons
  build3DFlightPath();

  // 7. Add 3D Quadrotor in City
  buildMissionDrone();

  // 8. 3D Reactive Obstacle Avoidance Cage & Evasion Arc
  build3DObstacleAvoidanceMarker();

  window.addEventListener('resize', onMissionResize);

}

function onMissionResize() {
  if (!state.missionContainer || !state.missionCamera || !state.missionRenderer) return;
  const w = state.missionContainer.clientWidth || 600;
  const h = state.missionContainer.clientHeight || 400;
  state.missionCamera.aspect = w / h;
  state.missionCamera.updateProjectionMatrix();
  state.missionRenderer.setSize(w, h);
}

function loadDubaiGlbWorld() {
  const loader = new GLTFLoader();
  const url = '/worlds/dubai_osm/dubai_2_5d_visual.glb';

  loader.load(
    url,
    (gltf) => {
      state.missionWorldGroup = gltf.scene;

      // Style materials for pristine aerospace aesthetic
      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          const name = (child.name || '').toLowerCase();
          
          if (name.includes('restricted') || name.includes('demo_restricted')) {
            // Synthetic restricted non-entry polygon in glowing crimson
            child.material = new THREE.MeshStandardMaterial({
              color: 0xef4444,
              transparent: true,
              opacity: 0.55,
              roughness: 0.3,
              metalness: 0.1,
              side: THREE.DoubleSide
            });
          } else if (name.includes('road')) {
            // Dark highway asphalt
            child.material = new THREE.MeshStandardMaterial({
              color: 0x1e293b,
              roughness: 0.8,
              metalness: 0.2
            });
          } else if (name.includes('terrain') || name.includes('flat')) {
            // Neutral desert sand / ground
            child.material = new THREE.MeshStandardMaterial({
              color: 0x141f36,
              roughness: 0.9,
              metalness: 0.1
            });
          } else {
            // 158 OSM Building Footprints: Architectural sandstone/orange
            child.material = new THREE.MeshStandardMaterial({
              color: 0xf59e0b,
              roughness: 0.45,
              metalness: 0.15
            });
          }
        }
      });

      state.missionScene.add(state.missionWorldGroup);
      console.log('Loaded 3D Dubai OSM World (dubai_2_5d_visual.glb) successfully.');
    },
    undefined,
    (err) => {
      console.warn('Failed to load dubai_2_5d_visual.glb, building fallback geometry:', err);
      buildFallbackGeometry();
    }
  );
}

function buildFallbackGeometry() {
  // Ground
  const groundGeo = new THREE.PlaneGeometry(1024, 814);
  const groundMat = new THREE.MeshStandardMaterial({ color: 0x141f36, roughness: 0.9 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  state.missionScene.add(ground);

  // Restricted Zone Box
  const rzGeo = new THREE.BoxGeometry(140, 20, 170);
  const rzMat = new THREE.MeshStandardMaterial({ color: 0xef4444, transparent: true, opacity: 0.45 });
  const rz = new THREE.Mesh(rzGeo, rzMat);
  rz.position.set(0, 10, 0);
  state.missionScene.add(rz);
}

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
  const startPos = nedToGltf(-360.0, -400.0, -6.0);
  const goalPos = nedToGltf(300.0, 400.0, -6.0);

  // Start Beacon (Cyan)
  const startPin = create3DBeacon(0x38bdf8, 'START');
  startPin.position.copy(startPos);
  state.missionScene.add(startPin);

  // Goal Beacon (Emerald)
  const goalPin = create3DBeacon(0x10b981, 'GOAL');
  goalPin.position.copy(goalPos);
  state.missionScene.add(goalPin);

  // Blocked Direct Route (Dashed Line)
  const directPts = [startPos, goalPos];
  const directGeo = new THREE.BufferGeometry().setFromPoints(directPts);
  const directMat = new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: 15,
    gapSize: 10
  });
  const directLine = new THREE.Line(directGeo, directMat);
  directLine.computeLineDistances();
  state.missionScene.add(directLine);
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

  // 2. Obstacle Proximity Warning Ring at base
  const ringGeo = new THREE.RingGeometry(22, 26, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xf59e0b,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.65
  });
  const ringMesh = new THREE.Mesh(ringGeo, ringMat);
  ringMesh.rotation.x = Math.PI / 2;
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

  // Vertical light pillar
  const pillarGeo = new THREE.CylinderGeometry(0.8, 0.8, 40, 12);
  const pillarMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 });
  const pillar = new THREE.Mesh(pillarGeo, pillarMat);
  pillar.position.y = 20;
  group.add(pillar);

  // Floating sphere beacon
  const sphereGeo = new THREE.SphereGeometry(6, 16, 16);
  const sphereMat = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.8
  });
  const sphere = new THREE.Mesh(sphereGeo, sphereMat);
  sphere.position.y = 42;
  group.add(sphere);

  return group;
}

function buildMissionDrone() {
  const group = new THREE.Group();

  // Central body
  const bodyGeo = new THREE.BoxGeometry(6.0, 1.4, 4.0);
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, metalness: 0.3, roughness: 0.2 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  group.add(body);

  // 4 Arms & Motor Pods
  const armMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
  const bladeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.8 });
  const armOffsets = [[4.5, 4.5], [-4.5, 4.5], [4.5, -4.5], [-4.5, -4.5]];

  armOffsets.forEach(([ax, az]) => {
    const armGeo = new THREE.CylinderGeometry(0.3, 0.3, 6.5);
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.position.set(ax * 0.5, 0, az * 0.5);
    arm.rotation.z = Math.PI / 4 * Math.sign(ax);
    group.add(arm);

    // Motor
    const motorGeo = new THREE.CylinderGeometry(0.8, 0.8, 1.2, 16);
    const motor = new THREE.Mesh(motorGeo, bodyMat);
    motor.position.set(ax, 0.5, az);
    group.add(motor);

    // Spinning Blade
    const bladeGeo = new THREE.BoxGeometry(5.0, 0.15, 0.6);
    const blade = new THREE.Mesh(bladeGeo, bladeMat);
    blade.position.set(ax, 1.2, az);
    state.missionDroneProps.push(blade);
    group.add(blade);
  });

  // Spotlight shining down from drone
  const spot = new THREE.SpotLight(0x38bdf8, 3.0, 120, Math.PI / 6, 0.3);
  spot.position.set(0, 0, 0);
  spot.target.position.set(0, -60, 0);
  group.add(spot);
  group.add(spot.target);

  state.missionDroneGroup = group;
  state.missionScene.add(group);
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

  if (btnPlay) btnPlay.addEventListener('click', () => { state.isPlaying = true; });
  if (btnPause) btnPause.addEventListener('click', () => { state.isPlaying = false; });
  if (btnRestart) {
    btnRestart.addEventListener('click', () => {
      state.traceIndex = 0;
      state.isPlaying = true;
      if (scrubber) scrubber.value = 0;
    });
  }
  if (scrubber) {
    scrubber.addEventListener('input', (e) => {
      state.traceIndex = parseInt(e.target.value, 10);
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
  const mode = state.avoidanceMode;

  // Real-time dynamic obstacle approach cycle:
  // Real-time dynamic obstacle approach cycle:
  // Continuous progression moving towards the drone
  const cycleLength = 140;
  const cycleProgress = (state.traceIndex % cycleLength) / cycleLength; // 0.0 -> 1.0
  const approachRatio = Math.min(1.0, cycleProgress / 0.82); // 0.0 (far away) -> 1.0 (very close)
  const isEvasionPhase = cycleProgress > 0.82; // Last 18% is lateral bypass

  // Distance counts down continuously in real meters: 48m down to 8.2m
  const dist = 48.0 - approachRatio * 39.8;
  const speed = Math.max(0.5, state.realFlightSpeed !== undefined ? state.realFlightSpeed : 4.2);
  const ttc = Math.max(1.8, dist / speed);

  // In 3D perspective projection, bounding box continuously expands towards the drone:
  // Starts small in the distance (width: 16%, height: 22%, top: 38%)
  // Moves towards drone, growing into full proximity (width: 44%, height: 60%, top: 18%)
  const widthPct = 16 + approachRatio * 28;
  const heightPct = 22 + approachRatio * 38;
  const topPct = 38 - approachRatio * 20;

  let leftPct = 36;
  let optimalDirection = 'RIGHT';
  let optimalAction = 'OPTIMAL PATH: VEER RIGHT (+18°)';
  let optimalPrim = 'BANK_RIGHT_EVADE';
  let obstacleId = 'BLDG-158-WEST';
  let threatLevel = dist < 16 ? 'CRITICAL' : (dist < 28 ? 'CAUTION' : 'CLEAR');

  if (mode === 'left') {
    optimalDirection = 'LEFT';
    optimalAction = 'OPTIMAL PATH: VEER LEFT (-22°)';
    optimalPrim = 'LATERAL_EVADE_WEST';
    obstacleId = 'BLDG-088-EAST';
    // Building on right side: moves from 46% to 54% as it approaches
    leftPct = 46 + approachRatio * 8;
    if (isEvasionPhase) {
      leftPct += (cycleProgress - 0.82) * 160; // Slides off to the right as drone evades left
    }
  } else if (mode === 'top') {
    optimalDirection = 'TOP';
    optimalAction = 'OPTIMAL PATH: CLIMB TOP (+8.5m)';
    optimalPrim = 'ASCEND_RAPID_CLEAR';
    obstacleId = 'BARRIER-014-LOW';
    leftPct = 50 - widthPct / 2;
    if (isEvasionPhase) {
      topPct += (cycleProgress - 0.82) * 80;
    }
  } else {
    // 'auto' or 'right'
    optimalDirection = 'RIGHT';
    optimalAction = 'OPTIMAL PATH: VEER RIGHT (+18°)';
    optimalPrim = 'BANK_RIGHT_EVADE';
    obstacleId = 'BLDG-042-WEST';
    // Building on left side: moves from 38% outwards to 16% as it approaches
    leftPct = 38 - approachRatio * 22;
    if (isEvasionPhase) {
      leftPct -= (cycleProgress - 0.82) * 160; // Slides off to the left as drone evades right
    }
  }

  return {
    obstacle_detected: true,
    obstacle_type: 'Building',
    obstacle_id: obstacleId,
    distance_m: dist,
    ttc_s: ttc,
    threat_level: threatLevel,
    bbox: {
      top: Math.round(topPct),
      left: Math.round(leftPct),
      width: Math.round(widthPct),
      height: Math.round(heightPct)
    },
    candidate_paths: [
      { direction: 'LEFT', clearance_m: optimalDirection === 'LEFT' ? 15.2 : 3.2, cost: optimalDirection === 'LEFT' ? 2.4 : 25.0, primitive: 'LATERAL_EVADE_WEST', status: optimalDirection === 'LEFT' ? 'OPTIMAL' : 'BLOCKED' },
      { direction: 'RIGHT', clearance_m: optimalDirection === 'RIGHT' ? 14.8 : 2.8, cost: optimalDirection === 'RIGHT' ? 2.7 : 28.0, primitive: 'BANK_RIGHT_EVADE', status: optimalDirection === 'RIGHT' ? 'OPTIMAL' : 'BLOCKED' },
      { direction: 'TOP', clearance_m: 8.5, cost: optimalDirection === 'TOP' ? 3.5 : 12.5, primitive: 'ASCEND_RAPID_CLEAR', status: optimalDirection === 'TOP' ? 'OPTIMAL' : 'FEASIBLE' }
    ],
    optimal_path: optimalDirection,
    optimal_action: optimalAction,
    optimal_primitive: optimalPrim,
    clearance_margin_m: 14.8
  };
}

function updateReactivePerceptionHUD() {
  const dec = getEffectiveReactiveDecision();
  const box = document.getElementById('hud-building-box');
  const banner = document.getElementById('camera-evasion-banner');
  const bannerAction = document.getElementById('banner-action');
  const bannerPrim = document.getElementById('banner-primitive');
  const tagDist = document.getElementById('tag-building-dist');
  const tagTtc = document.getElementById('tag-building-ttc');
  const tagRec = document.getElementById('tag-building-rec');
  const tagHazard = document.getElementById('tag-building-hazard');

  const mfLeft = document.getElementById('mf-left');
  const mfRight = document.getElementById('mf-right');
  const mfTop = document.getElementById('mf-top');

  if (box && dec) {
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

    if (tagDist) tagDist.textContent = `DIST: ${dec.distance_m.toFixed(1)}m`;
    if (tagTtc) tagTtc.textContent = `TTC ${dec.ttc_s.toFixed(1)}s`;
    if (tagRec) tagRec.textContent = `OPTIMAL: ${dec.optimal_path || 'RIGHT'}`;
    if (tagHazard) {
      tagHazard.textContent = dec.threat_level === 'CRITICAL' ? 'CRITICAL HAZARD' : (dec.threat_level === 'CAUTION' ? 'PROXIMITY HAZARD' : 'NOMINAL CLEAR');
    }

    if (bannerAction) bannerAction.textContent = dec.optimal_action || `OPTIMAL PATH: VEER ${dec.optimal_path || 'RIGHT'}`;
    if (bannerPrim) bannerPrim.textContent = `KINEMATIC PRIMITIVE: ${dec.optimal_primitive || 'BANK_RIGHT_EVADE'} (3.8 m/s)`;
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

  // Center Flight Path Vector (FPV) reticle
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(vpX, vpY, 5, 0, Math.PI * 2);
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

  // Evasion Gateway Target Reticle
  ctx.save();
  ctx.translate(endX, endY);

  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 1.3;
  ctx.setLineDash([3, 2.5]);
  ctx.beginPath();
  ctx.arc(0, 0, 9, t * 1.5, t * 1.5 + Math.PI * 2);
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.arc(0, 0, 5, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = '#10b981';
  ctx.fillRect(-1, -1, 2, 2);

  ctx.font = '7px "JetBrains Mono", monospace';
  ctx.fillStyle = '#10b981';
  ctx.textAlign = 'center';
  ctx.fillText('OPTIMAL ESCAPE', 0, -12);
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

  updateAirTrafficMetrics();
  updateBatteryPowerFailsafe();
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
    alertEl.textContent = isConflict ? 'WARNING' : 'CLEAR';
    alertEl.className = 't-stat-val ' + (isConflict ? 'text-red' : 'text-emerald');
  }
}

function update3DTrafficDome() {
  if (!state.missionScene) return;
  if (state.trafficDome3D) {
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

function initViewSwitch() {
  const btnBlender = document.getElementById('btn-switch-blender');
  const btnAerial = document.getElementById('btn-switch-aerial');

  if (btnBlender && btnAerial) {
    btnBlender.addEventListener('click', () => {
      btnBlender.classList.add('active');
      btnAerial.classList.remove('active');
      // Reset camera to 3D isometric view over dubai_2_5d.blend
      if (state.missionCamera && state.missionControls) {
        state.missionCamera.position.set(-650, 780, 750);
        state.missionControls.target.set(0, 0, 0);
        state.missionControls.update();
      }
    });

    btnAerial.addEventListener('click', () => {
      btnAerial.classList.add('active');
      btnBlender.classList.remove('active');
      // Set camera to top-down 2D aerial perspective
      if (state.missionCamera && state.missionControls) {
        state.missionCamera.position.set(0, 1100, 0);
        state.missionControls.target.set(0, 0, 0);
        state.missionControls.update();
      }
    });
  }
}

// =============================================================================
// 6. MAIN ANIMATION LOOP
// =============================================================================
let lastTs = performance.now();

function animationLoop(timestamp) {
  try {
    const dt = (timestamp - lastTs) / 1000;
    lastTs = timestamp;

    // 1. Advance Telemetry Index based on real flight speed
    if (state.isPlaying && state.totalPoints > 0) {
      if (state.realFlightSpeed > 0.05) {
        const speedScale = state.realFlightSpeed / 4.0;
        state.traceIndex += Math.max(1, Math.round(state.playSpeed * speedScale * 0.6));
        if (state.traceIndex >= state.totalPoints) {
          state.traceIndex = 0; // Loop seamlessly
        }
      }
    }

    // 2. Update Drone in 3D City (Mission Viewport)
    if (state.traceData && state.missionDroneGroup) {
      const cur = state.traceData[state.traceIndex];
      if (cur && cur.position) {
        const altNED = -Math.max(1.0, state.cruiseAltitude || 6.0);
        const pos = nedToGltf(cur.position[0], cur.position[1], altNED);
        state.missionDroneGroup.position.copy(pos);

        // Heading & Velocity tilt
        if (cur.velocity) {
          const vx = cur.velocity[0];
          const vy = cur.velocity[1];
          const heading = Math.atan2(-vx, vy);
          state.missionDroneGroup.rotation.y = heading;
        }

        // Spin rotors based on real speed (stopped if speed <= 0.05)
        if (state.isPlaying && state.realFlightSpeed > 0.05) {
          const propSpeed = (state.realFlightSpeed / 4.0) * 0.65;
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
    }

    // 3. Render Mission 3D Scene
    if (state.missionRenderer && state.missionScene && state.missionCamera) {
      if (state.missionControls) state.missionControls.update();
      state.missionRenderer.render(state.missionScene, state.missionCamera);
    }

    // 4. Update Vehicle Twin (gray_drone.glb CAD Viewport)
    if (state.twinDroneModel) {
      const isMoving = state.isPlaying && state.realFlightSpeed > 0.05;
      const speedRatio = state.realFlightSpeed / 4.0;

      if (isMoving) {
        // Built-in hover animation clip
        if (state.twinMixer && state.twinHoverAction) {
          state.twinHoverAction.timeScale = Math.max(0.15, speedRatio * 1.5);
          state.twinMixer.update(dt);
        }
        // Direct propeller node rotation
        state.twinPropellers.forEach(p => {
          p.rotation.y += 0.85 * speedRatio;
          p.rotation.z += 0.85 * speedRatio;
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
  initViewSwitch();

  // Initialize Camera HUD & Obstacle Scenario Buttons
  initCameraHUDCanvas();
  initAvoidanceScenarioButtons();

  // Initialize Real-time Telemetry & Mission Toolbar
  initTelemetryAndMissionControls();

  loadFlightTrace();
  loadLivePerception();

  // Initialize both 3D viewports
  initMission3DViewport();
  initThreeJsTwin();
  initLatencyChart();

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
