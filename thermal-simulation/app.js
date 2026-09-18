/**
 * UAV AERODIAGNOSTICS PRO - CORE 3D SIMULATION ENGINE
 * Premium Studio White Background | Light Gray Drone Finish | Thermal Heatmap Engine
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- System State & Telemetry ---
const state = {
  // Palette: 'flir', 'spectrum', 'alert'
  currentPalette: 'flir',
  explodedProgress: 0,
  rpm: 4800,
  targetRpm: 4800,
  isAudioAlarmActive: false,
  showBadges: true,
  autoRotate: true,
  selectedComponent: null,
  activeScenario: 'nominal',

  // Component Subsystems State
  components: {
    m1: {
      id: 'm1',
      name: 'M1 Front-Right Motor',
      short: 'M1',
      temp: 31.2,
      targetTemp: 31.2,
      baseTemp: 31.0,
      maxSafe: 75.0,
      critThreshold: 85.0,
      rpm: 4820,
      status: 'OPTIMAL',
      faultCode: null,
      desc: 'Front-right brushless motor. Direct drive stator assembly.',
      recommendation: 'Optimal phase balancing. Normal thermal dissipation.',
      meshes: [],
      materials: [],
      badgeEl: null,
      worldPos: new THREE.Vector3()
    },
    m2: {
      id: 'm2',
      name: 'M2 Rear-Left Motor',
      short: 'M2',
      temp: 32.0,
      targetTemp: 32.0,
      baseTemp: 31.5,
      maxSafe: 75.0,
      critThreshold: 85.0,
      rpm: 4810,
      status: 'OPTIMAL',
      faultCode: null,
      desc: 'Rear-left counter-rotating brushless propulsion motor.',
      recommendation: 'Nominal operational status.',
      meshes: [],
      materials: [],
      badgeEl: null,
      worldPos: new THREE.Vector3()
    },
    m3: {
      id: 'm3',
      name: 'M3 Front-Left Motor',
      short: 'M3',
      temp: 30.8,
      targetTemp: 30.8,
      baseTemp: 30.5,
      maxSafe: 75.0,
      critThreshold: 85.0,
      rpm: 4815,
      status: 'OPTIMAL',
      faultCode: null,
      desc: 'Front-left brushless motor with telemetry-integrated ESC.',
      recommendation: 'No thermal stress detected. Bearings lubricated.',
      meshes: [],
      materials: [],
      badgeEl: null,
      worldPos: new THREE.Vector3()
    },
    m4: {
      id: 'm4',
      name: 'M4 Rear-Right Motor',
      short: 'M4',
      temp: 31.5,
      targetTemp: 31.5,
      baseTemp: 31.0,
      maxSafe: 75.0,
      critThreshold: 85.0,
      rpm: 4822,
      status: 'OPTIMAL',
      faultCode: null,
      desc: 'Rear-right counter-rotating motor assembly.',
      recommendation: 'Nominal operational status.',
      meshes: [],
      materials: [],
      badgeEl: null,
      worldPos: new THREE.Vector3()
    },
    battery: {
      id: 'battery',
      name: '6S LiPo Battery Module',
      short: 'BAT',
      temp: 33.1,
      targetTemp: 33.1,
      baseTemp: 32.5,
      maxSafe: 65.0,
      critThreshold: 75.0,
      rpm: 0,
      status: 'OPTIMAL',
      faultCode: null,
      desc: '22.2V 5000mAh 100C High-Discharge Lithium-Polymer Pack.',
      recommendation: 'Individual cell delta < 8mV. Internal impedance within nominal specs.',
      meshes: [],
      materials: [],
      badgeEl: null,
      worldPos: new THREE.Vector3()
    },
    fc: {
      id: 'fc',
      name: 'Flight Controller & Avionics',
      short: 'FC',
      temp: 35.4,
      targetTemp: 35.4,
      baseTemp: 34.5,
      maxSafe: 70.0,
      critThreshold: 80.0,
      rpm: 0,
      status: 'OPTIMAL',
      faultCode: null,
      desc: 'STM32 H7 Dual-Core Flight MCU, redundant IMUs and telemetry bus.',
      recommendation: 'Real-time loop frequency 8kHz, CPU load 22%. Thermals stable.',
      meshes: [],
      materials: [],
      badgeEl: null,
      worldPos: new THREE.Vector3()
    },
    camera: {
      id: 'camera',
      name: '4K Gimbal Optical Payload',
      short: 'CAM',
      temp: 34.0,
      targetTemp: 34.0,
      baseTemp: 33.5,
      maxSafe: 65.0,
      critThreshold: 75.0,
      rpm: 0,
      status: 'OPTIMAL',
      faultCode: null,
      desc: '3-Axis stabilized gimbal with 1/2-inch Sony sensor.',
      recommendation: 'Sensor cooling heatsink clear, encoder temperatures normal.',
      meshes: [],
      materials: [],
      badgeEl: null,
      worldPos: new THREE.Vector3()
    }
  },

  // Rolling Telemetry History for Canvas Chart
  history: {
    maxPoints: 60,
    m1: [],
    m3: [],
    battery: [],
    fc: []
  }
};

// --- Three.js Globals ---
let scene, camera, renderer, controls;
let droneRoot = null;
let mixer = null;
let hoverAction = null;
let explodedAction = null;
let clock = new THREE.Clock();
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let hoveredMesh = null;

// Camera views config
const CAMERA_PRESETS = {
  isometric: { pos: new THREE.Vector3(2.6, 1.8, 2.6), target: new THREE.Vector3(0, 0.1, 0) },
  top: { pos: new THREE.Vector3(0.01, 3.8, 0.01), target: new THREE.Vector3(0, 0.1, 0) },
  front: { pos: new THREE.Vector3(0.0, 0.5, 3.2), target: new THREE.Vector3(0, 0.1, 0) },
  battery: { pos: new THREE.Vector3(1.2, 1.1, 0.6), target: new THREE.Vector3(0, 0.2, 0) },
  reset: { pos: new THREE.Vector3(2.6, 1.8, 2.6), target: new THREE.Vector3(0, 0.1, 0) }
};

// Audio Context for High-Tech Alarms
let audioCtx = null;
let lastAlarmTime = 0;

function playThermalAlarmChirp() {
  if (!state.isAudioAlarmActive) return;
  const now = performance.now();
  if (now - lastAlarmTime < 1800) return; // limit frequency
  lastAlarmTime = now;

  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1320, audioCtx.currentTime + 0.12);
    osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.24);

    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.28);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.3);
  } catch (e) {
    console.warn('Audio not allowed yet:', e);
  }
}

// ==========================================================================
// 1. INITIALIZATION & STUDIO ENVIRONMENT
// ==========================================================================

function init() {
  const container = document.getElementById('viewport-container');
  const canvas = document.getElementById('webgl-canvas');

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f6f9);

  // Camera
  camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.05, 50);
  camera.position.set(2.6, 1.8, 2.6);

  // Renderer
  renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance'
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.18;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // Orbit Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 0.8;
  controls.maxDistance = 6.5;
  controls.maxPolarAngle = Math.PI / 2 + 0.04; // don't go below floor
  controls.autoRotate = state.autoRotate;
  controls.autoRotateSpeed = 0.8;
  controls.target.set(0, 0.1, 0);

  // Studio Lighting
  setupStudioLighting();

  // Studio Floor Shadow & Circular Precision Grid
  setupStudioFloor();

  // Load Drone Model (fly.glb)
  loadDroneModel();

  // Initialize UI & Event Listeners
  setupUI();
  setup3DBadges();
  setupHistory();

  // Window Resize
  window.addEventListener('resize', onWindowResize);

  // Raycaster Events
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('click', onPointerClick);

  // Animation Loop
  animate();
}

function setupStudioLighting() {
  // Hemispheric Ambient Light (Soft sky white, subtle gray bounce from floor)
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0xe2e8f0, 1.15);
  hemiLight.position.set(0, 8, 0);
  scene.add(hemiLight);

  // Key Studio Directional Light with soft contact shadows
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
  keyLight.position.set(3.5, 5.0, 3.5);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 15;
  keyLight.shadow.camera.left = -2.5;
  keyLight.shadow.camera.right = 2.5;
  keyLight.shadow.camera.top = 2.5;
  keyLight.shadow.camera.bottom = -2.5;
  keyLight.shadow.bias = -0.0005;
  keyLight.shadow.normalBias = 0.02;
  scene.add(keyLight);

  // Soft Fill Light (Cool light tone)
  const fillLight = new THREE.DirectionalLight(0xe0f2fe, 0.85);
  fillLight.position.set(-3.5, 3.0, -2.5);
  scene.add(fillLight);

  // Rim / Kicker Light to accent drone chassis contours
  const rimLight = new THREE.DirectionalLight(0xf8fafc, 0.65);
  rimLight.position.set(0, 2.5, -4.0);
  scene.add(rimLight);
}

function setupStudioFloor() {
  // Studio Shadow Floor Plane
  const shadowPlaneGeo = new THREE.PlaneGeometry(12, 12);
  const shadowPlaneMat = new THREE.ShadowMaterial({
    opacity: 0.12
  });
  const shadowPlane = new THREE.Mesh(shadowPlaneGeo, shadowPlaneMat);
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = -0.52;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  // Engineering Circular Grid
  const gridHelper = new THREE.PolarGridHelper(3.2, 16, 8, 64, 0xd1d5db, 0xe2e8f0);
  gridHelper.position.y = -0.518;
  scene.add(gridHelper);
}

// ==========================================================================
// 2. 3D DRONE MODEL LOADING & LIGHT GRAY MATERIAL PIPELINE
// ==========================================================================

function loadDroneModel() {
  const loader = new GLTFLoader();
  const progressEl = document.getElementById('loader-progress');
  const statusText = document.getElementById('loader-status-text');

  loader.load(
    'gray_drone.glb',
    (gltf) => {
      droneRoot = gltf.scene;

      // Fit and center model
      const bbox = new THREE.Box3().setFromObject(droneRoot);
      const center = bbox.getCenter(new THREE.Vector3());
      const size = bbox.getSize(new THREE.Vector3());

      const maxAxis = Math.max(size.x, size.y, size.z);
      const targetScale = 2.2 / maxAxis; // scale to comfortable viewing size
      droneRoot.scale.setScalar(targetScale);

      // Re-center around origin
      droneRoot.position.x = -center.x * targetScale;
      droneRoot.position.y = -center.y * targetScale + 0.12;
      droneRoot.position.z = -center.z * targetScale;

      // Group & classify meshes into components
      classifyAndStyleModel(droneRoot);

      scene.add(droneRoot);

      // Setup Animation Mixer (Hover and Exploded View)
      if (gltf.animations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(droneRoot);

        gltf.animations.forEach((clip) => {
          if (clip.name === 'hover') {
            hoverAction = mixer.clipAction(clip);
            hoverAction.setLoop(THREE.LoopRepeat);
            hoverAction.play();
          } else if (clip.name === 'exploded_view') {
            explodedAction = mixer.clipAction(clip);
            explodedAction.setLoop(THREE.LoopOnce);
            explodedAction.clampWhenFinished = true;
            explodedAction.paused = true;
            explodedAction.time = 0;
          }
        });
      }

      // Hide loading screen smoothly
      setTimeout(() => {
        const loadingScreen = document.getElementById('loading-screen');
        loadingScreen.classList.add('hidden');
        addEventLog('System Ready: 3D UAV Model loaded with 77 light gray subcomponents.', 'info');
      }, 400);
    },
    (xhr) => {
      if (xhr.lengthComputable) {
        const percent = Math.round((xhr.loaded / xhr.total) * 100);
        if (progressEl) progressEl.style.width = percent + '%';
        if (statusText) statusText.textContent = `Loading geometry & textures (${percent}%)...`;
      }
    },
    (err) => {
      console.error('Failed to load gray_drone.glb:', err);
      if (statusText) statusText.textContent = 'Error loading model file.';
    }
  );
}

/**
 * Classifies model nodes into subsystems and applies the premium light gray satin finish.
 */
function classifyAndStyleModel(root) {
  root.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;

      const nodeName = (child.name || '').toLowerCase();
      let compKey = getComponentKeyForNode(nodeName);

      // Create Light Gray Satin PBR Material
      // Sleek #cfd4dc / #d8dde6 with high specular response
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0xd2d7df),
        roughness: 0.38,
        metalness: 0.24,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0.0
      });

      // Special styling for carbon rotors & lenses
      if (nodeName.includes('prop_')) {
        mat.color.setHex(0x27272a); // dark carbon fiber for props
        mat.roughness = 0.5;
        mat.metalness = 0.1;
      } else if (nodeName.includes('camera') && nodeName.includes('screw') === false) {
        mat.color.setHex(0xb0b7c3);
        mat.metalness = 0.45;
        mat.roughness = 0.25;
      }

      child.material = mat;

      if (compKey && state.components[compKey]) {
        state.components[compKey].meshes.push(child);
        state.components[compKey].materials.push(mat);
        child.userData.componentKey = compKey;
      } else {
        child.userData.componentKey = 'frame';
      }
    }
  });

  // Calculate world positions for 3D badges
  calculateSubsystemAnchorPositions();
}

function getComponentKeyForNode(name) {
  if (name.includes('motor_1') || name.includes('prop_motor_base_1')) return 'm1';
  if (name.includes('motor_2') || name.includes('prop_motor_base_2')) return 'm2';
  if (name.includes('motor_3') || name.includes('prop_motor_base_3')) return 'm3';
  if (name.includes('motor_4') || name.includes('prop_motor_base_4')) return 'm4';
  if (name.includes('battery')) return 'battery';
  if (name.includes('receiver') || name.includes('top_board')) return 'fc';
  if (name.includes('camera')) return 'camera';
  return null;
}

function calculateSubsystemAnchorPositions() {
  for (const key in state.components) {
    const comp = state.components[key];
    if (comp.meshes.length > 0) {
      const box = new THREE.Box3();
      comp.meshes.forEach((m) => box.expandByObject(m));
      box.getCenter(comp.worldPos);
    }
  }
}

// ==========================================================================
// 3. THERMAL HEATMAP COLOR ENGINE
// ==========================================================================

/**
 * Maps a temperature in °C to a dynamic RGB color based on selected palette.
 */
function getThermalColor(temp, palette = state.currentPalette) {
  // Normalize temp between 25°C and 95°C
  const t = THREE.MathUtils.clamp((temp - 25) / (95 - 25), 0, 1);

  if (palette === 'flir') {
    // FLIR Ironbow: Cool Gray -> Deep Purple -> Vivid Magenta -> Warm Orange -> Solar Yellow -> White
    if (t < 0.15) {
      return new THREE.Color().lerpColors(new THREE.Color(0xd2d7df), new THREE.Color(0x3b154c), t / 0.15);
    } else if (t < 0.38) {
      return new THREE.Color().lerpColors(new THREE.Color(0x3b154c), new THREE.Color(0x9333ea), (t - 0.15) / 0.23);
    } else if (t < 0.65) {
      return new THREE.Color().lerpColors(new THREE.Color(0x9333ea), new THREE.Color(0xf97316), (t - 0.38) / 0.27);
    } else if (t < 0.85) {
      return new THREE.Color().lerpColors(new THREE.Color(0xf97316), new THREE.Color(0xfde047), (t - 0.65) / 0.20);
    } else {
      return new THREE.Color().lerpColors(new THREE.Color(0xfde047), new THREE.Color(0xffffff), (t - 0.85) / 0.15);
    }
  } else if (palette === 'spectrum') {
    // Jet Spectrum: Cool Light Gray -> Cyan -> Green -> Yellow -> Red -> White
    if (t < 0.18) {
      return new THREE.Color().lerpColors(new THREE.Color(0xd2d7df), new THREE.Color(0x06b6d4), t / 0.18);
    } else if (t < 0.40) {
      return new THREE.Color().lerpColors(new THREE.Color(0x06b6d4), new THREE.Color(0x10b981), (t - 0.18) / 0.22);
    } else if (t < 0.65) {
      return new THREE.Color().lerpColors(new THREE.Color(0x10b981), new THREE.Color(0xeab308), (t - 0.40) / 0.25);
    } else if (t < 0.85) {
      return new THREE.Color().lerpColors(new THREE.Color(0xeab308), new THREE.Color(0xef4444), (t - 0.65) / 0.20);
    } else {
      return new THREE.Color().lerpColors(new THREE.Color(0xef4444), new THREE.Color(0xffffff), (t - 0.85) / 0.15);
    }
  } else {
    // 'alert' mode: Pristine Light Gray with glowing warning highlights when hot
    if (t < 0.4) {
      return new THREE.Color(0xd2d7df);
    } else if (t < 0.7) {
      return new THREE.Color().lerpColors(new THREE.Color(0xd2d7df), new THREE.Color(0xf59e0b), (t - 0.4) / 0.3);
    } else {
      return new THREE.Color().lerpColors(new THREE.Color(0xf59e0b), new THREE.Color(0xdc2626), (t - 0.7) / 0.3);
    }
  }
}

/**
 * Updates component materials, emissive glow, and pulses each frame.
 */
function updateThermalMaterials(time) {
  let peakTemp = 0;
  let hottestCompName = 'M1 Motor';
  let activeFaults = 0;

  for (const key in state.components) {
    const comp = state.components[key];

    // Smooth temperature interpolation
    comp.temp += (comp.targetTemp - comp.temp) * 0.04;

    if (comp.temp > peakTemp) {
      peakTemp = comp.temp;
      hottestCompName = comp.name;
    }

    // Determine status & faults
    if (comp.temp >= comp.critThreshold) {
      comp.status = 'CRITICAL';
      activeFaults++;
    } else if (comp.temp >= comp.maxSafe) {
      comp.status = 'WARNING';
      activeFaults++;
    } else if (comp.temp >= 45.0) {
      comp.status = 'ELEVATED';
    } else {
      comp.status = 'OPTIMAL';
    }

    // Compute color & thermal emission
    const thermColor = getThermalColor(comp.temp);
    const isOverheating = comp.temp > 50.0;
    const isCritical = comp.temp >= comp.critThreshold;

    // Emissive pulsing effect on overheating parts
    let emissiveInt = 0;
    if (isCritical) {
      emissiveInt = 0.5 + 0.5 * Math.sin(time * 8.0);
    } else if (isOverheating) {
      emissiveInt = (comp.temp - 50.0) / 45.0 * 0.4;
    }

    comp.materials.forEach((mat) => {
      mat.color.copy(thermColor);

      if (isOverheating) {
        mat.emissive.copy(thermColor);
        mat.emissiveIntensity = emissiveInt;
      } else {
        mat.emissive.setHex(0x000000);
        mat.emissiveIntensity = 0.0;
      }
    });

    // Update UI Elements for this component
    updateComponentUI(comp);
  }

  // Header quick metrics
  const peakEl = document.getElementById('hdr-peak-temp');
  const hotEl = document.getElementById('hdr-hottest-node');
  const faultEl = document.getElementById('hdr-faults-count');
  if (peakEl) peakEl.textContent = `${peakTemp.toFixed(1)}°C`;
  if (hotEl) hotEl.textContent = hottestCompName;
  if (faultEl) {
    faultEl.textContent = activeFaults;
    if (activeFaults > 0) {
      faultEl.classList.remove('alert-zero');
      faultEl.style.color = '#dc2626';
    } else {
      faultEl.classList.add('alert-zero');
      faultEl.style.color = '';
    }
  }

  // Global Status Badge
  updateGlobalStatus(peakTemp, activeFaults);

  // Update Bottom Legend Cursor
  updateLegendCursor(peakTemp);

  // Play audio alert if critical
  if (activeFaults > 0 && peakTemp >= 78.0) {
    playThermalAlarmChirp();
  }
}

function updateGlobalStatus(peakTemp, activeFaults) {
  const badge = document.getElementById('global-status-badge');
  const statusText = document.getElementById('status-text');
  if (!badge || !statusText) return;

  badge.className = 'system-status-indicator';
  if (activeFaults > 0 && peakTemp >= 80.0) {
    badge.classList.add('status-critical');
    statusText.textContent = `CRITICAL THERMAL ALERT (${activeFaults} FAULTS)`;
  } else if (activeFaults > 0 || peakTemp >= 60.0) {
    badge.classList.add('status-elevated');
    statusText.textContent = `SYSTEM ELEVATED THERMAL (${peakTemp.toFixed(1)}°C)`;
  } else {
    statusText.textContent = 'SYSTEM NOMINAL';
  }
}

function updateLegendCursor(peakTemp) {
  const cursor = document.getElementById('legend-cursor');
  const label = document.getElementById('cursor-label');
  if (!cursor || !label) return;

  const pct = THREE.MathUtils.clamp((peakTemp - 20) / (105 - 20) * 100, 2, 98);
  cursor.style.left = `${pct}%`;
  label.textContent = `${peakTemp.toFixed(1)}°C`;
}

// ==========================================================================
// 4. 3D FLOATING SCREEN-PINNED CALLOUT BADGES
// ==========================================================================

function setup3DBadges() {
  const overlay = document.getElementById('badges-overlay');
  if (!overlay) return;
  overlay.innerHTML = '';

  for (const key in state.components) {
    const comp = state.components[key];
    const badge = document.createElement('div');
    badge.className = 'floating-3d-badge';
    badge.dataset.comp = key;

    badge.innerHTML = `
      <span class="badge-tag-id">${comp.short}</span>
      <span class="badge-tag-temp" id="badge-temp-${key}">${comp.temp.toFixed(0)}°C</span>
      <span class="badge-tag-dot" id="badge-dot-${key}"></span>
    `;

    badge.addEventListener('click', () => {
      openComponentInspector(key);
    });

    overlay.appendChild(badge);
    comp.badgeEl = badge;
  }
}

function update3DBadgesPosition() {
  if (!state.showBadges) {
    document.getElementById('badges-overlay').style.display = 'none';
    return;
  }
  document.getElementById('badges-overlay').style.display = 'block';

  const tempVec = new THREE.Vector3();
  const widthHalf = window.innerWidth / 2;
  const heightHalf = window.innerHeight / 2;

  for (const key in state.components) {
    const comp = state.components[key];
    if (!comp.badgeEl || comp.meshes.length === 0) continue;

    // Recalculate center from first mesh in world space
    comp.meshes[0].getWorldPosition(tempVec);

    // Project to 2D NDC
    tempVec.project(camera);

    // Check if in front of camera
    if (tempVec.z < 1) {
      const x = (tempVec.x * widthHalf) + widthHalf;
      const y = -(tempVec.y * heightHalf) + heightHalf;

      comp.badgeEl.style.left = `${x}px`;
      comp.badgeEl.style.top = `${y}px`;
      comp.badgeEl.style.display = 'flex';

      const tempEl = document.getElementById(`badge-temp-${key}`);
      if (tempEl) tempEl.textContent = `${comp.temp.toFixed(0)}°C`;

      if (comp.temp >= comp.maxSafe) {
        comp.badgeEl.classList.add('badge-hot');
      } else {
        comp.badgeEl.classList.remove('badge-hot');
      }
    } else {
      comp.badgeEl.style.display = 'none';
    }
  }
}

// ==========================================================================
// 5. UI SYNCHRONIZATION & TELEMETRY CHARTS
// ==========================================================================

function updateComponentUI(comp) {
  // Sidebar cards
  const tempDisp = document.getElementById(`temp-display-${comp.id}`);
  const bar = document.getElementById(`bar-${comp.id}`);
  const badge = document.getElementById(`badge-${comp.id}`);

  if (tempDisp) tempDisp.textContent = `${comp.temp.toFixed(1)}°C`;

  if (bar) {
    const pct = THREE.MathUtils.clamp((comp.temp - 20) / (100 - 20) * 100, 10, 100);
    bar.style.width = `${pct}%`;

    if (comp.status === 'CRITICAL') {
      bar.style.backgroundColor = 'var(--status-critical)';
    } else if (comp.status === 'WARNING') {
      bar.style.backgroundColor = 'var(--status-warning)';
    } else if (comp.status === 'ELEVATED') {
      bar.style.backgroundColor = 'var(--status-elevated)';
    } else {
      bar.style.backgroundColor = 'var(--status-optimal)';
    }
  }

  if (badge) {
    badge.textContent = comp.status;
    badge.className = 'comp-badge';
    if (comp.status === 'CRITICAL') badge.classList.add('badge-hot-alert');
    else if (comp.status === 'WARNING') badge.classList.add('badge-warm');
    else if (comp.status === 'ELEVATED') badge.classList.add('badge-warm');
    else badge.classList.add('badge-normal');
  }

  // Sync manual slider label
  const valSlider = document.getElementById(`val-${comp.id}`);
  if (valSlider) valSlider.textContent = `${comp.temp.toFixed(0)}°C`;
}

// Sparkline Canvas Chart
function setupHistory() {
  for (let i = 0; i < state.history.maxPoints; i++) {
    state.history.m1.push(31.2);
    state.history.m3.push(30.8);
    state.history.battery.push(33.1);
    state.history.fc.push(35.4);
  }

  setInterval(sampleTelemetryHistory, 1000);
}

function sampleTelemetryHistory() {
  state.history.m1.push(state.components.m1.temp);
  state.history.m3.push(state.components.m3.temp);
  state.history.battery.push(state.components.battery.temp);
  state.history.fc.push(state.components.fc.temp);

  if (state.history.m1.length > state.history.maxPoints) {
    state.history.m1.shift();
    state.history.m3.shift();
    state.history.battery.shift();
    state.history.fc.shift();
  }

  drawSparklineChart();
}

function drawSparklineChart() {
  const canvas = document.getElementById('telemetry-sparkline');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Draw subtle grid lines
  ctx.strokeStyle = '#f1f5f9';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h * 0.25); ctx.lineTo(w, h * 0.25);
  ctx.moveTo(0, h * 0.50); ctx.lineTo(w, h * 0.50);
  ctx.moveTo(0, h * 0.75); ctx.lineTo(w, h * 0.75);
  ctx.stroke();

  // Helper to draw series
  const drawSeries = (data, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (state.history.maxPoints - 1)) * w;
      // Map 20°C to 105°C
      const norm = THREE.MathUtils.clamp((data[i] - 20) / (105 - 20), 0, 1);
      const y = h - (norm * (h - 10)) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  drawSeries(state.history.m1, '#2563eb');
  drawSeries(state.history.m3, '#ea580c');
  drawSeries(state.history.battery, '#dc2626');
  drawSeries(state.history.fc, '#0284c7');
}

// ==========================================================================
// 6. INTERACTIVE FAULT SCENARIOS
// ==========================================================================

const SCENARIOS = {
  nominal: {
    title: 'All Systems Nominal',
    temps: { m1: 31.2, m2: 32.0, m3: 30.8, m4: 31.5, battery: 33.1, fc: 35.4, camera: 34.0 },
    rpm: 4800,
    log: 'Normal flight profile engaged. Balanced thermals, no anomalies.'
  },
  m1_friction: {
    title: 'Motor 1 Bearing Friction & Overheat',
    temps: { m1: 88.5, m2: 33.0, m3: 31.5, m4: 32.0, battery: 38.2, fc: 37.0, camera: 34.5 },
    rpm: 4100,
    log: 'ALERT: M1 Bearing friction detected. Temperature escalated to 88.5°C.'
  },
  m3_mosfet: {
    title: 'Motor 3 ESC Mosfet Breakdown',
    temps: { m1: 33.5, m2: 32.5, m3: 96.2, m4: 33.0, battery: 42.0, fc: 39.5, camera: 35.0 },
    rpm: 3800,
    log: 'CRITICAL: M3 ESC Mosfet short circuit! Temperature critical at 96.2°C.'
  },
  battery_runaway: {
    title: 'LiPo Battery Core Thermal Runaway',
    temps: { m1: 34.0, m2: 34.0, m3: 34.0, m4: 34.0, battery: 82.5, fc: 44.0, camera: 36.0 },
    rpm: 4400,
    log: 'CRITICAL: Battery internal core surge (82.5°C). Immediate RTH recommended.'
  },
  fc_throttle: {
    title: 'Flight Controller MCU Overload',
    temps: { m1: 32.5, m2: 32.5, m3: 32.5, m4: 32.5, battery: 36.0, fc: 74.8, camera: 35.5 },
    rpm: 4800,
    log: 'WARNING: Flight controller thermal throttling active (74.8°C).'
  },
  gimbal_heat: {
    title: '4K Gimbal Camera Core Overheat',
    temps: { m1: 32.0, m2: 32.0, m3: 32.0, m4: 32.0, battery: 35.0, fc: 38.0, camera: 69.2 },
    rpm: 4800,
    log: 'WARNING: Camera ISP encoder thermal peak (69.2°C).'
  },
  max_thrust: {
    title: 'Sustained Max Thrust Test',
    temps: { m1: 67.5, m2: 68.2, m3: 67.0, m4: 67.8, battery: 52.0, fc: 48.5, camera: 38.0 },
    rpm: 8200,
    log: 'Stress Test: Full 100% throttle load across all propulsion motors.'
  }
};

function applyScenario(scenarioKey) {
  const sc = SCENARIOS[scenarioKey];
  if (!sc) return;

  state.activeScenario = scenarioKey;

  // Apply temperatures
  for (const key in sc.temps) {
    if (state.components[key]) {
      state.components[key].targetTemp = sc.temps[key];
      // Update slider input value
      const slider = document.querySelector(`.temp-slider[data-target="${key}"]`);
      if (slider) slider.value = sc.temps[key];
    }
  }

  // Set throttle
  state.targetRpm = sc.rpm;
  updateThrottleUI(sc.rpm);

  // Log to stream
  addEventLog(sc.log, scenarioKey === 'nominal' ? 'info' : 'warn');

  // Update active button
  document.querySelectorAll('.scenario-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.scenario === scenarioKey);
  });
}

function updateThrottleUI(rpm) {
  state.rpm = rpm;
  const disp = document.getElementById('val-throttle');
  const hdrThrot = document.getElementById('hdr-throttle');

  if (disp) disp.textContent = `${rpm.toLocaleString()} RPM`;
  if (hdrThrot) {
    if (rpm === 0) hdrThrot.textContent = 'STOPPED (0%)';
    else if (rpm <= 2500) hdrThrot.textContent = 'IDLE (30%)';
    else if (rpm <= 5500) hdrThrot.textContent = 'HOVER (60%)';
    else hdrThrot.textContent = 'MAX THRUST (100%)';
  }

  // Update button active
  document.querySelectorAll('.throttle-btn').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.dataset.rpm) === rpm);
  });

  // Adjust hover animation speed
  if (hoverAction) {
    if (rpm === 0) hoverAction.paused = true;
    else {
      hoverAction.paused = false;
      hoverAction.timeScale = rpm / 4800;
    }
  }
}

function addEventLog(msg, type = 'info') {
  const container = document.getElementById('event-stream-container');
  if (!container) return;

  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];

  const item = document.createElement('div');
  item.className = 'event-item';

  let typeClass = '';
  if (type === 'warn') typeClass = 'event-warn';
  if (type === 'crit') typeClass = 'event-crit';

  item.innerHTML = `
    <span class="event-time">${timeStr}</span>
    <span class="event-msg ${typeClass}">${msg}</span>
  `;

  container.prepend(item);

  // Keep max 20 events
  while (container.children.length > 20) {
    container.removeChild(container.lastChild);
  }
}

// ==========================================================================
// 7. COMPONENT DETAIL INSPECTOR (CLICK-TO-INSPECT)
// ==========================================================================

function openComponentInspector(compKey) {
  const comp = state.components[compKey];
  if (!comp) return;

  state.selectedComponent = compKey;
  const modal = document.getElementById('inspector-modal');
  if (!modal) return;

  document.getElementById('modal-comp-name').textContent = comp.name.toUpperCase();
  document.getElementById('modal-comp-status').textContent = comp.status;
  document.getElementById('modal-temp').textContent = `${comp.temp.toFixed(1)}°C`;

  const descEl = document.getElementById('modal-thermal-desc');
  if (comp.status === 'CRITICAL') {
    descEl.textContent = 'CRITICAL OVERHEAT';
    descEl.className = 'stat-val stat-crit';
    descEl.style.color = 'var(--status-critical)';
  } else if (comp.status === 'WARNING') {
    descEl.textContent = 'WARNING ELEVATED';
    descEl.className = 'stat-val stat-warn';
    descEl.style.color = 'var(--status-warning)';
  } else {
    descEl.textContent = 'NOMINAL COOL';
    descEl.className = 'stat-val stat-ok';
    descEl.style.color = 'var(--status-optimal)';
  }

  document.getElementById('modal-recommendation').textContent =
    comp.status === 'CRITICAL'
      ? `CRITICAL ALERT: Component thermal dissipation limit exceeded. Immediate power throttle and inspection of coil winding / ESC required.`
      : comp.recommendation;

  modal.classList.remove('modal-hidden');
}

function closeComponentInspector() {
  const modal = document.getElementById('inspector-modal');
  if (modal) modal.classList.add('modal-hidden');
  state.selectedComponent = null;
}

// ==========================================================================
// 8. RAYCASTING & 3D INTERACTION
// ==========================================================================

function onPointerMove(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  if (!droneRoot) return;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(droneRoot.children, true);

  if (intersects.length > 0) {
    const hit = intersects[0].object;
    if (hit.userData && hit.userData.componentKey && hit.userData.componentKey !== 'frame') {
      document.body.style.cursor = 'pointer';
      hoveredMesh = hit;
      return;
    }
  }

  document.body.style.cursor = 'default';
  hoveredMesh = null;
}

function onPointerClick(event) {
  // Prevent raycast if clicked inside UI panels
  if (event.target.closest('#top-header') ||
      event.target.closest('.sidebar-drawer') ||
      event.target.closest('#bottom-bar') ||
      event.target.closest('#inspector-modal')) {
    return;
  }

  if (!droneRoot) return;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(droneRoot.children, true);

  if (intersects.length > 0) {
    const hit = intersects[0].object;
    if (hit.userData && hit.userData.componentKey && hit.userData.componentKey !== 'frame') {
      openComponentInspector(hit.userData.componentKey);
    }
  }
}

// ==========================================================================
// 9. UI EVENT BINDINGS
// ==========================================================================

function setupUI() {
  // Scenario Buttons
  document.querySelectorAll('.scenario-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      applyScenario(btn.dataset.scenario);
    });
  });

  // Exploded View Slider
  const explodedSlider = document.getElementById('slider-exploded');
  const explodedVal = document.getElementById('val-exploded');
  if (explodedSlider) {
    explodedSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state.explodedProgress = val / 100;
      if (explodedVal) {
        explodedVal.textContent = val === 0 ? '0% (Assembled)' : `${val}% (Exploded)`;
      }

      // Scrub exploded action
      if (explodedAction) {
        const duration = explodedAction.getClip().duration;
        explodedAction.time = (val / 100) * duration;
        mixer.update(0); // force update pose
      }
    });
  }

  // Throttle Buttons
  document.querySelectorAll('.throttle-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      updateThrottleUI(parseInt(btn.dataset.rpm));
    });
  });

  // Manual Temperature Sliders
  document.querySelectorAll('.temp-slider').forEach((slider) => {
    slider.addEventListener('input', (e) => {
      const target = e.target.dataset.target;
      const val = parseFloat(e.target.value);
      if (state.components[target]) {
        state.components[target].targetTemp = val;
      }
    });
  });

  // Reset all to 28°C
  const btnCoolAll = document.getElementById('btn-cool-all');
  if (btnCoolAll) {
    btnCoolAll.addEventListener('click', () => {
      for (const key in state.components) {
        state.components[key].targetTemp = 28.0;
        const slider = document.querySelector(`.temp-slider[data-target="${key}"]`);
        if (slider) slider.value = 28;
      }
      addEventLog('All drone components reset to 28.0°C nominal ambient.', 'info');
    });
  }

  // Palette Buttons
  document.querySelectorAll('.palette-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.palette-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentPalette = btn.dataset.palette;

      // Update bottom gradient bar CSS representation
      const gradBar = document.getElementById('thermal-gradient-bar');
      if (gradBar) {
        if (state.currentPalette === 'flir') {
          gradBar.style.background = 'linear-gradient(90deg, #d2d7df 0%, #3b154c 18%, #9333ea 42%, #f97316 70%, #fde047 88%, #ffffff 100%)';
        } else if (state.currentPalette === 'spectrum') {
          gradBar.style.background = 'linear-gradient(90deg, #d2d7df 0%, #06b6d4 20%, #10b981 42%, #eab308 65%, #ef4444 86%, #ffffff 100%)';
        } else {
          gradBar.style.background = 'linear-gradient(90deg, #d2d7df 0%, #d2d7df 45%, #f59e0b 70%, #dc2626 100%)';
        }
      }
    });
  });

  // Camera Presets
  document.querySelectorAll('.view-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.view-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      setCameraView(btn.dataset.view);
    });
  });

  // Toggle Badges Button
  const toggleBadgesBtn = document.getElementById('toggle-badges-btn');
  if (toggleBadgesBtn) {
    toggleBadgesBtn.addEventListener('click', () => {
      state.showBadges = !state.showBadges;
      toggleBadgesBtn.classList.toggle('active', state.showBadges);
    });
  }

  // Turntable Auto-rotate
  const autorotateBtn = document.getElementById('camera-autorotate-btn');
  if (autorotateBtn) {
    autorotateBtn.addEventListener('click', () => {
      state.autoRotate = !state.autoRotate;
      controls.autoRotate = state.autoRotate;
      autorotateBtn.classList.toggle('active', state.autoRotate);
    });
  }

  // Sound Alarm Toggle
  const soundToggleBtn = document.getElementById('sound-toggle-btn');
  const soundStatusText = document.getElementById('sound-status-text');
  if (soundToggleBtn) {
    soundToggleBtn.addEventListener('click', () => {
      state.isAudioAlarmActive = !state.isAudioAlarmActive;
      soundToggleBtn.classList.toggle('active', state.isAudioAlarmActive);
      if (soundStatusText) {
        soundStatusText.textContent = state.isAudioAlarmActive ? 'ALARM ON' : 'ALARM OFF';
      }
      if (state.isAudioAlarmActive) {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
      }
    });
  }

  // Subsystem card click in sidebar
  document.querySelectorAll('.subsystem-card').forEach((card) => {
    card.addEventListener('click', () => {
      openComponentInspector(card.dataset.comp);
    });
  });

  // Modal Close & Actions
  const closeModalBtn = document.getElementById('close-modal-btn');
  if (closeModalBtn) closeModalBtn.addEventListener('click', closeComponentInspector);

  const modalStressBtn = document.getElementById('modal-stress-btn');
  if (modalStressBtn) {
    modalStressBtn.addEventListener('click', () => {
      if (state.selectedComponent && state.components[state.selectedComponent]) {
        state.components[state.selectedComponent].targetTemp = 92.0;
        addEventLog(`Manual stress injection on ${state.components[state.selectedComponent].name}: 92.0°C`, 'crit');
        openComponentInspector(state.selectedComponent);
      }
    });
  }

  const modalCoolBtn = document.getElementById('modal-cool-btn');
  if (modalCoolBtn) {
    modalCoolBtn.addEventListener('click', () => {
      if (state.selectedComponent && state.components[state.selectedComponent]) {
        state.components[state.selectedComponent].targetTemp = 28.0;
        addEventLog(`Forced cooling on ${state.components[state.selectedComponent].name}: 28.0°C`, 'info');
        openComponentInspector(state.selectedComponent);
      }
    });
  }

  // Export Log Button
  const exportBtn = document.getElementById('export-report-btn');
  if (exportBtn) {
    exportBtn.addEventListener('click', exportTelemetryReport);
  }
}

function setCameraView(viewName) {
  const preset = CAMERA_PRESETS[viewName] || CAMERA_PRESETS.isometric;

  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const startTime = performance.now();
  const duration = 800; // ms

  function transition(t) {
    const elapsed = t - startTime;
    const progress = Math.min(elapsed / duration, 1.0);
    // Smooth cubic ease out
    const ease = 1 - Math.pow(1 - progress, 3);

    camera.position.lerpVectors(startPos, preset.pos, ease);
    controls.target.lerpVectors(startTarget, preset.target, ease);

    if (progress < 1.0) {
      requestAnimationFrame(transition);
    }
  }

  requestAnimationFrame(transition);
}

function exportTelemetryReport() {
  const report = {
    platform: 'SKY-X4 Autonomous UAV',
    timestamp: new Date().toISOString(),
    ambientTemperature: 24.5,
    palette: state.currentPalette,
    components: {}
  };

  for (const key in state.components) {
    const c = state.components[key];
    report.components[key] = {
      name: c.name,
      currentTemperature: parseFloat(c.temp.toFixed(2)),
      safeThreshold: c.maxSafe,
      criticalThreshold: c.critThreshold,
      status: c.status
    };
  }

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `UAV_Diagnostics_Report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);

  addEventLog('Telemetry incident report downloaded.', 'info');
}

// ==========================================================================
// 10. ANIMATION LOOP & RENDER
// ==========================================================================

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  const delta = clock.getDelta();
  const elapsedTime = clock.getElapsedTime();

  // Update Three.js Animation Mixer for hover props
  if (mixer) {
    mixer.update(delta);
  }

  // Update OrbitControls
  controls.update();

  // Update Dynamic Thermal Heatmap Materials
  updateThermalMaterials(elapsedTime);

  // Update 3D Floating Badges positions
  update3DBadgesPosition();

  // Render Scene
  renderer.render(scene, camera);
}

// Start application
window.addEventListener('DOMContentLoaded', init);
