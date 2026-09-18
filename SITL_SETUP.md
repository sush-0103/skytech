# Windows ArduCopter SITL: verified 19 September 2026

Mission Planner portable 1.3.83 is installed at
`C:/Users/ahile/Downloads/MissionPlanner-portable/MissionPlanner.exe`.
Official download: https://firmware.ardupilot.org/Tools/MissionPlanner/MissionPlanner-latest.zip
Downloaded archive SHA-256: `FF3694F7A9038CAF756545049DD7AEF74E3D0D8FE3586E87AA881045704DD6F9`.
This locally recorded hash identifies the download; it is not an independently
published vendor checksum. The executable has no Authenticode signature.

Mission Planner downloaded and started stable **ArduCopter 4.7.1 (4776a3fb)**.
Its simulator files are in
`C:/Users/ahile/OneDrive/Documents/Mission Planner/sitl/`.
Mission Planner currently owns the simulator process lifecycle.

## Repeat startup

1. Open Mission Planner, choose SIMULATION, Stable release, model `quad`.
2. Set Extra command line to `--home=25.142895,55.258255,0,0`.
3. Click the Multirotor picture. Mission Planner connects via TCP 127.0.0.1:5760.
4. From the FlYtech project root, run `python tools/check_sitl_telemetry.py`.

The probe uses a separate local TCP 5762 connection. It requests telemetry but
never arms, changes flight mode, or sends flight setpoints. Its timestamped result
is saved in `runtime/sitl-telemetry.json`. It checks ArduPilot quadrotor identity,
GPS fix, disarmed state, attitude/local-position reception, and a position within
five metres of the map origin.

Zero metres is an **assumed simulation elevation**, not a surveyed Dubai height.
The scene centre is inside the synthetic restricted polygon. This startup at the
origin is for stationary telemetry only; a route flight must use the smoke-test
start location and explicitly transform between vehicle EKF origin and map origin.

## Connection separation and next gate

The old `ai_server.py` demo was broadcasting companion system 254 to UDP 14550.
Mission Planner auto-connected to it and waited for nonexistent autopilot
parameters. That process was stopped and automatic broadcasting was removed from
the demo server. Restarting the demo HTTP service no longer starts its MAVLink
sender. Demo metrics are still synthetic and are not live flight evidence.

TCP 5760 is the GCS connection, TCP 5762 is available for the telemetry probe or a
future companion (one client at a time). The earlier UDP 14550 diagram is not the
verified connection layout of this Mission Planner-managed instance.

Completed: native Windows startup, actual ArduCopter telemetry and Dubai origin.
Pending: telemetry-driven route follower, explicit frame alignment, velocity-only
command gate, stale-state handling, and actual flight validation. The existing
bridge still does not consume vehicle telemetry and must not be used as a closed
loop controller. Its failsafe position/descent policy also needs revision before
connecting route commands.

The Blender buildings, restricted polygon and rendered camera are not connected
to ArduCopter's physics model. The previous A* test was an offline geometric test
with sampled segment checks, not proof of continuous collision avoidance or a
kinematically feasible flight trajectory. No route flight or camera stream has
been validated in this stage.

Official instructions: https://ardupilot.org/planner/docs/mission-planner-simulation.html
