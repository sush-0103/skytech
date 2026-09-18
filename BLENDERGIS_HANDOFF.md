# BlenderGIS scene handoff

## Generated OSM scene now available

The user supplied `C:/Users/ahile/Downloads/map.osm`. A derived scene is available
at `worlds/dubai_osm/dubai_2_5d.blend`, with a rendered preview, a visual GLB,
geometry JSON and a provenance manifest alongside it. See that directory's
README for assumptions and rebuild commands. This completes the first scene
asset handoff; live SITL camera and collision integration remain pending.

Save the selected approximately 1 km by 1 km region as a `.blend` file and provide its path. Add-on installation alone does not produce a map scene. Keep the source scene and packed textures, and record source URLs, licence, selected bounds, CRS, geographic origin, height datum and scale. Never invent georeferencing for a decorative city model.

Use metre units and a local origin. Preserve a mapping from the local Blender ENU axes to the WGS84 origin. Separate terrain/building visual meshes from simplified collision meshes using the collection names in Section 7.3 of the master report. Record which heights are measured, tagged or assumed.

Read-only inspection, from the project directory in PowerShell:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background 'PATH_TO_MAP.blend' --python tools/inspect_blender_scene.py
python -m unittest discover -s tests -p test_camera_geometry.py -v
```

The inspector prints base-mesh bounds, metre dimensions, scene properties, collision collections and missing textures. It does not save or alter the scene. Modifiers, animated bounds, geographic accuracy and simulator contact behavior require subsequent checks.

## Current implementation status

Camera transforms and timestamp validation are implemented independently of a renderer in `src/interface/camera_geometry.py`. Seven unit tests pass. This does not establish rendered camera or SITL integration.

The bridge now preserves candidate timestamps, no longer integrates a fictional position, and uses the corrected position/velocity/yaw mask. It still lacks received vehicle-state feedback and a validated failsafe policy. The demo server's automatic MAVLink broadcast has been removed. See `SITL_SETUP.md` for the verified Windows simulator and telemetry setup. Existing benchmark/README claims have not been independently verified. The dataset loader uses a different class-index order from the report and asserts eight parent groups; reconcile these against provenance before reusing trained weights or changing labels.

The next integration gate is an actual saved geographic scene, followed by synchronized rendered RGB/depth. Mission Planner remains the Windows GCS; the additional renderer is needed for camera pixels and collision geometry.
