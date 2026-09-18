# Dubai OSM 2.5D scene

Open `dubai_2_5d.blend` in Blender. `preview.png` is a render of the actual scene.
`dubai_2_5d_visual.glb` contains visual meshes for a future dashboard/renderer import.
The GLB uses standard glTF Y-up export; the Blender scene uses local X-east,
Y-north, Z-up coordinates in metres. Apply the appropriate axis conversion when
integrating either asset with ArduPilot NED coordinates.

The input is the user's `C:/Users/ahile/Downloads/map.osm`. Its hash, geographic
bounds, local origin, dimensions, attribution, height assumptions and feature
counts are recorded in `manifest.json`. Geometry is clipped to the extract's
bounds because the OSM export includes complete ways extending outside them.

This approximately 1,023 by 814 metre region has 158 clipped building pieces.
157 use a provisional eight-metre height; one uses mapped floor count multiplied
by an assumed 3.2 metre floor height. Building courtyards are preserved. Materials
are generic. Terrain is flat at an assumed ground plane; no measured geographic
altitude or detailed facade reconstruction is available. Road surfaces are merged
to avoid coplanar flickering. Road widths are estimates, and overpasses/tunnels
are currently flattened. Trees and powerlines are not modeled.

Blender includes separate building and terrain collision collections, hidden in
the viewport/render. These are geometry assets, not active simulator collision
objects. The red `DEMO_RESTRICTED_BASE` is a synthetic validation polygon configured
in `smoke_scenario.json`; it is not an authoritative Dubai flight restriction. The
blue line is the deterministic route recorded in `smoke_result.json`. The drone camera is movable in
Blender but has no live vehicle/gimbal connection. No SITL flight is claimed.

## Rebuild

From the project root in PowerShell:

```powershell
python -m venv .venv-map
.\.venv-map\Scripts\python.exe -m pip install -r tools\requirements-map.txt
.\.venv-map\Scripts\python.exe tools\prepare_osm_world.py C:\Users\ahile\Downloads\map.osm worlds\dubai_osm
python tools\run_dubai_smoke_test.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --factory-startup --python tools\build_osm_blender.py -- worlds\dubai_osm
```

Rebuilding replaces the generated outputs in this directory. Save manual scene
edits to a different `.blend` filename before rebuilding.

## Attribution

© OpenStreetMap contributors. Source data: Open Database License (ODbL) 1.0.
See https://www.openstreetmap.org/copyright and
https://opendatacommons.org/licenses/odbl/1-0/ . Retain attribution with distributed
maps and observe the source database licence when redistributing derived data.
