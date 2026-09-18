"""Run in a fresh background Blender process: --python this.py -- WORLD_DIRECTORY."""
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

out = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
features = json.loads((out/'geometry.json').read_text(encoding='utf-8'))
manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1
collections = {}
for name in ('terrain_visual', 'terrain_collision', 'buildings_visual', 'buildings_collision',
             'roads_visual', 'water_visual', 'vegetation_visual', 'restricted_visual',
             'restricted_constraints', 'landmarks', 'drone_spawn', 'presentation'):
    coll = bpy.data.collections.new(name)
    scene.collection.children.link(coll)
    collections[name] = coll
    if name.endswith('_collision'):
        coll.hide_render = True
        coll.hide_viewport = True


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = .8
    return mat


mats = {'building': material('Generic warm facade', (.63, .54, .41)),
        'roof': material('Generic light roof', (.83, .79, .68)),
        'road': material('Asphalt', (.105, .12, .14)),
        'water': material('Water', (.08, .35, .48)),
        'green': material('Vegetation ground', (.22, .36, .17)),
        'ground': material('Sand ground', (.59, .48, .32)),
        'restricted': material('Restricted area', (.65, .025, .02)),
        'route': material('Verified route', (.02, .35, .95))}
for key, color in (('restricted', (.65, .025, .02, 1)), ('route', (.02, .35, .95, 1))):
    shader = mats[key].node_tree.nodes.get('Principled BSDF')
    shader.inputs['Emission Color'].default_value = color
    shader.inputs['Emission Strength'].default_value = .8


def mesh_obj(name, vertices, faces, coll, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collections[coll].objects.link(obj)
    obj.data.materials.append(mat)
    return obj


for f in features:
    kind = f['kind']
    z = f['height_m'] if kind == 'building' else {'road': .06, 'water': .025, 'green': .03}[kind]
    verts, faces, lookup = [], [], {}

    def vertex(x, y, h):
        key = (round(x, 7), round(y, 7), h)
        if key not in lookup:
            lookup[key] = len(verts)
            verts.append((x, y, h))
        return lookup[key]

    for tri in f['triangles']:
        # Ensure upward-facing roof/ground triangles.
        a, b, c = tri
        if (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) < 0:
            tri = list(reversed(tri))
        faces.append(tuple(vertex(x, y, z) for x, y in tri))
        if kind == 'building':
            faces.append(tuple(vertex(x, y, 0.) for x, y in reversed(tri)))
    roof_face_count = len(faces)
    if kind == 'building':
        for ring_index, ring in enumerate(f['rings']):
            area = sum(ring[i][0]*ring[(i+1)%len(ring)][1] - ring[(i+1)%len(ring)][0]*ring[i][1] for i in range(len(ring)))
            if (area > 0) != (ring_index == 0):
                ring = list(reversed(ring))
            for i, a in enumerate(ring):
                b = ring[(i+1)%len(ring)]
                faces.append((vertex(*a, 0.), vertex(*b, 0.), vertex(*b, z), vertex(*a, z)))
    coll = {'building': 'buildings_visual', 'road': 'roads_visual', 'water': 'water_visual', 'green': 'vegetation_visual'}[kind]
    obj = mesh_obj(f['id'], verts, faces, coll, mats[kind])
    obj['osm_tags'] = json.dumps(f['tags'], ensure_ascii=False)
    obj['height_source'] = f['height_source']
    obj['height_m'] = f['height_m']
    if kind == 'building':
        obj.data.materials.append(mats['roof'])
        for p in obj.data.polygons[:roof_face_count]:
            p.material_index = 1
        collision = bpy.data.objects.new(f['id']+'_collision', obj.data.copy())
        collections['buildings_collision'].objects.link(collision)
        collision['role'] = 'static_collision_geometry_for_future_simulator_export'

x0, y0, x1, y1 = manifest['local_bounds_m']
ground = mesh_obj('Flat assumed terrain', [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0)],
                  [(0,1,2,3)], 'terrain_visual', mats['ground'])
terrain_collision = bpy.data.objects.new('Terrain collision', ground.data.copy())
collections['terrain_collision'].objects.link(terrain_collision)

# Show the configured software-only restriction and last verified smoke route.
scenario_path = out/'smoke_scenario.json'
result_path = out/'smoke_result.json'
if scenario_path.exists():
    scenario = json.loads(scenario_path.read_text(encoding='utf-8'))
    for zone in scenario.get('restricted_zones', []):
        # Scenario coordinates are NED horizontal (north, east); Blender is ENU.
        vertices = [(east, north, .18) for north, east in zone['polygon_ned']]
        signed_area = sum(vertices[i][0]*vertices[(i+1)%len(vertices)][1] -
                          vertices[(i+1)%len(vertices)][0]*vertices[i][1]
                          for i in range(len(vertices)))
        if signed_area < 0:
            vertices.reverse()
        zone_obj = mesh_obj(zone['id'], vertices, [tuple(range(len(vertices)))],
                            'restricted_visual', mats['restricted'])
        zone_obj['reason'] = zone['reason']
        constraint = bpy.data.objects.new(zone['id']+'_constraint', zone_obj.data.copy())
        collections['restricted_constraints'].objects.link(constraint)
        constraint['role'] = 'software_non_entry_constraint'
if result_path.exists():
    smoke_result = json.loads(result_path.read_text(encoding='utf-8'))
    route_curve = bpy.data.curves.new('Verified smoke route', type='CURVE')
    route_curve.dimensions = '3D'
    route_curve.bevel_depth = 1.5
    route_curve.bevel_resolution = 3
    spline = route_curve.splines.new('POLY')
    points = smoke_result['route_waypoints_ned']
    spline.points.add(len(points)-1)
    for point, (north, east) in zip(spline.points, points):
        point.co = (east, north, smoke_result['altitude_m_agl'], 1)
    route_obj = bpy.data.objects.new('Verified smoke route', route_curve)
    collections['presentation'].objects.link(route_obj)
    route_obj.data.materials.append(mats['route'])
    route_obj['status'] = smoke_result['status']
    scene['smoke_result'] = json.dumps(smoke_result)
scene['world_manifest'] = json.dumps(manifest)
scene['origin_latitude'] = manifest['origin_wgs84']['latitude']
scene['origin_longitude'] = manifest['origin_wgs84']['longitude']
scene['axes'] = 'X east, Y north, Z up; metres'
scene['height_datum'] = 'Assumed flat ground z=0; geographic altitude unknown'

# A movable example camera, not yet linked to simulated vehicle telemetry.
camdata = bpy.data.cameras.new('Drone RGB camera')
camdata.lens = 24
camdata.clip_end = 3000
dronecam = bpy.data.objects.new('Drone RGB camera - unconnected', camdata)
collections['drone_spawn'].objects.link(dronecam)
if scenario_path.exists():
    start_north, start_east = scenario['start_ned']
    dronecam.location = (start_east, start_north, scenario['altitude_m_agl'])
    look_north, look_east = scenario['goal_ned']
    dronecam.rotation_euler = (Vector((look_east, look_north, scenario['altitude_m_agl']))-dronecam.location).to_track_quat('-Z', 'Y').to_euler()
else:
    dronecam.location = (0, -250, 60)
    dronecam.rotation_euler = (Vector((0, 80, 0))-dronecam.location).to_track_quat('-Z', 'Y').to_euler()
dronecam['integration_status'] = 'Example camera only; no SITL pose or gimbal feed'

camdata = bpy.data.cameras.new('Overview camera')
camera = bpy.data.objects.new('Overview camera', camdata)
collections['presentation'].objects.link(camera)
camera.location = (650, -900, 950)
camera.rotation_euler = (-camera.location).to_track_quat('-Z', 'Y').to_euler()
camdata.type = 'ORTHO'
camdata.ortho_scale = max(x1-x0, y1-y0)*1.4
camdata.clip_end = 5000
scene.camera = camera
sun_data = bpy.data.lights.new('Sun', 'SUN')
sun_data.energy = 2.5
sun_data.angle = math.radians(15)
sun = bpy.data.objects.new('Sun', sun_data)
collections['presentation'].objects.link(sun)
sun.rotation_euler = (math.radians(30), math.radians(-25), math.radians(-25))
scene.world.color = (.3, .3, .3)
scene.render.engine = 'CYCLES'
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x = 1400
scene.render.resolution_y = 1100
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(out/'preview.png')
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_distance = 1300
            area.spaces.active.clip_end = 5000
            area.spaces.active.region_3d.view_rotation = camera.rotation_euler.to_quaternion()
bpy.ops.wm.save_as_mainfile(filepath=str(out/'dubai_2_5d.blend'))
bpy.ops.object.select_all(action='DESELECT')
for name in ('terrain_visual', 'buildings_visual', 'roads_visual', 'water_visual',
             'vegetation_visual', 'restricted_visual', 'presentation'):
    for obj in collections[name].objects:
        obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(out/'dubai_2_5d_visual.glb'), export_format='GLB', use_selection=True)
bpy.ops.render.render(write_still=True)
print('WORLD_BUILD_COMPLETE', str(out))
