"""Read-only Blender scene inspection; prints JSON, never saves the blend.

Run: blender --background MAP.blend --python tools/inspect_blender_scene.py
BlenderGIS fields are reported as evidence, not guessed to be georeferencing.
"""
import json
import math
import bpy
from mathutils import Vector


def inspect_scene():
    scene = bpy.context.scene
    required = ('terrain_visual', 'terrain_collision', 'buildings_visual',
                'buildings_collision', 'roads_visual', 'water_visual',
                'restricted_constraints', 'drone_spawn')
    objects = []
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        corners = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        lo = [min(v[i] for v in corners) for i in range(3)]
        hi = [max(v[i] for v in corners) for i in range(3)]
        collections = [c.name for c in obj.users_collection]
        objects.append(dict(name=obj.name, vertices=len(obj.data.vertices),
                            polygons=len(obj.data.polygons),
                            bounds_min=lo, bounds_max=hi,
                            dimensions_m=[(hi[i]-lo[i])*scene.unit_settings.scale_length
                                          for i in range(3)],
                            scale=list(obj.scale), collections=collections,
                            collision=any(c.endswith('_collision') for c in collections)))
    missing_images = []
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            import os
            if not os.path.isfile(bpy.path.abspath(image.filepath)):
                missing_images.append(image.name)
    metadata = {key: repr(scene[key]) for key in scene.keys()}
    issues = []
    if not bpy.data.filepath:
        issues.append('Scene has not been saved as a .blend file')
    if scene.unit_settings.system != 'METRIC' or not math.isclose(scene.unit_settings.scale_length, 1.):
        issues.append('Configure metric units with one Blender unit equal to one metre')
    missing = [name for name in required if name not in bpy.data.collections]
    if missing:
        issues.append('Missing required collections: ' + ', '.join(missing))
    if missing_images:
        issues.append('Missing external textures')
    if not any(o['collision'] for o in objects):
        issues.append('No meshes assigned to collision collections')
    return dict(file=bpy.data.filepath, blender_version=bpy.app.version_string,
                unit_system=scene.unit_settings.system,
                metres_per_unit=scene.unit_settings.scale_length,
                scene_custom_properties=metadata,
                mesh_count=len(objects), objects=objects,
                missing_textures=missing_images, issues=issues,
                geographic_alignment_verified=False,
                note='Manual/source-backed CRS, origin and height validation still required; bounds are unevaluated base geometry')


if __name__ == '__main__':
    print('SCENE_AUDIT_JSON_BEGIN')
    print(json.dumps(inspect_scene(), indent=2))
    print('SCENE_AUDIT_JSON_END')
