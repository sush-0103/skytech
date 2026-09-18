"""Prepare clipped, metre-scale geometry from an OSM extract. Requires Shapely 2.1+."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from shapely import constrained_delaunay_triangles, make_valid
from shapely.geometry import Polygon, LineString, box
from shapely.ops import polygonize, unary_union


def polygons(g):
    if g.geom_type == 'Polygon':
        yield g
    elif hasattr(g, 'geoms'):
        for part in g.geoms:
            yield from polygons(part)


def height(tags):
    value = tags.get('height', '')
    if re.fullmatch(r'\s*\d+(?:\.\d+)?\s*(?:m|ft)?\s*', value):
        h = float(re.search(r'\d+(?:\.\d+)?', value)[0])
        if 'ft' in value:
            h *= .3048
        if 0 < h <= 1000:
            return h, 'osm_height'
    try:
        levels = float(tags.get('building:levels', ''))
        if 0 < levels < 200:
            return levels * 3.2, 'osm_levels_times_assumed_3.2m'
    except ValueError:
        pass
    return 8., 'assumed_8m'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = ET.parse(args.source).getroot()
    bounds = {k: float(v) for k, v in root.find('bounds').attrib.items()}
    lat0 = (bounds['minlat'] + bounds['maxlat']) / 2
    lon0 = (bounds['minlon'] + bounds['maxlon']) / 2
    # Local tangent linearization using WGS84 meridional/prime-vertical radii.
    phi = math.radians(lat0)
    e2 = 6.69437999014e-3
    n = 6378137 / math.sqrt(1 - e2 * math.sin(phi)**2)
    m = 6378137 * (1-e2) / (1-e2*math.sin(phi)**2)**1.5

    def xy(lat, lon):
        return (math.radians(lon-lon0)*n*math.cos(phi), math.radians(lat-lat0)*m)

    low = xy(bounds['minlat'], bounds['minlon'])
    high = xy(bounds['maxlat'], bounds['maxlon'])
    clip = box(*low, *high)
    nodes = {x.get('id'): xy(float(x.get('lat')), float(x.get('lon'))) for x in root.findall('node')}
    ways = {x.get('id'): x for x in root.findall('way')}
    features, warnings = [], []

    def tags_of(x):
        return {t.get('k'): t.get('v') for t in x.findall('tag')}

    def points(w):
        refs = [x.get('ref') for x in w.findall('nd')]
        if any(ref not in nodes for ref in refs):
            warnings.append('Missing node in way ' + w.get('id'))
            return []
        return [nodes[ref] for ref in refs]

    def add(g, kind, identity, tags):
        g = make_valid(g).intersection(clip)
        h, provenance = height(tags) if kind == 'building' else (0., 'ground')
        for i, poly in enumerate(polygons(g)):
            if poly.area < .1:
                continue
            triangles = constrained_delaunay_triangles(poly)
            tri = [[list(p) for p in list(t.exterior.coords)[:3]] for t in triangles.geoms]
            if not math.isclose(sum(t.area for t in triangles.geoms), poly.area, rel_tol=1e-6, abs_tol=.01):
                raise ValueError('Triangulation area mismatch: ' + identity)
            features.append(dict(id=f'{identity}_{i}', kind=kind, height_m=h, height_source=provenance,
                                 tags=tags, rings=[list(poly.exterior.coords)[:-1]] +
                                 [list(r.coords)[:-1] for r in poly.interiors], triangles=tri))

    members = set()
    for rel in root.findall('relation'):
        tags = tags_of(rel)
        if tags.get('type') != 'multipolygon' or tags.get('building', 'no') == 'no':
            continue
        outer, inner = [], []
        for member in rel.findall('member'):
            if member.get('type') != 'way':
                continue
            ref = member.get('ref')
            members.add(ref)
            if ref not in ways:
                raise ValueError('Incomplete building relation: ' + rel.get('id'))
            coords = points(ways[ref])
            if len(coords) < 2:
                raise ValueError('Incomplete building ring')
            (inner if member.get('role') == 'inner' else outer).append(LineString(coords))
        shells = unary_union(list(polygonize(outer)))
        holes = unary_union(list(polygonize(inner)))
        if shells.is_empty:
            raise ValueError('Unclosed building relation: ' + rel.get('id'))
        add(shells.difference(holes), 'building', 'relation_' + rel.get('id'), tags)

    widths = {'motorway': 14., 'trunk': 12., 'primary': 10., 'secondary': 9.,
              'tertiary': 8., 'residential': 6., 'service': 4., 'footway': 1.5, 'path': 1.5}
    for identity, w in ways.items():
        tags, coords = tags_of(w), points(w)
        if len(coords) < 2:
            continue
        if tags.get('building', 'no') != 'no' and identity not in members:
            if len(coords) >= 4 and coords[0] == coords[-1]:
                add(Polygon(coords), 'building', 'way_' + identity, tags)
            else:
                warnings.append('Skipped unclosed building way ' + identity)
        elif 'highway' in tags and tags['highway'] not in ('proposed', 'construction'):
            width = widths.get(tags['highway'], 5.)
            roadtags = dict(tags, modeled_width_m=str(width), width_source='assumed_by_road_class')
            add(LineString(coords).buffer(width/2, cap_style=2, join_style=2), 'road', 'way_' + identity, roadtags)
        elif len(coords) >= 4 and coords[0] == coords[-1]:
            kind = 'water' if tags.get('natural') == 'water' else 'green' if (
                tags.get('landuse') in ('grass', 'forest', 'meadow') or tags.get('leisure') in ('park', 'garden')) else None
            if kind:
                add(Polygon(coords), kind, 'way_' + identity, tags)

    # Merge road surfaces so coplanar intersections do not flicker in the renderer.
    road_segments = [f for f in features if f['kind'] == 'road']
    road_surface = unary_union([Polygon(f['rings'][0], f['rings'][1:]) for f in road_segments])
    features = [f for f in features if f['kind'] != 'road']
    add(road_surface, 'road', 'merged_road_surface', {'width_source': 'assumed_by_road_class'})
    buildings = [f for f in features if f['kind'] == 'building']
    manifest = dict(source_file=args.source.name, source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    attribution='© OpenStreetMap contributors', license='ODbL 1.0',
                    source_url='https://www.openstreetmap.org/copyright', bounds_wgs84=bounds,
                    origin_wgs84=dict(latitude=lat0, longitude=lon0, altitude_m=None),
                    coordinates='Local ENU metres; WGS84 tangent linearization at origin; z=0 is assumed ground, not sea level',
                    local_bounds_m=[*low, *high], dimensions_m=[high[0]-low[0], high[1]-low[1]],
                    feature_counts=dict(Counter(f['kind'] for f in features)),
                    source_road_pieces_before_merge=len(road_segments),
                    height_sources=dict(Counter(f['height_source'] for f in buildings)), warnings=warnings,
                    limitations=['Flat assumed terrain; no elevation survey', 'Generic building appearance; no real facade textures',
                                 'Road widths estimated; bridges and tunnels flattened',
                                 'OSM omissions remain; powerlines and trees not modeled',
                                 'No authoritative aviation restrictions in this file; constraint collection starts empty',
                                 'Geometry package only; no running SITL, sensor stream or collision engine integration'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'geometry.json').write_text(json.dumps(features), encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
