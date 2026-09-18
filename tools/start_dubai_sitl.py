"""Launch the downloaded native SITL in an isolated run directory at route start."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.navigation.route_follower import LocalMapFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sitl-dir', type=Path, required=True)
    args = parser.parse_args()
    exe = (args.sitl_dir/'ArduCopter.exe').resolve()
    defaults = (args.sitl_dir/'default_params'/'copter.parm').resolve()
    if not exe.is_file() or not defaults.is_file():
        raise RuntimeError('Missing downloaded ArduCopter or quad defaults')
    for port in (5760,5762,5763):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} already in use; stop the previous disarmed SITL first')
    manifest = json.loads((ROOT/'worlds/dubai_osm/manifest.json').read_text())
    scenario = json.loads((ROOT/'worlds/dubai_osm/smoke_scenario.json').read_text())
    origin = manifest['origin_wgs84']
    frame = LocalMapFrame(origin['latitude'], origin['longitude'])
    lat, lon = frame.to_geographic(*scenario['start_ned'])
    directory = ROOT/'runtime'/('sitl-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    directory.mkdir(parents=True)
    command = [str(exe), '--model=quad', f'--home={lat:.9f},{lon:.9f},0,0',
               '--speedup=1', '--defaults', str(defaults), '--serial0=tcp:5760', '--serial1=tcp:5762']
    with (directory/'stdout.log').open('w') as stdout, (directory/'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, cwd=directory, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    record = dict(pid=process.pid, executable=str(exe), command=command, directory=str(directory),
                  home_wgs84=[lat,lon,0], start_map_ned=scenario['start_ned'])
    (ROOT/'runtime'/'active_sitl.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
