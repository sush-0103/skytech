"""Check local ArduCopter telemetry without arming or sending flight setpoints."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5762)
    parser.add_argument('--duration', type=float, default=6)
    parser.add_argument('--output', type=Path, default=ROOT/'runtime'/'sitl-telemetry.json')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.duration <= 30:
        parser.error('port must be valid and duration must be 1–30 seconds')
    endpoint = f'tcp:127.0.0.1:{args.port}'
    connection = mavutil.mavlink_connection(endpoint, source_system=253)
    try:
        heartbeat = connection.wait_heartbeat(timeout=10)
        if heartbeat is None:
            raise RuntimeError('No vehicle heartbeat received')
        if heartbeat.autopilot != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA or heartbeat.type != mavutil.mavlink.MAV_TYPE_QUADROTOR:
            raise RuntimeError('Expected ArduPilot quadrotor, not a companion or another vehicle')
        sysid, compid = heartbeat.get_srcSystem(), heartbeat.get_srcComponent()
        connection.mav.request_data_stream_send(sysid, compid, mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1)
        wanted = {'HEARTBEAT', 'ATTITUDE', 'GLOBAL_POSITION_INT', 'LOCAL_POSITION_NED', 'GPS_RAW_INT'}
        received = {'HEARTBEAT': heartbeat.to_dict()}
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            message = connection.recv_match(blocking=True, timeout=.2)
            if message and message.get_srcSystem() == sysid and message.get_srcComponent() == compid and message.get_type() in wanted:
                received[message.get_type()] = message.to_dict()
        missing = sorted(wanted - received.keys())
        if missing:
            raise RuntimeError(f'Missing telemetry: {missing}')
        origin = json.loads((ROOT/'worlds'/'dubai_osm'/'manifest.json').read_text(encoding='utf-8'))['origin_wgs84']
        gps = received['GLOBAL_POSITION_INT']
        latitude, longitude = gps['lat']/1e7, gps['lon']/1e7
        delta_n = math.radians(latitude-origin['latitude'])*6371000
        delta_e = math.radians(longitude-origin['longitude'])*6371000*math.cos(math.radians(origin['latitude']))
        origin_error = math.hypot(delta_n, delta_e)
        armed = bool(received['HEARTBEAT']['base_mode'] & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        passed = not armed and origin_error < 5 and received['GPS_RAW_INT']['fix_type'] >= 3
        result = dict(status='PASS' if passed else 'FAIL', captured_utc=datetime.now(timezone.utc).isoformat(),
                      endpoint=endpoint, system_id=sysid, component_id=compid, armed=armed,
                      latitude=latitude, longitude=longitude, origin_error_m=round(origin_error, 3),
                      telemetry=received, flight_commands_sent=0,
                      scope='Stationary SITL startup check only; not route/camera/collision validation')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k != 'telemetry'}, indent=2))
        if not passed:
            raise RuntimeError('Expected disarmed vehicle with GPS at configured Dubai origin')
    finally:
        connection.close()


if __name__ == '__main__':
    main()
