"""Local, managed SITL only: GUIDED takeoff, feedback route following, LAND.

Default is preflight only. --fly explicitly enables the simulated flight.
Requires the process record produced by start_dubai_sitl.py and TCP 5762.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.navigation.osm_world import OSMWorld, RestrictedZone
from src.navigation.route_follower import LocalMapFrame, VehicleState, RouteFollower, shortcut_route, gate_velocity


class Link:
    def __init__(self):
        self.connection = mavutil.mavlink_connection('tcp:127.0.0.1:5762', source_system=253)
        self.messages, self.times = {}, {}
        self.heartbeat_at = 0.
        self.boot = time.monotonic()
        self.sysid, self.compid = 1, 1
        self.sent = 0
        self.command_count = 0
        self.last_position_boot_ms = None

    def pump(self):
        now = time.monotonic()
        if now-self.heartbeat_at >= 1:
            self.connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                             mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self.heartbeat_at = now
        for _ in range(1000):
            m = self.connection.recv_match(blocking=False)
            if m is None:
                break
            if m.get_srcSystem() != self.sysid or m.get_srcComponent() != self.compid:
                continue
            if m.get_type() == 'LOCAL_POSITION_NED':
                boot_ms = m.time_boot_ms
                if self.last_position_boot_ms is not None:
                    advance = (boot_ms-self.last_position_boot_ms) & 0xffffffff
                    if advance == 0:
                        continue  # A repeated sample must not refresh freshness.
                    if advance >= 0x80000000:
                        raise RuntimeError('Position clock moved backwards or simulator restarted')
                self.last_position_boot_ms = boot_ms
            self.messages[m.get_type()] = m
            self.times[m.get_type()] = time.monotonic()
            if m.get_type() == 'STATUSTEXT':
                print('SITL:', m.text, flush=True)

    def wait(self, condition, timeout, description):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            self.pump()
            if condition():
                return
            time.sleep(.02)
        raise RuntimeError('Timed out: '+description)

    def command(self, command, *params):
        self.messages.pop('COMMAND_ACK', None)
        self.connection.mav.command_long_send(self.sysid,self.compid,command,0,*(list(params)+[0.]*(7-len(params))))
        self.command_count += 1

    def acknowledged(self, command):
        self.wait(lambda: 'COMMAND_ACK' in self.messages and self.messages['COMMAND_ACK'].command == command,
                  5, 'command acknowledgement')
        if self.messages['COMMAND_ACK'].result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f'Command {command} rejected: {self.messages["COMMAND_ACK"].result}')

    def mode(self, number):
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, number)
        self.acknowledged(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        self.wait(lambda: self.messages.get('HEARTBEAT') and self.messages['HEARTBEAT'].custom_mode == number,
                  5,'mode change')

    def velocity(self, vector):
        self.connection.mav.set_position_target_local_ned_send(
            int((time.monotonic()-self.boot)*1000) & 0xffffffff, self.sysid,self.compid,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0x0DC7, 0,0,0,*vector,0,0,0,0,0)
        self.sent += 1

    def armed(self):
        return bool(self.messages['HEARTBEAT'].base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def verify_managed_process(record):
    # Verify this is the local native executable launched by our launcher.
    pid = int(record['pid'])
    code = f'Get-CimInstance Win32_Process -Filter "ProcessId = {pid}" | Select-Object ExecutablePath,CommandLine | ConvertTo-Json -Compress'
    result = subprocess.run(['powershell','-NoProfile','-Command',code],capture_output=True,text=True,check=True)
    process = json.loads(result.stdout)
    if not process or Path(process['ExecutablePath']).resolve() != Path(record['executable']).resolve():
        raise RuntimeError('Managed simulator process does not match')
    if '--model=quad' not in process['CommandLine'] or '--serial1=tcp:5762' not in process['CommandLine']:
        raise RuntimeError('Unexpected simulator command line')
    ownership = subprocess.run(['powershell','-NoProfile','-Command',
        '(Get-NetTCPConnection -State Listen -LocalPort 5762).OwningProcess | ConvertTo-Json -Compress'],
        capture_output=True,text=True,check=True)
    if json.loads(ownership.stdout) != pid:
        raise RuntimeError('Companion port is not owned by the managed simulator')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fly', action='store_true')
    args = parser.parse_args()
    record = json.loads((ROOT/'runtime/active_sitl.json').read_text())
    verify_managed_process(record)
    directory = Path(record['directory'])
    scenario = json.loads((ROOT/'worlds/dubai_osm/smoke_scenario.json').read_text())
    zones = [RestrictedZone(z['id'],tuple(map(tuple,z['polygon_ned']))) for z in scenario['restricted_zones']]
    world = OSMWorld.from_directory(ROOT/'worlds/dubai_osm', zones)
    altitude = scenario['altitude_m_agl']
    # Two metres of tracking reserve beyond the six metre runtime static margin.
    route = world.plan(tuple(scenario['start_ned']),tuple(scenario['goal_ned']),altitude,8,8)
    path = shortcut_route(world,route.path_ned,altitude,8)
    origin = world.manifest['origin_wgs84']
    frame = LocalMapFrame(origin['latitude'],origin['longitude'])
    link = Link()
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),status='PREFLIGHT',
                  path_ned=path, altitude_m=altitude, planning_clearance_m=8, runtime_clearance_m=6,
                  scope='Static map feedback route flight; no camera or physics mesh collisions')
    took_control = False
    try:
        link.wait(lambda: 'HEARTBEAT' in link.messages,10,'heartbeat')
        hb = link.messages['HEARTBEAT']
        if hb.autopilot != 3 or hb.type != 2 or link.armed():
            raise RuntimeError('Expected disarmed ArduPilot quadrotor')
        link.connection.mav.request_data_stream_send(1,1,mavutil.mavlink.MAV_DATA_STREAM_ALL,20,1)
        link.command(mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,49)  # GPS_GLOBAL_ORIGIN
        link.wait(lambda: all(k in link.messages for k in ('GPS_GLOBAL_ORIGIN','LOCAL_POSITION_NED','GLOBAL_POSITION_INT','GPS_RAW_INT')),30,'position and EKF origin')
        ekf = link.messages['GPS_GLOBAL_ORIGIN']
        north0,east0 = frame.to_map(ekf.latitude/1e7, ekf.longitude/1e7)
        down0 = -ekf.altitude/1000.
        report['ekf_origin_map_ned'] = [north0,east0,down0]

        def state():
            m = link.messages['LOCAL_POSITION_NED']
            return VehicleState((north0+m.x,east0+m.y,down0+m.z),(m.vx,m.vy,m.vz),link.times['LOCAL_POSITION_NED'])

        link.wait(lambda: link.messages['GPS_RAW_INT'].fix_type >= 3,20,'GPS fix')
        initial = state()
        if math.dist(initial.position[:2],path[0]) > 2 or abs(initial.position[2]) > 1:
            raise RuntimeError('Simulator is not at scenario start/ground')
        if world.violation(*initial.position, horizontal_clearance_m=8):
            raise RuntimeError('Unsafe takeoff location')
        print(f'PREFLIGHT PASS: {len(path)} waypoints; start={initial.position}; fly={args.fly}',flush=True)
        if not args.fly:
            report['status']='PREFLIGHT_PASS'
            return
        link.mode(4)  # GUIDED
        took_control = True  # Also recover if the arm ACK is lost after acceptance.
        link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1)
        link.acknowledged(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        took_control = True
        link.wait(link.armed,10,'arming')
        link.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,altitude)
        link.acknowledged(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        link.wait(lambda: -state().position[2] > altitude-.3,40,'takeoff')
        follower = RouteFollower(path,altitude)
        last, started, next_print = time.monotonic(), time.monotonic(), 0.
        previous_state = state()
        max_error = 0.
        with (directory/'flight_trace.jsonl').open('w',encoding='utf-8') as trace:
            while not follower.done:
                tick = time.monotonic()
                if tick-started > 1200:
                    raise RuntimeError('Flight deadline exceeded')
                link.pump()
                current = state()
                if tick-link.times['HEARTBEAT'] > 2.5 or not link.armed() or link.messages['HEARTBEAT'].custom_mode != 4:
                    raise RuntimeError('Lost GUIDED mode, arming, or heartbeat')
                dt = tick-last
                if dt < .01:
                    time.sleep(.05)
                    continue
                last = tick
                command = follower.propose(current,dt)
                error = follower.cross_track_error(current.position)
                max_error = max(error,max_error)
                violation = gate_velocity(world,current,command,time.monotonic(),error)
                violation = violation or world.segment_violation(previous_state.position,current.position,6)
                if violation:
                    raise RuntimeError('Safety gate: '+violation)
                previous_state = current
                link.velocity(command)
                trace.write(json.dumps(dict(t=tick-started,position=current.position,velocity=current.velocity,
                                            command=command,target_index=follower.index,cross_track_m=error))+'\n')
                if tick >= next_print:
                    print(f'FLIGHT waypoint {follower.index}/{len(path)-1}, position={tuple(round(v,1) for v in current.position)}, cross-track={error:.2f}m',flush=True)
                    trace.flush()
                    next_print = tick+20
                time.sleep(max(0.,.05-(time.monotonic()-tick)))
        report['goal_error_m']=math.dist(state().position[:2],path[-1])
        report['max_cross_track_m']=max_error
        report['flight_seconds']=time.monotonic()-started
        link.velocity((0,0,0))
        link.mode(9)  # LAND at the checked destination
        link.wait(lambda: not link.armed(),60,'landing and disarming')
        took_control = False
        report['status']='PASS'
        print('PASS: destination reached, landed and disarmed',flush=True)
    except BaseException as exc:
        report['status']='FAIL'
        report['error']=str(exc)
        if took_control:
            try:
                link.velocity((0,0,0))
                link.mode(5)  # LOITER; no blind automatic descent on failure
                report['failure_action']='LOITER confirmed'
            except Exception as recovery:
                report['failure_action']='Could not confirm LOITER: '+str(recovery)
        raise
    finally:
        report['velocity_packets_sent']=link.sent
        report['command_long_packets_sent']=link.command_count
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        (directory/('flight_result.json' if args.fly else 'preflight_result.json')).write_text(json.dumps(report,indent=2))
        link.connection.close()


if __name__ == '__main__':
    main()
