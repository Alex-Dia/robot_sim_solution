"""
3006ICT Group Project - Vision-Guided Autonomous Search and Navigation

Final simplified pipeline:
    read mission -> localise -> choose lowest-cost unvisited station ->
    A* to a map-validated camera inspection pose -> face station centre ->
    target-specific ORB over multiple frames -> reject and continue OR
    A* final approach to the designated observation point -> stop.

Important design rule:
    The station containing the requested target is determined only from RGB camera
    observations.  The supplied map/GPS/IMU are used for localisation/navigation.
"""

from __future__ import annotations

import json
import math
from collections import deque
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from controller import Robot

from navigation import AStarPlanner, WaypointFollower, compress_collinear, wrap_angle
from project_utils import CONFIG, ROOT, grid_to_world, world_to_grid
from vision import TargetMatcher


# ------------------------------------------------------------------
# Webots setup
# ------------------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
camera = robot.getDevice("camera")
ps = [robot.getDevice(f"ps{i}") for i in range(8)]
gps = robot.getDevice("gps")
imu = robot.getDevice("imu")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

camera.enable(timestep)
gps.enable(timestep)
imu.enable(timestep)
for sensor in ps:
    sensor.enable(timestep)

MAX_SPEED = 6.28
GRID = np.load(ROOT / "maps" / "occupancy_grid.npy")
MISSION = json.loads((ROOT / "config" / "assessment_mission.json").read_text())
target = MISSION["target"]

# Navigation/safety tuning.
PS_SLOW = 70.0
PS_EMERGENCY = 120.0
PS_CLEAR = 55.0
RECOVERY_TURN_SPEED = 1.8
STUCK_TIMEOUT_S = 3.5
MISSION_LIMIT_S = 240.0

# Vision/inspection tuning.
# Live camera evidence showed that simply stopping at (or only 0.17 m behind) the
# designated observation point can crop the poster.  Revision 4 therefore searches
# for a collision-free camera pose farther from the station, while preserving a clear
# line of sight.  The robot then aims at the station centre rather than relying on a
# fixed yaw.  These constants describe camera/navigation geometry only; they do NOT
# encode which target is displayed at any station.
STATION_CENTER_AHEAD_M = 0.33
PREFERRED_VIEW_DISTANCE_M = 0.72
PREFERRED_VIEW_BAND_M = (0.66, 0.80)
MIN_VIEW_DISTANCE_M = 0.33
MAX_VIEW_DISTANCE_M = 0.88
INSPECTION_VIEW_DISTANCES_M = (0.72, 0.68, 0.76, 0.64, 0.80, 0.60, 0.56, 0.52, 0.48, 0.44, 0.40, 0.36, 0.33, 0.84, 0.88)
MAX_VIEW_OBLIQUITY = math.radians(4.0)
ALIGN_TOLERANCE = math.radians(2.5)
ALIGN_STABLE_STEPS = 6
OBSERVATION_WARMUP_STEPS = 6
OBSERVATION_SAMPLES = 5
OBSERVATION_REQUIRED_VOTES = 3

SAVE_DEBUG_FRAMES = True


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------
def set_speed(left: float, right: float) -> None:
    left = np.clip(left, -MAX_SPEED, MAX_SPEED)
    right = np.clip(right, -MAX_SPEED, MAX_SPEED)
    left_motor.setVelocity(float(left))
    right_motor.setVelocity(float(right))


def get_pose() -> tuple[float, float, float]:
    """Return (x, y, yaw) from GPS + InertialUnit."""
    x, y, _ = gps.getValues()
    yaw = imu.getRollPitchYaw()[2]
    return x, y, yaw


def camera_bgr() -> np.ndarray:
    h, w = camera.getHeight(), camera.getWidth()
    image = np.frombuffer(camera.getImage(), np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def proximity_values() -> list[float]:
    return [sensor.getValue() for sensor in ps]


def front_proximity(values: list[float]) -> float:
    return max(values[0], values[1], values[6], values[7])


def station_center_point(station: dict) -> tuple[float, float]:
    """Approximate station centre from the supplied observation pose geometry.

    The observation yaw points from the designated observation position toward the
    station.  This is used only to aim the RGB camera and choose a clear viewing pose;
    target identity still comes exclusively from camera pixels.
    """
    ox, oy = station["observe"]
    yaw = station["observe_yaw"]
    return (
        ox + STATION_CENTER_AHEAD_M * math.cos(yaw),
        oy + STATION_CENTER_AHEAD_M * math.sin(yaw),
    )


def _line_of_sight_free(start_xy: tuple[float, float], end_xy: tuple[float, float]) -> bool:
    """Return True when the occupancy grid is clear between two world points.

    The ray stops at the designated observation position, which is in front of the
    station structure, so the station's own occupied cell does not invalidate the ray.
    """
    distance = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    samples = max(2, int(math.ceil(distance / 0.04)) + 1)
    for t in np.linspace(0.0, 1.0, samples):
        x = start_xy[0] + t * (end_xy[0] - start_xy[0])
        y = start_xy[1] + t * (end_xy[1] - start_xy[1])
        row, col = world_to_grid(x, y)
        if row < 0 or col < 0 or row >= GRID.shape[0] or col >= GRID.shape[1]:
            return False
        if GRID[row, col] != 0:
            return False
    return True


def inspection_candidates(station: dict) -> list[tuple[float, float]]:
    """Return frontal inspection cells centred on the station normal.

    The robot must inspect from the centre of a free occupancy-grid square that lies
    directly in front of the station centre.  No lateral-offset candidates are used.
    We try the preferred viewing distance first, then nearby distances on the same
    centre line if that grid cell is blocked or unreachable.
    """
    centre = station_center_point(station)
    yaw = station["observe_yaw"]
    candidates: list[tuple[float, float, float]] = []
    seen_cells: set[tuple[int, int]] = set()

    for distance in INSPECTION_VIEW_DISTANCES_M:
        desired_x = centre[0] - distance * math.cos(yaw)
        desired_y = centre[1] - distance * math.sin(yaw)
        row, col = world_to_grid(desired_x, desired_y)

        if (row, col) in seen_cells:
            continue
        seen_cells.add((row, col))
        if row < 0 or col < 0 or row >= GRID.shape[0] or col >= GRID.shape[1]:
            continue
        if GRID[row, col] != 0:
            continue

        # Use the exact centre of the occupancy-grid square as the stopping point.
        x, y = grid_to_world(row, col)
        look_yaw = math.atan2(centre[1] - y, centre[0] - x)
        obliquity = abs(wrap_angle(look_yaw - yaw))
        if obliquity > MAX_VIEW_OBLIQUITY:
            continue

        # Check the view ray only up to the supplied observation point so the
        # station's own occupied geometry does not invalidate the line of sight.
        if not _line_of_sight_free((x, y), tuple(station["observe"])):
            continue

        actual_distance = math.hypot(centre[0] - x, centre[1] - y)
        if MIN_VIEW_DISTANCE_M <= actual_distance <= MAX_VIEW_DISTANCE_M:
            candidates.append((x, y, actual_distance))

    candidates.sort(key=lambda item: abs(item[2] - PREFERRED_VIEW_DISTANCE_M))
    return [(x, y) for x, y, _ in candidates]


def final_observation_point(station: dict) -> tuple[float, float]:
    return tuple(station["observe"])


# ------------------------------------------------------------------
# Planning/control components
# ------------------------------------------------------------------
planner = AStarPlanner(GRID, clearance_cells=2.0, clearance_weight=1.5)
follower = WaypointFollower(max_speed=MAX_SPEED, cruise_speed=5.3, final_tolerance=0.05)
# The dedicated inspection pose makes the poster roughly 60-80 px in the
# 160x120 camera frame.  A 64 px reference gives ORB a comparable feature scale.
# Recognition answers only: "Is the requested mission target in this frame?"
matcher = TargetMatcher(
    ROOT,
    target,
    reference_size=64,
    min_good_matches=18,
    min_inliers=9,
    min_inlier_ratio=0.40,
)

DEBUG_DIR = ROOT / "debug_frames"
if SAVE_DEBUG_FRAMES:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for old in DEBUG_DIR.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass


class State(Enum):
    SELECT_STATION = auto()
    PLAN_PATH = auto()
    FOLLOW_PATH = auto()
    RECOVERY = auto()
    ALIGN_STATION = auto()
    OBSERVE = auto()
    COMPLETE = auto()
    FAILED = auto()


state = State.SELECT_STATION
visited: set[str] = set()
inspection_order: list[str] = []   # Record the order of stations visually inspected
current_station: dict | None = None
selected_plan = None
final_approach = False
current_inspection_point: tuple[float, float] | None = None

station_best_scores: dict[str, float] = {}

# Observation state
align_stable_count = 0
observation_warmup = 0
observation_samples = 0
vision_votes: deque[bool] = deque(maxlen=OBSERVATION_SAMPLES)

# Recovery/progress state
recovery_turn_sign = 1.0
recovery_clear_count = 0
last_progress_time = 0.0
best_waypoint_distance = float("inf")
last_waypoint_index = -1


def current_navigation_goal() -> tuple[float, float]:
    if current_station is None:
        raise RuntimeError("current_navigation_goal() called without a station")
    if final_approach:
        return final_observation_point(current_station)
    if current_inspection_point is None:
        raise RuntimeError("Inspection pose has not been selected")
    return current_inspection_point


def best_inspection_plan(station: dict, start_cell: tuple[int, int]):
    """Return (pose, plan) for the best reachable camera inspection pose."""
    ranked = []
    centre = station_center_point(station)
    yaw = station["observe_yaw"]

    for candidate in inspection_candidates(station):
        goal_cell = world_to_grid(candidate[0], candidate[1])
        plan = planner.plan(start_cell, goal_cell)
        if plan is None:
            continue
        view_distance = math.hypot(centre[0] - candidate[0], centre[1] - candidate[1])
        look_yaw = math.atan2(centre[1] - candidate[1], centre[0] - candidate[0])
        obliquity = abs(wrap_angle(look_yaw - yaw))
        ranked.append((
            0 if PREFERRED_VIEW_BAND_M[0] <= view_distance <= PREFERRED_VIEW_BAND_M[1] else 1,
            obliquity,
            abs(view_distance - PREFERRED_VIEW_DISTANCE_M),
            plan.cost,
            candidate,
            plan,
        ))

    if not ranked:
        return None, None
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, _, _, _, pose, plan = ranked[0]
    return pose, plan


def choose_station(pose: tuple[float, float, float]):
    """Choose a nearby station while also considering the next likely journey."""
    start_cell = world_to_grid(pose[0], pose[1])
    candidates = []

    remaining = [
        station for station in CONFIG["stations"]
        if station["id"] not in visited
    ]

    for station in remaining:
        inspect, plan = best_inspection_plan(station, start_cell)

        if plan is None:
            continue

        # Cost to reach this station.
        current_cost = plan.cost

        # Estimate the cheapest journey from this station
        # to another unvisited station.
        next_cost = 0.0

        others = [
            other for other in remaining
            if other["id"] != station["id"]
        ]

        if others:
            candidate_cell = world_to_grid(inspect[0], inspect[1])
            future_costs = []

            for other in others:
                _, next_plan = best_inspection_plan(
                    other,
                    candidate_cell
                )

                if next_plan is not None:
                    future_costs.append(next_plan.cost)

            if future_costs:
                next_cost = min(future_costs)

        # Current journey matters most, but avoid poor follow-up positions.
        score = current_cost + 0.5 * next_cost

        candidates.append(
            (
                score,
                current_cost,
                next_cost,  # Store estimated follow-up cost
                station["id"],
                station,
                inspect,
                plan
            )
        )

    if not candidates:
        return None, None, None

    # Rank candidates by total score, then current cost and station ID:
    candidates.sort(
        key=lambda item: (item[0], item[1], item[3])
    )

    # Display station costs to verify the search strategy:
    print(" ")
    print("[SEARCH] Candidate station costs:")

    for candidate in candidates:
        score, current_cost, next_cost, station_id, _, _, _ = candidate

        # Display each candidate's evaluation:
        print(
            f"  {station_id}: "
            f"current={current_cost:.2f}, "
            f"next={next_cost:.2f}, "
            f"score={score:.2f}"
        )

    # Select the candidate with the lowest ranked score. _ ignores the three values.
    best_score, _, _, _, station, inspect, plan = candidates[0]

    # Display the best candidate selected.
    print(" ")
    print(
        f"[SEARCH] Selected {station['id']} "
        f"with lowest score={best_score:.2f}"
    )

    return station, inspect, plan


def make_world_waypoints(plan, exact_goal: tuple[float, float]) -> list[tuple[float, float]]:
    cells = compress_collinear(plan.cells)
    waypoints = [grid_to_world(row, col) for row, col in cells[1:]]
    if not waypoints or math.hypot(waypoints[-1][0] - exact_goal[0], waypoints[-1][1] - exact_goal[1]) > 0.015:
        waypoints.append(exact_goal)
    return waypoints


def reset_progress_tracking(now: float, pose: tuple[float, float, float]) -> None:
    global last_progress_time, best_waypoint_distance, last_waypoint_index
    last_progress_time = now
    best_waypoint_distance = follower.distance_to_current(pose[0], pose[1])
    last_waypoint_index = follower.index


def begin_recovery(values: list[float], now: float) -> None:
    global state, recovery_turn_sign, recovery_clear_count, last_progress_time
    left_strength = values[6] + values[7]
    right_strength = values[0] + values[1]
    recovery_turn_sign = -1.0 if left_strength > right_strength else 1.0
    recovery_clear_count = 0
    last_progress_time = now
    set_speed(0.0, 0.0)
    state = State.RECOVERY
    print("[SAFETY] Proximity emergency -> recovery/replan")


def align_command(current_yaw: float, desired_yaw: float) -> tuple[float, float, bool]:
    error = wrap_angle(desired_yaw - current_yaw)
    if abs(error) <= ALIGN_TOLERANCE:
        return 0.0, 0.0, True
    turn = float(np.clip(2.8 * error, -2.4, 2.4))
    if abs(turn) < 0.55:
        turn = math.copysign(0.55, turn)
    return -turn, turn, False


def save_debug_frame(frame: np.ndarray, result, station_id: str) -> None:
    if not SAVE_DEBUG_FRAMES:
        return
    raw_name = DEBUG_DIR / f"raw_{station_id}_sample1.png"
    annotated_name = DEBUG_DIR / f"annotated_{station_id}_sample1.png"
    cv2.imwrite(str(raw_name), frame)
    cv2.imwrite(str(annotated_name), matcher.draw_debug(frame, result))


def reject_current_station(reason: str) -> None:
    global current_station, current_inspection_point, state
    station_id = current_station["id"]
    visited.add(station_id)
    print(f"[VISION] {station_id} rejected ({reason}); visited={sorted(visited)}")
    current_station = None
    current_inspection_point = None
    state = State.SELECT_STATION


# ------------------------------------------------------------------
# Main autonomous state machine
# ------------------------------------------------------------------
def main() -> None:
    global state, current_station, selected_plan, final_approach, current_inspection_point
    global align_stable_count, observation_warmup, observation_samples
    global recovery_clear_count, last_progress_time, best_waypoint_distance, last_waypoint_index

    print(" ")
    print("Group-project controller started (simplified target-only ORB version).")
    print("Mission target:", target)
    print("Stations:", [s["id"] for s in CONFIG["stations"]])
    print("Camera:", camera.getWidth(), "x", camera.getHeight())
    print(
        f"Inspection view target: {PREFERRED_VIEW_DISTANCE_M:.2f} m from station centre "
        f"(allowed {MIN_VIEW_DISTANCE_M:.2f}-{MAX_VIEW_DISTANCE_M:.2f} m)"
    )

    # Record simulation time at the start of the mission:
    mission_start_time = robot.getTime()

    while robot.step(timestep) != -1:
        now = robot.getTime()
        pose = get_pose()
        values = proximity_values()

        if now >= MISSION_LIMIT_S and state not in (State.COMPLETE, State.FAILED):
            print("[FAIL] 4:00 mission time limit reached.")
            set_speed(0.0, 0.0)
            state = State.FAILED

        if state == State.SELECT_STATION:
            set_speed(0.0, 0.0)
            final_approach = False
            current_station, current_inspection_point, selected_plan = choose_station(pose)
            if current_station is None:
                print("[FAIL] All reachable stations were inspected but the target was not visually confirmed.")
                print("[FAIL] Best ORB scores:", station_best_scores)
                if SAVE_DEBUG_FRAMES:
                    print(f"[DEBUG] Inspect raw + annotated camera frames in: {DEBUG_DIR}")
                state = State.FAILED
                continue
            centre = station_center_point(current_station)
            view_distance = math.hypot(
                centre[0] - current_inspection_point[0],
                centre[1] - current_inspection_point[1],
            )

            print(
                f"[SEARCH] Next station {current_station['id']} "
                f"(A* inspection cost={selected_plan.cost:.2f}, "
                f"view=({current_inspection_point[0]:.2f}, {current_inspection_point[1]:.2f}), "
                f"camera distance={view_distance:.2f} m)"
            )
            state = State.PLAN_PATH

        elif state == State.PLAN_PATH:
            start_cell = world_to_grid(pose[0], pose[1])
            goal = current_navigation_goal()
            goal_cell = world_to_grid(goal[0], goal[1])
            selected_plan = planner.plan(start_cell, goal_cell)
            if selected_plan is None:
                if final_approach:
                    print(f"[FAIL] Target found at {current_station['id']} but final observation point is unreachable.")
                    set_speed(0.0, 0.0)
                    state = State.FAILED
                else:
                    # Current pose may have changed after recovery.  Re-select a
                    # reachable camera pose before giving up on the station.
                    alternate, alternate_plan = best_inspection_plan(current_station, start_cell)
                    if alternate_plan is not None:
                        current_inspection_point = alternate
                        selected_plan = alternate_plan
                        print(
                            f"[PLAN] Inspection pose changed after replan -> "
                            f"({alternate[0]:.2f}, {alternate[1]:.2f})"
                        )
                        goal = alternate
                        goal_cell = world_to_grid(goal[0], goal[1])
                        selected_plan = planner.plan(start_cell, goal_cell)
                    if selected_plan is None:
                        print(f"[WARN] Could not plan to any valid inspection pose for {current_station['id']}; skipping.")
                        visited.add(current_station["id"])
                        current_station = None
                        current_inspection_point = None
                        state = State.SELECT_STATION
                        continue

            waypoints = make_world_waypoints(selected_plan, goal)
            follower.set_waypoints(waypoints)
            reset_progress_tracking(now, pose)
            phase = "FINAL" if final_approach else "INSPECT"
            print(" ")
            print(
                f"[PLAN-{phase}] {current_station['id']}: {len(selected_plan.cells)} grid cells -> "
                f"{len(waypoints)} waypoints"
            )
            state = State.FOLLOW_PATH

        elif state == State.FOLLOW_PATH:
            front = front_proximity(values)
            if front >= PS_EMERGENCY:
                begin_recovery(values, now)
                continue

            speed_scale = 0.55 if front >= PS_SLOW else 1.0
            left, right, done = follower.command(pose, speed_scale=speed_scale)
            set_speed(left, right)

            if done:
                set_speed(0.0, 0.0)
                if final_approach:
                    position_error = math.hypot(
                        pose[0] - current_station["observe"][0],
                        pose[1] - current_station["observe"][1],
                    )
                    if position_error <= 0.20:
                        # Record end of mission for total mission time, in mins and secs:
                        mission_time = now - mission_start_time
                        mission_minutes = int(mission_time // 60)
                        mission_seconds = mission_time % 60

                        print(" ")
                        print(
                            f"[SUCCESS] Target '{target}' confirmed at {current_station['id']}\n"
                            f"          Mission time: {mission_minutes} min {mission_seconds:.2f} s ({mission_time:.2f} s)\n"
                            f"          Stations inspected: {len(inspection_order)}\n"
                            f"          Search order: {' -> '.join(inspection_order)}\n"
                            f"          Final observation-position error: {position_error:.3f} m"
                        )
                        state = State.COMPLETE
                    else:
                        # This should be rare because the follower's final tolerance is 0.07 m.
                        print(
                            f"[FINAL] Position error {position_error:.3f} m > 0.20 m -> replanning exact final approach"
                        )
                        state = State.PLAN_PATH
                else:
                    align_stable_count = 0
                    state = State.ALIGN_STATION
                    print(
                        f"[NAV] Reached inspection pose for {current_station['id']} "
                        f"at ({current_inspection_point[0]:.2f}, {current_inspection_point[1]:.2f})"
                    )
                continue

            if follower.index != last_waypoint_index:
                reset_progress_tracking(now, pose)
            else:
                distance = follower.distance_to_current(pose[0], pose[1])
                if distance < best_waypoint_distance - 0.025:
                    best_waypoint_distance = distance
                    last_progress_time = now
                elif now - last_progress_time >= STUCK_TIMEOUT_S:
                    print("[RECOVERY] No waypoint progress -> replanning from current pose")
                    set_speed(0.0, 0.0)
                    state = State.PLAN_PATH

        elif state == State.RECOVERY:
            front = front_proximity(values)
            if front <= PS_CLEAR:
                recovery_clear_count += 1
                set_speed(0.0, 0.0)
                if recovery_clear_count >= 5:
                    print("[RECOVERY] Clearance restored -> A* replan")
                    state = State.PLAN_PATH
            else:
                recovery_clear_count = 0
                turn = RECOVERY_TURN_SPEED * recovery_turn_sign
                set_speed(-turn, turn)

        elif state == State.ALIGN_STATION:
            centre = station_center_point(current_station)
            desired_yaw = math.atan2(centre[1] - pose[1], centre[0] - pose[0])
            left, right, aligned = align_command(pose[2], desired_yaw)
            set_speed(left, right)
            if aligned:
                align_stable_count += 1
            else:
                align_stable_count = 0

            if align_stable_count >= ALIGN_STABLE_STEPS:
                set_speed(0.0, 0.0)
                observation_warmup = OBSERVATION_WARMUP_STEPS
                observation_samples = 0
                inspection_order.append(current_station["id"])  # Record station inspection order
                vision_votes.clear()
                state = State.OBSERVE
                print(" ")
                print(f"[VISION] {current_station['id']} aligned; collecting 5 target-only ORB frames")

        elif state == State.OBSERVE:
            set_speed(0.0, 0.0)
            if observation_warmup > 0:
                observation_warmup -= 1
                continue

            frame = camera_bgr()
            result = matcher.match(frame)
            observation_samples += 1
            station_id = current_station["id"]
            vision_votes.append(result.is_match)
            station_best_scores[station_id] = max(station_best_scores.get(station_id, 0.0), result.score)

            print(
                f"[VISION] {station_id} sample {observation_samples}/{OBSERVATION_SAMPLES}: "
                f"ORB good={result.good_matches}, inliers={result.inliers}, "
                f"ratio={result.inlier_ratio:.2f}, geometry={result.geometry_valid}, "
                f"vote={result.is_match} ({sum(vision_votes)}/{len(vision_votes)})"
            )

            if SAVE_DEBUG_FRAMES and observation_samples == 1:
                save_debug_frame(frame, result, station_id)

            if sum(vision_votes) >= OBSERVATION_REQUIRED_VOTES:
                set_speed(0.0, 0.0)
                final_approach = True
                print(
                    f"[VISION] Target '{target}' confirmed at {station_id} "
                    f"({sum(vision_votes)}/{observation_samples} ORB votes) -> final approach"
                )
                state = State.PLAN_PATH
                continue

            if observation_samples >= OBSERVATION_SAMPLES:
                reject_current_station(
                    f"target-only ORB vote {sum(vision_votes)}/{OBSERVATION_SAMPLES}"
                )

        elif state == State.COMPLETE:
            set_speed(0.0, 0.0)

        elif state == State.FAILED:
            set_speed(0.0, 0.0)


if __name__ == "__main__":
    main()
