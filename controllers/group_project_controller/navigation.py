"""Map planning and waypoint-control helpers for the 3006ICT group project.

This module deliberately contains no Webots API calls so that planning and control
logic can be validated offline.  The runtime controller supplies the current pose
and applies the returned wheel-speed commands.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]


@dataclass
class PlannedPath:
    """A grid path plus the weighted cost used for station selection."""

    cells: List[GridCell]
    cost: float


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def manhattan(a: GridCell, b: GridCell) -> float:
    """Admissible A* heuristic for the four-connected grid used here."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def clearance_map(grid: np.ndarray) -> np.ndarray:
    """Return distance-to-obstacle (in grid cells) for every free cell."""
    free = (grid == 0).astype(np.uint8)
    return cv2.distanceTransform(free, cv2.DIST_L2, 5)


def hard_inflate(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """Utility used for diagnostics; not used by the runtime planner.

    A one-cell hard inflation was tested but made some supplied interior station
    observation positions unreachable, so the final planner uses a soft
    clearance cost instead.
    """
    if radius_cells <= 0:
        return grid.copy()
    size = radius_cells * 2 + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)


class AStarPlanner:
    """Four-connected A* with a soft obstacle-clearance penalty.

    The original occupancy grid remains authoritative: occupied cells are never
    traversed.  Free cells close to obstacles receive an extra cost, which keeps
    normal paths away from walls/barriers without making narrow approaches to
    interior observation stations impossible.
    """

    def __init__(
        self,
        grid: np.ndarray,
        clearance_cells: float = 2.0,
        clearance_weight: float = 1.5,
    ) -> None:
        self.grid = np.asarray(grid, dtype=np.uint8)
        if self.grid.ndim != 2:
            raise ValueError("occupancy grid must be two-dimensional")
        self.rows, self.cols = self.grid.shape
        self.clearance = clearance_map(self.grid)
        self.clearance_cells = float(clearance_cells)
        self.clearance_weight = float(clearance_weight)

    def in_bounds(self, cell: GridCell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and self.grid[cell] == 0

    def nearest_free(self, cell: GridCell, max_radius: int = 4) -> Optional[GridCell]:
        """Return cell itself if free, otherwise the nearest free cell nearby."""
        if self.is_free(cell):
            return cell
        r0, c0 = cell
        for radius in range(1, max_radius + 1):
            candidates: List[Tuple[float, GridCell]] = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    candidate = (r0 + dr, c0 + dc)
                    if self.is_free(candidate):
                        candidates.append((math.hypot(dr, dc), candidate))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
        return None

    def _step_cost(self, cell: GridCell) -> float:
        distance = float(self.clearance[cell])
        penalty = max(0.0, self.clearance_cells - distance) * self.clearance_weight
        return 1.0 + penalty

    def plan(self, start: GridCell, goal: GridCell) -> Optional[PlannedPath]:
        start = self.nearest_free(start)
        goal = self.nearest_free(goal)
        if start is None or goal is None:
            return None

        frontier: List[Tuple[float, float, GridCell]] = []
        heapq.heappush(frontier, (manhattan(start, goal), 0.0, start))
        came_from: dict[GridCell, GridCell] = {}
        g_cost: dict[GridCell, float] = {start: 0.0}

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost != g_cost.get(current):
                continue
            if current == goal:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                return PlannedPath(cells=cells, cost=current_cost)

            r, c = current
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (r + dr, c + dc)
                if not self.is_free(neighbour):
                    continue
                new_cost = current_cost + self._step_cost(neighbour)
                if new_cost < g_cost.get(neighbour, float("inf")):
                    g_cost[neighbour] = new_cost
                    came_from[neighbour] = current
                    priority = new_cost + manhattan(neighbour, goal)
                    heapq.heappush(frontier, (priority, new_cost, neighbour))
        return None


def compress_collinear(cells: Sequence[GridCell]) -> List[GridCell]:
    """Remove redundant cells while preserving all 90-degree grid turns."""
    if len(cells) <= 2:
        return list(cells)
    result = [cells[0]]
    previous_direction = None
    for index in range(1, len(cells)):
        direction = (
            cells[index][0] - cells[index - 1][0],
            cells[index][1] - cells[index - 1][1],
        )
        if previous_direction is not None and direction != previous_direction:
            result.append(cells[index - 1])
        previous_direction = direction
    result.append(cells[-1])
    return result


class WaypointFollower:
    """GPS/IMU feedback controller for a sequence of world-coordinate waypoints."""

    def __init__(
        self,
        max_speed: float = 6.28,
        cruise_speed: float = 5.0,
        waypoint_tolerance: float = 0.075,
        final_tolerance: float = 0.07,
        turn_in_place_angle: float = 0.45,
        heading_kp: float = 2.4,
        rotate_kp: float = 2.8,
    ) -> None:
        self.max_speed = float(max_speed)
        self.cruise_speed = float(cruise_speed)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.final_tolerance = float(final_tolerance)
        self.turn_in_place_angle = float(turn_in_place_angle)
        self.heading_kp = float(heading_kp)
        self.rotate_kp = float(rotate_kp)
        self.waypoints: List[WorldPoint] = []
        self.index = 0

    def set_waypoints(self, waypoints: Iterable[WorldPoint]) -> None:
        self.waypoints = list(waypoints)
        self.index = 0

    def clear(self) -> None:
        self.waypoints = []
        self.index = 0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def current_waypoint(self) -> Optional[WorldPoint]:
        if self.finished:
            return None
        return self.waypoints[self.index]

    def distance_to_current(self, x: float, y: float) -> float:
        waypoint = self.current_waypoint
        if waypoint is None:
            return 0.0
        return math.hypot(waypoint[0] - x, waypoint[1] - y)

    def command(self, pose: Tuple[float, float, float], speed_scale: float = 1.0) -> Tuple[float, float, bool]:
        """Return left speed, right speed and completion flag."""
        x, y, yaw = pose
        while not self.finished:
            wx, wy = self.waypoints[self.index]
            distance = math.hypot(wx - x, wy - y)
            tolerance = self.final_tolerance if self.index == len(self.waypoints) - 1 else self.waypoint_tolerance
            if distance > tolerance:
                break
            self.index += 1

        if self.finished:
            return 0.0, 0.0, True

        wx, wy = self.waypoints[self.index]
        distance = math.hypot(wx - x, wy - y)
        desired = math.atan2(wy - y, wx - x)
        error = wrap_angle(desired - yaw)

        if abs(error) > self.turn_in_place_angle:
            turn = float(np.clip(self.rotate_kp * error, -2.8, 2.8))
            if 0.0 < abs(turn) < 0.65:
                turn = math.copysign(0.65, turn)
            left = -turn
            right = turn
        else:
            base = self.cruise_speed * float(np.clip(speed_scale, 0.0, 1.0))
            if self.index == len(self.waypoints) - 1:
                base *= float(np.clip(distance / 0.25, 0.35, 1.0))
            turn = self.heading_kp * error
            left = base - turn
            right = base + turn

        left = float(np.clip(left, -self.max_speed, self.max_speed))
        right = float(np.clip(right, -self.max_speed, self.max_speed))
        return left, right, False
