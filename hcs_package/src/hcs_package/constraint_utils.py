"""Constraint parsing utilities.

PathConstraints become path-relative corridor_bounds for the MPC.
PolygonConstraints / RectangleConstraints stay as Cartesian constraints.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from .constraints import (
    ConstraintConfig, ConstraintRegion, ConstraintType,
    PathConstraint, PolygonConstraint, RectangleConstraint
)


def merge_rectangles_to_polygon(rects: List[RectangleConstraint]) -> PolygonConstraint:
    """Merge axis-aligned rectangles into a single polygon (CCW outer boundary).

    Discretises the plane at all rectangle edges, marks filled cells, extracts
    oriented boundary edges, and chains them into a closed polygon.
    """
    if len(rects) == 1:
        r = rects[0]
        return PolygonConstraint(vertices=[
            (r.x, r.y), (r.x + r.width, r.y),
            (r.x + r.width, r.y + r.height), (r.x, r.y + r.height),
        ])

    def _dedup_sorted(vals, tol=1e-9):
        vals = sorted(set(vals))
        if not vals:
            return vals
        out = [vals[0]]
        for v in vals[1:]:
            if v - out[-1] > tol:
                out.append(v)
        return out

    xs = _dedup_sorted([v for r in rects for v in (r.x, r.x + r.width)])
    ys = _dedup_sorted([v for r in rects for v in (r.y, r.y + r.height)])

    # Occupancy grid: cell (i,j) filled if any rectangle covers it
    nx, ny = len(xs) - 1, len(ys) - 1
    grid = [[False] * ny for _ in range(nx)]
    for r in rects:
        for i in range(nx):
            if xs[i] >= r.x - 1e-12 and xs[i + 1] <= r.x + r.width + 1e-12:
                for j in range(ny):
                    if ys[j] >= r.y - 1e-12 and ys[j + 1] <= r.y + r.height + 1e-12:
                        grid[i][j] = True

    def _filled(i, j):
        if 0 <= i < nx and 0 <= j < ny:
            return grid[i][j]
        return False

    # Oriented boundary edges (CCW): filled cell is to the LEFT of each edge.
    edges = []
    for i in range(nx):
        for j in range(ny):
            if not grid[i][j]:
                continue
            if not _filled(i, j - 1):
                edges.append(((xs[i], ys[j]), (xs[i + 1], ys[j])))
            if not _filled(i + 1, j):
                edges.append(((xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1])))
            if not _filled(i, j + 1):
                edges.append(((xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])))
            if not _filled(i - 1, j):
                edges.append(((xs[i], ys[j + 1]), (xs[i], ys[j])))

    if not edges:
        min_x = min(r.x for r in rects)
        min_y = min(r.y for r in rects)
        max_x = max(r.x + r.width for r in rects)
        max_y = max(r.y + r.height for r in rects)
        return PolygonConstraint(vertices=[
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y),
        ])

    def _key(pt):
        return (round(pt[0], 10), round(pt[1], 10))

    edge_map = {}
    for e in edges:
        k = _key(e[0])
        edge_map.setdefault(k, []).append(e)

    # Chain edges; at junctions prefer the rightmost turn to trace the outer
    # boundary and avoid entering concavities.
    import math

    def _angle(dx, dy):
        return math.atan2(dy, dx)

    used = set()
    start_edge = min(edges, key=lambda e: (e[0][1], e[0][0]))
    polygon = []
    current = start_edge

    for _ in range(len(edges) + 1):
        sk = _key(current[0])
        ek = _key(current[1])
        edge_id = (sk, ek)
        if edge_id in used:
            break
        used.add(edge_id)
        polygon.append(current[0])

        in_dx = current[0][0] - current[1][0]
        in_dy = current[0][1] - current[1][1]
        in_angle = _angle(in_dx, in_dy)

        candidates = edge_map.get(ek, [])
        best = None
        best_turn = float('inf')
        for c in candidates:
            ck = (_key(c[0]), _key(c[1]))
            if ck in used:
                continue
            out_dx = c[1][0] - c[0][0]
            out_dy = c[1][1] - c[0][1]
            out_angle = _angle(out_dx, out_dy)
            turn = (out_angle - in_angle) % (2 * math.pi)
            if turn < best_turn:
                best_turn = turn
                best = c
        if best is None:
            break
        current = best

    cleaned = []
    n = len(polygon)
    for i in range(n):
        p0 = polygon[(i - 1) % n]
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross) > 1e-12:
            cleaned.append(p1)

    return PolygonConstraint(vertices=cleaned if cleaned else polygon)


def parse_constraints_from_json(constraints_dict: Dict[str, Any]) -> Optional[ConstraintConfig]:
    """Parse constraints from a JSON dictionary, returning None if empty."""
    if not constraints_dict:
        return None

    regions = []

    if "left_boundary" in constraints_dict and "right_boundary" in constraints_dict:
        left_boundary = constraints_dict["left_boundary"]
        right_boundary = constraints_dict["right_boundary"]

        if left_boundary:
            regions.append(ConstraintRegion(
                constraint_type=ConstraintType.KEEP_IN,
                geometry=PathConstraint(
                    path=[tuple(p) for p in left_boundary],
                    width=None
                )
            ))

        if right_boundary:
            regions.append(ConstraintRegion(
                constraint_type=ConstraintType.KEEP_IN,
                geometry=PathConstraint(
                    path=[tuple(p) for p in right_boundary],
                    width=None
                )
            ))

    if "regions" in constraints_dict:
        for region_dict in constraints_dict["regions"]:
            if not region_dict.get("enabled", True):
                continue
            
            constraint_type_str = region_dict.get("constraint_type", "keep_out")
            constraint_type = ConstraintType.KEEP_OUT if constraint_type_str == "keep_out" else ConstraintType.KEEP_IN
            
            geometry_dict = region_dict.get("geometry", {})
            geometry_type = geometry_dict.get("type", "")
            
            geometry = None
            if geometry_type == "rectangle":
                geometry = RectangleConstraint(
                    x=geometry_dict["x"],
                    y=geometry_dict["y"],
                    width=geometry_dict["width"],
                    height=geometry_dict["height"]
                )
            elif geometry_type == "polygon":
                geometry = PolygonConstraint(
                    vertices=[tuple(v) for v in geometry_dict["vertices"]]
                )
            elif geometry_type == "line" or geometry_type == "path":
                path_points = geometry_dict.get("path", [])
                if not path_points and "start" in geometry_dict and "end" in geometry_dict:
                    path_points = [geometry_dict["start"], geometry_dict["end"]]
                geometry = PathConstraint(
                    path=[tuple(p) for p in path_points],
                    width=geometry_dict.get("width")
                )
            
            if geometry is not None:
                region = ConstraintRegion(
                    constraint_type=constraint_type,
                    geometry=geometry
                )
                if "margin" in region_dict:
                    region.margin = region_dict["margin"]
                regions.append(region)

    if not regions:
        return None

    # Merge keep_in rectangles into a single polygon so the cursor must be
    # inside the union, not the intersection.
    keep_in_rects = [r for r in regions
                     if r.constraint_type == ConstraintType.KEEP_IN
                     and isinstance(r.geometry, RectangleConstraint)]
    if len(keep_in_rects) > 1:
        merged = merge_rectangles_to_polygon(
            [r.geometry for r in keep_in_rects]
        )
        regions = [r for r in regions if r not in keep_in_rects]
        margin = keep_in_rects[0].margin
        regions.append(ConstraintRegion(
            constraint_type=ConstraintType.KEEP_IN,
            geometry=merged,
            margin=margin,
        ))

    config = ConstraintConfig(regions=regions)
    config.coordinate_system = constraints_dict.get("coordinate_system", "normalized")
    config.default_margin = constraints_dict.get("default_margin", 0.0)
    
    return config


def convert_constraints_to_corridor_bounds(
    constraints: Optional[ConstraintConfig],
    reference_path,
    default_margin: float = 0.0,
    screen_width: Optional[float] = None,
    screen_height: Optional[float] = None,
    max_bound: Optional[float] = 0.1,
) -> Optional[Tuple[Callable, Callable]]:
    """Convert PathConstraint regions to path-relative corridor bounds.

    The returned functions ``left_bound(s)`` and ``right_bound(s)`` give the
    maximum allowed lateral deviation from the reference path at arc-length
    *s*.  Only :class:`PathConstraint` regions with a ``width`` attribute are
    considered; Rectangle / Polygon regions are ignored (they are enforced
    separately as Cartesian constraints).

    Args (in addition to the constraint config):
        max_bound: Per-side clamp on the returned bounds (m). This exists
            ONLY to condition the planner's lateral-corridor QP/penalty — a
            +-5 m corridor is numerically useless to the MPCC. Pass None for
            the TRUE unclamped constraint distance (the task-width signal
            consumed by the gaze budget / speed model), which must never be
            clamped: a 10 m corridor is perceptually free space, not a 0.2 m
            tunnel.

    Returns:
        ``(left_bound_func, right_bound_func)`` or *None* when no
        PathConstraint regions with a width are found.
    """
    if constraints is None:
        return None

    margin = getattr(constraints, 'default_margin', default_margin)

    # Define sampling grid up front — needed for per-point width interpolation
    num_samples = max(50, int(reference_path.total_length / 0.01))
    s_samples = np.linspace(0, reference_path.total_length, num_samples)

    half_widths: List[float] = []
    for region in constraints.regions:
        geom = region.geometry
        if isinstance(geom, PathConstraint) and geom.width is not None:
            region_margin = getattr(region, 'margin', None)
            if region_margin is None:
                region_margin = margin
            w = geom.width
            if isinstance(w, list):
                w_arr = np.array(w, dtype=float)
                fracs = np.linspace(0, 1, len(w_arr))
                s_fracs = s_samples / reference_path.total_length
                w_interp = np.interp(s_fracs, fracs, w_arr)
                hw = np.maximum(0.0, w_interp / 2.0 - region_margin)
                half_widths.append(hw)
            else:
                half_widths.append(max(0.0, float(w) / 2.0 - region_margin))

    if not half_widths:
        return None

    # Normalize all entries to arrays
    hw_arrays = []
    for hw in half_widths:
        if isinstance(hw, np.ndarray):
            hw_arrays.append(hw)
        else:
            hw_arrays.append(np.full(num_samples, hw))

    bound_array = np.min(np.stack(hw_arrays, axis=0), axis=0)

    left_bounds = np.clip(bound_array, 0.0, max_bound)
    right_bounds = np.clip(bound_array, 0.0, max_bound)

    def left_bound_func(s):
        idx = np.clip(int(s / reference_path.total_length * (num_samples - 1)),
                       0, num_samples - 1)
        return float(left_bounds[idx])

    def right_bound_func(s):
        idx = np.clip(int(s / reference_path.total_length * (num_samples - 1)),
                       0, num_samples - 1)
        return float(right_bounds[idx])

    return (left_bound_func, right_bound_func)


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(vertices)
    inside = False
    px, py = float(point[0]), float(point[1])
    j = n - 1
    for i in range(n):
        xi, yi = float(vertices[i][0]), float(vertices[i][1])
        xj, yj = float(vertices[j][0]), float(vertices[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _distance_to_polygon_boundary(point: np.ndarray, vertices: np.ndarray) -> float:
    """Minimum distance from a point to any edge of the polygon."""
    min_dist = float('inf')
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        a = vertices[i]
        b = vertices[j]
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-15:
            dist = np.linalg.norm(point - a)
        else:
            t = np.clip(np.dot(point - a, ab) / ab_sq, 0.0, 1.0)
            closest = a + t * ab
            dist = np.linalg.norm(point - closest)
        min_dist = min(min_dist, dist)
    return min_dist
