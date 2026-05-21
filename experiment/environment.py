"""
Environment factory and task config generator for hcs_package experiments.

This module provides a two-stage configuration system:
1. Environment config (JSON) defines task parameters
2. Task config is generated from the environment for hcs_package
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from experiment.utils import (
    generateTunnelPath,
    generateCornerPath,
    generateTunnelBoundaries,
    parse_grid_layout,
    compute_window_size_for_lasso,
    compute_start_position_for_lasso,
    compute_end_position_for_lasso,
    generate_lasso_boundary_waypoints,
    Icon,
)


# Standard screen size assumption: 0.46m width maps to screen_width pixels
SCREEN_WIDTH_METERS = 0.46


def _interpolate_path(waypoints: List[Tuple[float, float]], step_size: float) -> List[Tuple[float, float]]:
    """
    Interpolate points along a path to create a dense centerline.
    
    Args:
        waypoints: Sparse waypoints (corner points)
        step_size: Distance between interpolated points
        
    Returns:
        Dense list of points along the path
    """
    if not waypoints or len(waypoints) < 2:
        return list(waypoints) if waypoints else []
    
    dense_path = []
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        segment_length = np.sqrt(dx**2 + dy**2)
        
        if segment_length < 1e-10:
            dense_path.append(p1)
            continue
        
        # Number of steps for this segment
        num_steps = max(1, int(np.ceil(segment_length / step_size)))
        
        # Interpolate points
        for j in range(num_steps):
            t = j / num_steps
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            dense_path.append((x, y))
    
    # Add the final point
    dense_path.append(waypoints[-1])
    
    return dense_path


def load_environment_config(config_path: str) -> Dict:
    """Load environment configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def create_environment(env_config: Dict) -> Dict:
    """
    Create an environment from configuration.
    
    Dispatches to appropriate generator based on env_type.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Environment dictionary with generated paths, boundaries, etc.
    """
    env_type = env_config.get("env_type", "")
    
    if env_type in ("tunnel_steering_smooth", "tunnel_steering_wave", "sigmoidal"):
        return _create_tunnel_env(env_config)
    elif env_type in ("tunnel_steering_corner", "corner", "turning"):
        return _create_corner_env(env_config)
    elif env_type in ("lasso_selection", "lasso"):
        return _create_lasso_env(env_config)
    elif env_type in ("cascading_menu", "menu"):
        return _create_menu_env(env_config)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def _create_tunnel_env(env_config: Dict) -> Dict:
    """Create a sigmoidal/wave tunnel environment."""
    screen_width = env_config.get("screen_width", 460)
    screen_height = env_config.get("screen_height", 260)
    curvature = env_config.get("curvature", 0.025)
    tunnel_width = env_config.get("tunnelWidth", env_config.get("tunnel_width", 0.02))
    start_x = env_config.get("start_x", 0.0)
    end_x = env_config.get("end_x", 0.46)
    y_base = env_config.get("y_base", 0.13)
    wavelength = env_config.get("wavelength", 0.23)
    step_size = env_config.get("step_size", 0.002)
    target_radius = env_config.get("target_radius", tunnel_width * 0.5)
    max_steps = env_config.get("max_steps", 500)

    # Generate centerline path
    centerline, _ = generateTunnelPath(
        width=tunnel_width,
        curvature=curvature,
        start_x=start_x,
        end_x=end_x,
        y_base=y_base,
        wavelength=wavelength,
        step_size=step_size
    )

    # Gets segment 1 & 2 widths from environment for narrowing tunnels, otherwise sets default to None
    segment1_width = env_config.get("segment1Width", None)
    segment2_width = env_config.get("segment2Width", None)

    if segment1_width is not None and segment2_width is not None:
        n = len(centerline)
        mid = n // 2
        width_profile = [segment1_width] * mid + [segment2_width] * (n - mid)
    else:
        width_profile = None
    
    # Generate boundaries for rendering
    left_boundary, right_boundary = generateTunnelBoundaries(centerline, width_profile)
    
    # Target is at end of centerline
    target_pos = centerline[-1] if centerline else (end_x, y_base)
    
    # Compute window dimensions in meters
    window_width_m = SCREEN_WIDTH_METERS
    window_height_m = window_width_m * screen_height / screen_width
    
    return {
        "env_type": env_config.get("env_type", "tunnel_steering_smooth"),
        "screen_width": screen_width,
        "screen_height": screen_height,
        "window_width_m": window_width_m,
        "window_height_m": window_height_m,
        "centerline": centerline,
        "left_boundary": left_boundary.tolist() if hasattr(left_boundary, 'tolist') else list(left_boundary),
        "right_boundary": right_boundary.tolist() if hasattr(right_boundary, 'tolist') else list(right_boundary),
        "tunnel_width": tunnel_width,
        "width_profile": width_profile, 
        "target_pos": target_pos,
        "target_radius": target_radius,
        "max_steps": max_steps,
        "description": env_config.get("description", ""),
    }


def _create_corner_env(env_config: Dict) -> Dict:
    """Create a corner/turning tunnel environment."""
    screen_width = env_config.get("screen_width", 460)
    screen_height = env_config.get("screen_height", 260)
    tunnel_width = env_config.get("tunnelWidth", env_config.get("tunnel_width", 0.02))
    start_x = env_config.get("start_x", 0.0)
    end_x = env_config.get("end_x", 0.46)
    y_base = env_config.get("y_base", 0.13)
    num_corners = env_config.get("num_corners", 3)
    corner_offset = env_config.get("corner_offset", 0.05)
    step_size = env_config.get("step_size", 0.002)
    target_radius = env_config.get("target_radius", tunnel_width * 0.5)
    max_steps = env_config.get("max_steps", 600)

    # Generate centerline path
    centerline, _ = generateCornerPath(
        width=tunnel_width,
        start_x=start_x,
        end_x=end_x,
        y_base=y_base,
        num_corners=num_corners,
        corner_offset=corner_offset,
        step_size=step_size
    )
    
    # Generate boundaries for rendering
    left_boundary, right_boundary = generateTunnelBoundaries(centerline, tunnel_width)
    
    # Target is at end of centerline
    target_pos = centerline[-1] if centerline else (end_x, y_base)
    
    # Compute window dimensions in meters
    window_width_m = SCREEN_WIDTH_METERS
    window_height_m = window_width_m * screen_height / screen_width
    
    return {
        "env_type": env_config.get("env_type", "tunnel_steering_corner"),
        "screen_width": screen_width,
        "screen_height": screen_height,
        "window_width_m": window_width_m,
        "window_height_m": window_height_m,
        "centerline": centerline,
        "left_boundary": left_boundary.tolist() if hasattr(left_boundary, 'tolist') else list(left_boundary),
        "right_boundary": right_boundary.tolist() if hasattr(right_boundary, 'tolist') else list(right_boundary),
        "tunnel_width": tunnel_width,
        "target_pos": target_pos,
        "target_radius": target_radius,
        "max_steps": max_steps,
        "description": env_config.get("description", ""),
    }


def _create_lasso_env(env_config: Dict) -> Dict:
    """Create a lasso selection environment (matching steering_simulator_ddm approach)."""
    screen_width = env_config.get("screen_width", 460)
    screen_height = env_config.get("screen_height", 300)
    grid_layout = env_config.get("grid_layout", ["O O O O O", "O X X X O", "O X X O O", "O O O O O"])
    grid_origin = tuple(env_config.get("grid_origin", [0.1, 0.05]))
    icon_spacing = env_config.get("icon_spacing", 0.035)
    icon_radius = env_config.get("icon_radius", 0.015)
    target_radius = env_config.get("target_radius", 0.004)
    max_steps = env_config.get("max_steps", 300)

    # Parse grid layout into icons
    icons = parse_grid_layout(grid_layout, grid_origin, icon_spacing, icon_radius)

    # Compute window size based on grid
    window_width_m, window_height_m = compute_window_size_for_lasso(
        grid_layout, grid_origin, icon_spacing, target_radius
    )

    # Margin size = gap between icons (same as steering_simulator_ddm)
    margin_size = icon_spacing - 2 * icon_radius
    # Effective corridor width = margin_size * 2 (same as tunnel_width in steering_simulator_ddm)
    corridor_width = margin_size * 2

    # Compute start and end positions (pass margin_size, not corridor_width)
    start_pos = compute_start_position_for_lasso(icons, grid_origin, icon_spacing, icon_radius, margin_size)
    end_pos = compute_end_position_for_lasso(icons, grid_origin, icon_spacing)

    # Generate boundary waypoints (lasso loop path)
    boundary_waypoints = generate_lasso_boundary_waypoints(icons, grid_origin, icon_spacing)

    if not boundary_waypoints:
        # Fallback: simple path
        boundary_waypoints = [start_pos, (0.2, 0.1), (0.2, 0.2), (0.1, 0.2), start_pos]

    # The boundary waypoints form a closed loop [c1, ..., cN, c1].
    # Remove the closing duplicate so the path becomes an open corridor
    # from start_pos to end_pos (tunnel, not ring).
    if len(boundary_waypoints) >= 2 and boundary_waypoints[-1] == boundary_waypoints[0]:
        boundary_waypoints.pop()

    if start_pos:
        boundary_waypoints.insert(0, start_pos)
    if end_pos and len(boundary_waypoints) > 0:
        boundary_waypoints.append(end_pos)

    # Interpolate points along the path to create a dense centerline
    # (consistent with corner_tunnel task which uses step_size for interpolation)
    step_size = env_config.get("step_size", 0.002)
    centerline = _interpolate_path(boundary_waypoints, step_size)

    # Convert Icon dataclass objects to dicts for JSON serialization
    icons_dict = [asdict(icon) for icon in icons]

    return {
        "env_type": env_config.get("env_type", "lasso_selection"),
        "screen_width": screen_width,
        "screen_height": screen_height,
        "window_width_m": window_width_m,
        "window_height_m": window_height_m,
        "centerline": centerline,
        "corridor_width": corridor_width,
        "icons": icons_dict,
        "grid_layout": grid_layout,
        "grid_origin": list(grid_origin),
        "icon_spacing": icon_spacing,
        "icon_radius": icon_radius,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "target_pos": end_pos,
        "target_radius": target_radius,
        "max_steps": max_steps,
        "description": env_config.get("description", ""),
    }


def _create_menu_env(env_config: Dict) -> Dict:
    """Create a cascading menu environment with detailed menu item structures."""
    screen_width = env_config.get("screen_width", 460)
    screen_height = env_config.get("screen_height", 300)
    main_menu_size = env_config.get("main_menu_size", 12)
    sub_menu_size = env_config.get("sub_menu_size", 12)
    target_main_idx = env_config.get("target_main_menu_index", 11)
    target_sub_idx = env_config.get("target_sub_menu_index", 11)
    main_menu_window_size = env_config.get("main_menu_window_size", [0.12, 0.12])
    sub_menu_window_size = env_config.get("sub_menu_window_size", [0.1, 0.12])
    main_menu_origin = env_config.get("main_menu_origin", [0.1, 0.02])
    target_radius = env_config.get("target_radius", 0.005)
    max_steps = env_config.get("max_steps", 500)

    main_menu_x, main_menu_y = main_menu_origin
    main_menu_w, main_menu_h = main_menu_window_size
    sub_menu_w, sub_menu_h = sub_menu_window_size

    main_item_height = main_menu_h / main_menu_size
    sub_item_height = sub_menu_h / sub_menu_size

    # Main menu rectangle (same structure as original)
    main_menu_rect = {
        'x': main_menu_x,
        'y': main_menu_y,
        'width': main_menu_w,
        'height': main_menu_h
    }

    # Main menu items (same structure as original)
    main_menu_items = []
    for i in range(main_menu_size):
        item_y = main_menu_y + i * main_item_height
        main_menu_items.append({
            'index': i,
            'x': main_menu_x,
            'y': item_y,
            'width': main_menu_w,
            'height': main_item_height,
            'is_target': (i == target_main_idx)
        })

    # Submenu position (aligned with target main menu item top)
    target_main_item_top = main_menu_y + target_main_idx * main_item_height
    sub_menu_x = main_menu_x + main_menu_w
    sub_menu_y = target_main_item_top

    sub_menu_rect = {
        'x': sub_menu_x,
        'y': sub_menu_y,
        'width': sub_menu_w,
        'height': sub_menu_h
    }

    # Submenu items
    sub_menu_items = []
    for i in range(sub_menu_size):
        item_y = sub_menu_y + i * sub_item_height
        sub_menu_items.append({
            'index': i,
            'x': sub_menu_x,
            'y': item_y,
            'width': sub_menu_w,
            'height': sub_item_height,
            'is_target': (i == target_sub_idx)
        })

    # Start position (center of first main menu item - same as original)
    start_x = main_menu_x + main_menu_w / 2
    start_y = main_menu_y + main_item_height / 2
    start_pos = (start_x, start_y)

    # Target position (center of target submenu item)
    target_sub_x = sub_menu_x + sub_menu_w / 2
    target_sub_y = sub_menu_y + (target_sub_idx + 0.5) * sub_item_height
    target_pos = (target_sub_x, target_sub_y)

    # Keypoints for reference path generation (matching steering_simulator_ddm approach):
    # 1. Start: center of first main menu item
    # 2. Decision: biased toward right side of target main menu item
    # 3. Transition: right edge of main menu, aligned with target item
    # 4. Target: center of target submenu item
    target_main_x = main_menu_x + main_menu_w / 2
    target_main_y = main_menu_y + (target_main_idx + 0.5) * main_item_height

    decision_x_offset = main_menu_w * 0.1
    decision_x = min(target_main_x + decision_x_offset, main_menu_x + main_menu_w - 0.001)
    decision_point = (decision_x, target_main_y)

    transition_point = (main_menu_x + main_menu_w, target_main_y)

    # Build a smooth centerline through the L-shaped menu.
    # The path goes: start → decision point → diagonal through junction → target.
    # Using a diagonal avoids the sharp 90-degree turn that causes
    # cubic spline oscillation and extreme curvature spikes.
    sub_entry_x = sub_menu_x + sub_menu_w * 0.5
    sub_entry_y = sub_menu_y + sub_item_height * 0.5

    keypoints = [
        start_pos,
        decision_point,
        (sub_entry_x, sub_entry_y),
        target_pos,
    ]

    step_size = env_config.get("step_size", 0.002)
    centerline = _interpolate_path(keypoints, step_size)

    # Compute window dimensions
    window_width_m = max(0.46, sub_menu_x + sub_menu_w + 0.05)
    window_height_m = max(0.26, main_menu_y + main_menu_h + 0.05)

    return {
        "env_type": env_config.get("env_type", "cascading_menu"),
        "screen_width": screen_width,
        "screen_height": screen_height,
        "window_width_m": window_width_m,
        "window_height_m": window_height_m,
        "centerline": centerline,
        "main_menu_size": main_menu_size,
        "sub_menu_size": sub_menu_size,
        "target_main_menu_index": target_main_idx,
        "target_sub_menu_index": target_sub_idx,
        "main_menu_window_size": main_menu_window_size,
        "sub_menu_window_size": sub_menu_window_size,
        "main_menu_origin": main_menu_origin,
        "sub_menu_origin": (sub_menu_x, sub_menu_y),
        "main_menu_rect": main_menu_rect,
        "sub_menu_rect": sub_menu_rect,
        "main_menu_items": main_menu_items,
        "sub_menu_items": sub_menu_items,
        "start_pos": start_pos,
        "target_pos": target_pos,
        "target_radius": target_radius,
        "max_steps": max_steps,
        "description": env_config.get("description", ""),
    }


def generate_task_config(
    environment: Dict,
    waypoint_density: float = None,
    include_constraints: bool = True
) -> Dict:
    """
    Generate hcs_package-compatible task config from environment.
    
    Args:
        environment: Environment dictionary from create_environment()
        waypoint_density: Spacing between waypoints in meters. If None, defaults to
                         0.5 * target_radius to ensure reliable waypoint detection.
        include_constraints: Whether to include PathConstraint
        
    Returns:
        Task config dictionary for hcs_package CursorSimulator
    """
    env_type = environment.get("env_type", "")
    screen_width = environment["screen_width"]
    screen_height = environment["screen_height"]
    window_width_m = environment.get("window_width_m", SCREEN_WIDTH_METERS)
    window_height_m = environment.get("window_height_m", window_width_m * screen_height / screen_width)
    centerline = environment.get("centerline", [])
    target_radius = environment.get("target_radius", 0.01)
    max_steps = environment.get("max_steps", 500)
    
    # Waypoint density should be less than target_radius so the cursor reliably
    # reaches each waypoint before moving to the next
    if waypoint_density is None:
        waypoint_density = target_radius * 0.5
    
    # Sample waypoints from centerline
    waypoints_m = _sample_waypoints(centerline, waypoint_density)
    
    # Convert waypoints from meters to pixels
    waypoints_px = _meters_to_pixels(waypoints_m, screen_width, screen_height, window_width_m, window_height_m)
    
    task_config = {
        "waypoints": waypoints_px,
        "screen_width": screen_width,
        "screen_height": screen_height,
        "target_radius": target_radius,
        "max_steps": max_steps,
    }
    
    # Add constraints based on environment type
    if include_constraints:
        constraints = _generate_constraints(environment)
        if constraints:
            task_config["constraints"] = constraints
    
    return task_config


def _sample_waypoints(
    centerline: List[Tuple[float, float]],
    density: float
) -> List[Tuple[float, float]]:
    """
    Sample waypoints from centerline at specified density.
    
    Args:
        centerline: Dense path in meters
        density: Spacing between waypoints
        
    Returns:
        Sampled waypoints
    """
    if not centerline:
        return []
    
    if len(centerline) <= 2:
        return list(centerline)
    
    # Compute cumulative arc length
    path = np.array(centerline)
    diffs = np.diff(path, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < density:
        return [centerline[0], centerline[-1]]
    
    # Sample at regular intervals
    num_samples = max(2, int(total_length / density) + 1)
    sample_distances = np.linspace(0, total_length, num_samples)
    
    waypoints = []
    for d in sample_distances:
        idx = np.searchsorted(cumulative_length, d, side='right') - 1
        idx = max(0, min(idx, len(path) - 2))
        
        seg_start = cumulative_length[idx]
        seg_length = segment_lengths[idx] if idx < len(segment_lengths) else 1e-6
        
        if seg_length < 1e-9:
            t = 0
        else:
            t = (d - seg_start) / seg_length
        t = max(0, min(1, t))
        
        pt = path[idx] + t * (path[idx + 1] - path[idx])
        waypoints.append((float(pt[0]), float(pt[1])))
    
    # Ensure last point is exactly the end
    waypoints[-1] = (float(path[-1][0]), float(path[-1][1]))
    
    return waypoints


def _meters_to_pixels(
    waypoints_m: List[Tuple[float, float]],
    screen_width: int,
    screen_height: int,
    window_width_m: float,
    window_height_m: float
) -> List[List[int]]:
    """Convert waypoints from meters to pixels."""
    scale_x = screen_width / window_width_m
    scale_y = screen_height / window_height_m
    
    return [[int(x * scale_x), int(y * scale_y)] for x, y in waypoints_m]


def _generate_constraints(environment: Dict) -> Optional[Dict]:
    """Generate constraints for hcs_package based on environment type."""
    env_type = environment.get("env_type", "")
    
    if "tunnel" in env_type or "corner" in env_type or "turning" in env_type:
        return _generate_tunnel_constraints(environment)
    elif "lasso" in env_type:
        return _generate_lasso_constraints(environment)
    elif "menu" in env_type:
        return _generate_menu_constraints(environment)
    
    return None


def _generate_tunnel_constraints(environment: Dict) -> Dict:
    """Generate PathConstraint for tunnel environments."""
    centerline = environment.get("centerline", [])
    tunnel_width = environment.get("tunnel_width", 0.02)
    width_profile = environment.get("width_profile", None)

    path_coords = [[float(x), float(y)] for x, y in centerline]

    # hcs_package constraint geometry only accepts a scalar width.
    # For wide-to-narrow tunnels, use the narrower segment width so the
    # hard constraint reflects the most restrictive part of the tunnel.
    if width_profile is not None:
        width_value = width_profile 
    else:
        width_value = tunnel_width

    return {
        "coordinate_system": "normalized",
        "default_margin": 0.0,
        "regions": [{
            "constraint_type": "keep_in",
            "geometry": {
                "type": "path",
                "path": path_coords,
                "width": width_value
            },
            "enabled": True
        }]
    }


def _generate_lasso_constraints(environment: Dict) -> Dict:
    """Generate PathConstraint for lasso environments (corridor along lasso path)."""
    centerline = environment.get("centerline", [])
    corridor_width = environment.get("corridor_width", 0.005)

    if not centerline:
        return None

    # Convert centerline to list of lists for JSON
    path_coords = [[float(x), float(y)] for x, y in centerline]

    return {
        "coordinate_system": "normalized",
        "default_margin": 0.0,
        "regions": [{
            "constraint_type": "keep_in",
            "geometry": {
                "type": "path",
                "path": path_coords,
                "width": corridor_width
            },
            "enabled": True
        }]
    }


def _generate_menu_constraints(environment: Dict) -> Dict:
    """Generate a keep-in polygon for the union of main + sub menu.

    Handles three cases for the vertical relationship between menus:
      1. Sub menu top == main menu bottom (L-shape, no overlap)
      2. Sub menu top < main menu bottom (overlap — produces a notched shape)
      3. Sub menu top > main menu bottom (gap — still produces valid polygon)

    A small margin is added around the polygon so the constraint boundary
    doesn't coincide exactly with the menu edges. This prevents the speed
    planner from over-decelerating near menu item targets that sit close
    to the boundary.
    """
    MARGIN = 0.005  # 5mm padding around the menu polygon

    mx, my = environment.get("main_menu_origin", [0.1, 0.02])
    mw, mh = environment.get("main_menu_window_size", [0.12, 0.12])
    sx, sy = environment.get("sub_menu_origin", (mx + mw, my))
    sw, sh = environment.get("sub_menu_window_size", [0.1, 0.12])

    # Apply margin to outer edges so the constraint boundary doesn't
    # coincide exactly with menu item edges.  This prevents the speed
    # planner from over-decelerating near targets that sit close to
    # the boundary.
    left_m  = mx - MARGIN
    right_m = mx + mw              # junction x (shared edge, no margin)
    top_m   = my - MARGIN
    bot_m   = my + mh + MARGIN
    top_s   = sy - MARGIN
    bot_s   = sy + sh + MARGIN
    right_s = sx + sw + MARGIN

    # Trace the outer boundary of the union clockwise, starting top-left.
    vertices = []

    vertices.append([left_m, top_m])

    if top_s >= bot_m:
        # Case 3: gap or exact L-shape
        vertices.append([right_m, top_m])
        vertices.append([right_m, top_s])
        vertices.append([right_s, top_s])
        vertices.append([right_s, bot_s])
        vertices.append([right_m, bot_s])
        vertices.append([right_m, bot_m])
        vertices.append([left_m, bot_m])
    elif top_s <= top_m:
        # Case: sub starts at or above main top
        vertices.append([right_m, top_m])
        vertices.append([right_m, top_s])
        vertices.append([right_s, top_s])
        vertices.append([right_s, bot_s])
        vertices.append([right_m, bot_s])
        if bot_s < bot_m:
            vertices.append([right_m, bot_m])
        vertices.append([left_m, bot_m])
    else:
        # Case 2: overlap — sub top is between main top and main bottom
        vertices.append([right_m, top_m])
        vertices.append([right_m, top_s])
        vertices.append([right_s, top_s])
        vertices.append([right_s, bot_s])
        vertices.append([right_m, bot_s])
        # Always trace back through the main menu bottom edge
        if bot_s != bot_m:
            vertices.append([right_m, bot_m])
        vertices.append([left_m, bot_m])

    regions = [
        {
            "constraint_type": "keep_in",
            "geometry": {
                "type": "polygon",
                "vertices": vertices
            },
            "enabled": True
        }
    ]

    return {
        "coordinate_system": "normalized",
        "default_margin": 0.0,
        "regions": regions
    }


def save_task_config(task_config: Dict, path: str) -> str:
    """Save task config to JSON file."""
    with open(path, 'w') as f:
        json.dump(task_config, f, indent=2)
    return path
