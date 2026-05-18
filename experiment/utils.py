from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from scipy.interpolate import interp1d, splprep, splev
import os


def draw_speed_profile(speeds, save_path="speed_profile.png", title="Speed Profile"):
    """Draws a speed profile plot from a list of speeds.

    Args:
        speeds (_type_): List or array of speed values (e.g., cursor speeds).
        save_path (str, optional): _description_. Defaults to "speed_profile.png".
        title (str, optional): _description_. Defaults to "Speed Profile".
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(speeds, label="Speed", linewidth=1, color='black')
    ax.scatter(range(len(speeds)), speeds, color='black', s=10, label="Time Steps", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    

def draw_trajectory(cursor_x, cursor_y, target_pos, radius,
                    window_width, window_height,
                    tunnel_path=None, tunnel_width=None, pause_coordinates=None,
                    save_path="trajectory.png", title="Cursor Trajectory",
                    planned_segments=None, planned_targets=None, show_planned_targets=False,
                    show_centerline=False,
                    correction_segments=None, correction_triggers=None,
                    reference_path=None, reference_speed_func=None,
                    tunnel_wall_left=True, tunnel_wall_right=True):
    """
    Draws the cursor trajectory, target, and tunnel boundaries.

    Args:
        cursor_x (list or np.ndarray): Cursor X trajectory.
        cursor_y (list or np.ndarray): Cursor Y trajectory.
        target_pos (tuple): (x, y) position of the target.
        radius (float): Target radius (same units as positions).
        window_width (float): Width of the environment (m).
        window_height (float): Height of the environment (m).
        tunnel_path (list of (x, y)): Center points of the tunnel path.
        tunnel_width (float): Width of the tunnel (same units as positions).
        save_path (str): Path to save image file.
        title (str): Title of the plot.
        planned_segments (list of (xs, ys)): Optional list of planned path segments.
        planned_targets (list of (x, y)): Optional list of target waypoints per plan.
        show_planned_targets (bool): If True, plot the planned target waypoints.
    """
    cursor_x = np.array(cursor_x)
    cursor_y = np.array(cursor_y)
    target_x, target_y = target_pos

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Plot cursor trajectory
    ax.plot(cursor_x, cursor_y, label="Cursor Trajectory", linewidth=0.5, color='black')
    ax.scatter(cursor_x[0], cursor_y[0], color='green', label="Start", zorder=5)
    ax.scatter(cursor_x[-1], cursor_y[-1], color='blue', label="End", zorder=5)

    # Planned segments overlay
    if planned_segments:
        for seg_x, seg_y in planned_segments:
            seg_x = np.asarray(seg_x)
            seg_y = np.asarray(seg_y)
            ax.plot(seg_x, seg_y, color='tab:orange', linewidth=1.0, alpha=0.5, label="Plan" if 'Plan' not in [h.get_label() for h in ax.lines] else "",)
            # start dot
            ax.scatter(seg_x[0], seg_y[0], color='tab:orange', s=15, zorder=6, alpha=0.5)
            # end cross
            ax.scatter(seg_x[-1], seg_y[-1], marker='x', color='tab:orange', s=20, zorder=6, alpha=0.5)

    # Planned targets overlay (debug)
    if show_planned_targets and planned_targets:
        pts = np.asarray(planned_targets, dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], marker='x', color='tab:blue', s=10, label='Planned Target', zorder=7,  alpha=0.8)

    # Correction plans overlay (post-run visualization)
    if correction_segments:
        existing_labels = [h.get_label() for h in ax.lines]
        for seg_x, seg_y in correction_segments:
            seg_x = np.asarray(seg_x)
            seg_y = np.asarray(seg_y)
            ax.plot(
                seg_x, seg_y,
                color=(0/255, 200/255, 0/255), linewidth=1.2, alpha=0.6,
                label="Correction Plan" if "Correction Plan" not in existing_labels else ""
            )
            existing_labels = [h.get_label() for h in ax.lines]
    # Correction triggers as green dots
    if correction_triggers:
        trigs = np.asarray(correction_triggers, dtype=float)
        if trigs.ndim == 2 and trigs.shape[1] == 2:
            ax.scatter(trigs[:, 0], trigs[:, 1], color=(0/255, 200/255, 0/255), s=15, zorder=8, label='Correction Trigger', alpha=0.5)

    # Draw target
    target_circle = plt.Circle((target_x, target_y), radius, color='red', alpha=0.5, label="Target")
    ax.add_patch(target_circle)
    ax.scatter([target_x], [target_y], color='red', edgecolor='black', zorder=5)

    # Draw tunnel as polygon built from per-point normals (works for turns)
    if tunnel_path is not None and tunnel_width is not None:
        path = np.asarray(tunnel_path, dtype=float)
        if path.ndim == 2 and path.shape[0] >= 2 and path.shape[1] == 2:
            # Generate boundaries only if at least one wall is visible
            if tunnel_wall_left or tunnel_wall_right:
                left_boundary, right_boundary = generateTunnelBoundaries(tunnel_path, tunnel_width)
            else:
                # No walls needed
                left_boundary = None
                right_boundary = None

            # Fill polygon: left forward, right reversed (only if both walls exist)
            if left_boundary is not None and right_boundary is not None:
                poly_x = np.concatenate([left_boundary[:, 0], right_boundary[::-1, 0]])
                poly_y = np.concatenate([left_boundary[:, 1], right_boundary[::-1, 1]])
                ax.fill(poly_x, poly_y, color='blue', alpha=0.15, label="Constraint Corridor")

            # Boundary lines (only draw walls that exist)
            if left_boundary is not None and tunnel_wall_left:
                ax.plot(left_boundary[:, 0], left_boundary[:, 1], color='blue', linestyle='--', linewidth=0.8, alpha=0.4, label="Corridor Boundary")
            if right_boundary is not None and tunnel_wall_right:
                # Only add label if left wall wasn't drawn (to avoid duplicate labels)
                label = "Corridor Boundary" if (left_boundary is None or not tunnel_wall_left) else ""
                ax.plot(right_boundary[:, 0], right_boundary[:, 1], color='blue', linestyle='--', linewidth=0.8, alpha=0.4, label=label)

            # Optional centerline
            if show_centerline:
                ax.plot(path[:, 0], path[:, 1], color='black', linewidth=0.6, alpha=0.6, label="Centerline")

    # Draw optimal reference path
    # reference_path can be either:
    # - A reference path object (has total_length attribute and is callable)
    # - A list of pre-converted (x, y) points in meters
    if reference_path is not None:
        try:
            if isinstance(reference_path, (list, tuple)) and len(reference_path) > 0:
                # Pre-converted list of points
                ref_xs = [p[0] for p in reference_path]
                ref_ys = [p[1] for p in reference_path]
            elif hasattr(reference_path, 'total_length'):
                # Reference path object - sample points along it
                theta_samples = np.linspace(0, reference_path.total_length, 200)
                ref_xs, ref_ys = [], []
                for theta in theta_samples:
                    try:
                        ref_pos = reference_path(theta)
                        ref_xs.append(ref_pos[0])
                        ref_ys.append(ref_pos[1])
                    except:
                        continue
            else:
                ref_xs, ref_ys = [], []
            
            if ref_xs and ref_ys:
                # Plot reference path as a smooth line
                ax.plot(ref_xs, ref_ys, color='gray', linewidth=0.5, alpha=0.4, 
                       label="Optimal Reference Path", linestyle='-')
                
                # Add speed visualization as colored dots
                # if reference_speed_func is not None:
                #     for i, theta in enumerate(theta_samples[::10]):  # Sample every 10th point
                #         try:
                #             ref_pos = reference_path(theta)
                #             v_ref = reference_speed_func(theta)
                #             # Color intensity based on speed (darker = faster)
                #             color_intensity = min(1.0, v_ref / 0.3)  # Normalize to max expected speed
                #             ax.scatter(ref_pos[0], ref_pos[1], 
                #                      color='gray', 
                #                      s=8, alpha=0.6, zorder=4)
                #         except:
                #             continue
        except Exception as e:
            print(f"Warning: Could not plot reference path: {e}")

    # --- Pause markers ---
    if pause_coordinates is not None:
        # Draw hollow circles around pause points
        # Auto radius if not supplied
        pause_radius = 0.008 * max(window_width, window_height)
        pause_edgecolor = "orange"
        pause_linewidth=1.0
        for x, y in pause_coordinates:
            circ = Circle((x, y), pause_radius, fill=False,
                          edgecolor=pause_edgecolor, linewidth=pause_linewidth, zorder=7)
            ax.add_patch(circ)
    # Set environment limits
    ax.set_xlim(0, window_width)
    ax.set_ylim(0, window_height)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    ax.set_title(title)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    
    # Place legend outside plot area to avoid overlapping trajectory
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' includes legend
    plt.close()
    print(f"Trajectory saved to {save_path}")
    
def generate_constraints_from_task_config(task_config: Dict, tunnel_path=None):
    """Generate ConstraintConfig from task_config.
    
    This function creates constraints based on task configuration:
    - For tunnel tasks: Creates PathConstraint from tunnel centerline
    - For cascading menu tasks: Creates RectangleConstraint from menu rectangles
    - For other tasks: Parses explicit constraints from task_config

    Requires the ``simulators`` package (used by legacy experiment scripts).
    
    Args:
        task_config: Task configuration dictionary
        tunnel_path: List of (x, y) points for tunnel centerline (for tunnel tasks)
        
    Returns:
        ConstraintConfig object or None if no constraints
    """
    from simulators.constraints import (  # noqa: lazy import – only needed by legacy callers
        ConstraintConfig, ConstraintRegion, ConstraintType,
        PathConstraint, PolygonConstraint,
        CircleConstraint, RectangleConstraint
    )
    regions = []
    task_type = task_config.get("task_type", "tunnel")
    
    # For tunnel tasks, create PathConstraint from tunnel centerline
    if task_type == "tunnel" and tunnel_path is not None:
        tunnel_width = task_config.get("tunnel_width", 0.02)
        
        # Convert centerline to list of tuples for PathConstraint
        centerline_path_tuples = [(float(p[0]), float(p[1])) for p in tunnel_path]
        
        # Create PathConstraint with tunnel centerline and width
        tunnel_constraint = PathConstraint(
            path=centerline_path_tuples,
            width=tunnel_width
        )
        
        # Add as KEEP_IN constraint (stay within the tunnel corridor)
        regions.append(ConstraintRegion(
            constraint_type=ConstraintType.KEEP_IN,
            geometry=tunnel_constraint
        ))
    
    # For cascading menu tasks, create RectangleConstraint from menu rectangles
    elif task_type == "cascading_menu":
        # Get cascading menu parameters (same logic as setup_cascading_menu_task)
        main_menu_size = task_config.get("main_menu_size", 5)
        target_main_menu_index = task_config.get("target_main_menu_index", 2)
        main_menu_window_size = task_config.get("main_menu_window_size", [0.08, 0.15])
        sub_menu_window_size = task_config.get("sub_menu_window_size", [0.08, 0.15])
        main_menu_origin = task_config.get("main_menu_origin", [0.2, 0.15])
        
        main_menu_x, main_menu_y = main_menu_origin
        main_menu_w, main_menu_h = main_menu_window_size
        sub_menu_w, sub_menu_h = sub_menu_window_size
        
        main_item_height = main_menu_h / main_menu_size
        
        # Main menu rectangle
        main_constraint = RectangleConstraint(
            x=main_menu_x,
            y=main_menu_y,
            width=main_menu_w,
            height=main_menu_h
        )
        regions.append(ConstraintRegion(
            constraint_type=ConstraintType.KEEP_IN,
            geometry=main_constraint
        ))
        
        # Submenu rectangle (positioned to the right of main menu)
        target_main_item_top = main_menu_y + target_main_menu_index * main_item_height
        sub_menu_x = main_menu_x + main_menu_w
        sub_menu_y = target_main_item_top
        
        sub_constraint = RectangleConstraint(
            x=sub_menu_x,
            y=sub_menu_y,
            width=sub_menu_w,
            height=sub_menu_h
        )
        regions.append(ConstraintRegion(
            constraint_type=ConstraintType.KEEP_IN,
            geometry=sub_constraint
        ))
    
    # Parse explicit constraints from constraints_dict if provided
    constraints_dict = task_config.get("constraints", None)
    if constraints_dict is not None and "regions" in constraints_dict:
        for region_dict in constraints_dict["regions"]:
            constraint_type = ConstraintType(region_dict["constraint_type"])
            geometry_dict = region_dict["geometry"]
            geometry_type = geometry_dict.get("type")
            
            geometry = None
            if geometry_type == "point":
                geometry = CircleConstraint(
                    center=(geometry_dict["x"], geometry_dict["y"]),
                    radius=geometry_dict.get("radius", 0.001)
                )
            elif geometry_type == "line":
                geometry = PathConstraint(
                    path=[tuple(geometry_dict["start"]), tuple(geometry_dict["end"])],
                    width=geometry_dict.get("width")
                )
            elif geometry_type == "polygon":
                geometry = PolygonConstraint(
                    vertices=[tuple(v) for v in geometry_dict["vertices"]],
                    closed=geometry_dict.get("closed", True)
                )
            elif geometry_type == "circle":
                geometry = CircleConstraint(
                    center=tuple(geometry_dict["center"]),
                    radius=geometry_dict["radius"]
                )
            elif geometry_type == "rectangle":
                geometry = RectangleConstraint(
                    x=geometry_dict["x"],
                    y=geometry_dict["y"],
                    width=geometry_dict["width"],
                    height=geometry_dict["height"]
                )
            
            if geometry is not None:
                regions.append(ConstraintRegion(
                    constraint_type=constraint_type,
                    geometry=geometry
                ))
    
    return ConstraintConfig(regions=regions) if regions else None


def generate_menu_constraints_from_env(env):
    """Generate constraints from menu rectangles for cascading menu tasks.
    
    This should be called after env.reset() when menu rectangles are set up.
    
    Args:
        env: Environment instance with menu rectangles set up
        
    Returns:
        ConstraintConfig object with menu rectangle constraints
    """
    if not hasattr(env, 'task_type') or env.task_type != "cascading_menu":
        return None
    
    from simulators.constraints import (  # noqa: lazy import – only needed by legacy callers
        ConstraintConfig, ConstraintRegion, ConstraintType,
        RectangleConstraint
    )
    
    regions = []
    
    # Create KEEP_IN constraints for main menu and submenu rectangles
    if hasattr(env, 'main_menu_rect') and env.main_menu_rect:
        main_rect = env.main_menu_rect
        main_constraint = RectangleConstraint(
            x=main_rect['x'],
            y=main_rect['y'],
            width=main_rect['width'],
            height=main_rect['height']
        )
        regions.append(ConstraintRegion(
            constraint_type=ConstraintType.KEEP_IN,
            geometry=main_constraint
        ))
    
    if hasattr(env, 'sub_menu_rect') and env.sub_menu_rect:
        sub_rect = env.sub_menu_rect
        sub_constraint = RectangleConstraint(
            x=sub_rect['x'],
            y=sub_rect['y'],
            width=sub_rect['width'],
            height=sub_rect['height']
        )
        regions.append(ConstraintRegion(
            constraint_type=ConstraintType.KEEP_IN,
            geometry=sub_constraint
        ))
    
    return ConstraintConfig(regions=regions) if regions else None


def _get_screen_size_mac():
    """Return (screen_width, screen_height) on macOS without creating a Tk root.

    Order:
      1) Environment overrides SIM_SCREEN_WIDTH/SIM_SCREEN_HEIGHT
      2) AppKit (PyObjC) NSScreen
      3) Safe fallback 1920x1080
    """
    try:
        env_w = int(os.environ.get("SIM_SCREEN_WIDTH", "0"))
        env_h = int(os.environ.get("SIM_SCREEN_HEIGHT", "0"))
        if env_w > 0 and env_h > 0:
            return env_w, env_h
    except Exception:
        pass

    try:
        from AppKit import NSScreen  # type: ignore
        screen = NSScreen.mainScreen()
        if screen is not None:
            frame = screen.frame()
            return int(frame.size.width), int(frame.size.height)
    except Exception:
        pass

    return 1920, 1080


def position_matplotlib_window(fig, align="center", width=None, height=None):
    """Position a Matplotlib window on the primary display.

    align: one of {"center", "left_center", "right_center", "top_center", "bottom_center"}
    """
    try:
        mgr = plt.get_current_fig_manager()
        # Ensure window exists
        fig.canvas.draw_idle()
        try:
            # TkAgg backend
            win = mgr.window  # Tkinter.Tk
            sw, sh = _get_screen_size_mac()
            if width is None or height is None:
                win.update_idletasks()
                cur_w = win.winfo_width()
                cur_h = win.winfo_height()
                # Fallback if uninitialized
                if cur_w <= 1 or cur_h <= 1:
                    cur_w = 800
                    cur_h = 450
                width = cur_w if width is None else width
                height = cur_h if height is None else height
            # Compute target top-left based on alignment
            if align == "center":
                x = int((sw - int(width)) / 2)
                y = int((sh - int(height)) / 2)
            elif align == "left_center":
                x = int((sw * 0.25) - int(width) / 2)
                y = int((sh - int(height)) / 2)
            elif align == "right_center":
                x = int((sw * 0.75) - int(width) / 2)
                y = int((sh - int(height)) / 2)
            elif align == "top_center":
                x = int((sw - int(width)) / 2)
                y = int((sh * 0.25) - int(height) / 2)
            elif align == "bottom_center":
                x = int((sw - int(width)) / 2)
                y = int((sh * 0.75) - int(height) / 2)
            else:
                x = int((sw - int(width)) / 2)
                y = int((sh - int(height)) / 2)
            win.geometry(f"{int(width)}x{int(height)}+{x}+{y}")
        except Exception:
            # Qt5Agg backend
            try:
                win = mgr.window
                sw, sh = _get_screen_size_mac()
                if width is None or height is None:
                    geo = win.frameGeometry()
                    cur_w = geo.width() if geo else 800
                    cur_h = geo.height() if geo else 450
                    width = cur_w if width is None else width
                    height = cur_h if height is None else height
                if align == "center":
                    x = int((sw - int(width)) / 2)
                    y = int((sh - int(height)) / 2)
                elif align == "left_center":
                    x = int((sw * 0.25) - int(width) / 2)
                    y = int((sh - int(height)) / 2)
                elif align == "right_center":
                    x = int((sw * 0.75) - int(width) / 2)
                    y = int((sh - int(height)) / 2)
                elif align == "top_center":
                    x = int((sw - int(width)) / 2)
                    y = int((sh * 0.25) - int(height) / 2)
                elif align == "bottom_center":
                    x = int((sw - int(width)) / 2)
                    y = int((sh * 0.75) - int(height) / 2)
                else:
                    x = int((sw - int(width)) / 2)
                    y = int((sh - int(height)) / 2)
                win.move(x, y)
                win.resize(int(width), int(height))
            except Exception:
                pass
    except Exception:
        # Silently ignore if backend doesn't expose a window
        pass


def get_centered_pygame_position(window_size):
    sw, sh = _get_screen_size_mac()
    ww, wh = int(window_size[0]), int(window_size[1])
    x = max(0, int((sw - ww) / 2))
    y = max(0, int((sh - wh) / 2))
    return str(x), str(y)


def get_left_center_pygame_position(window_size):
    sw, sh = _get_screen_size_mac()
    ww, wh = int(window_size[0]), int(window_size[1])
    x = max(0, int((sw * 0.25) - ww / 2))
    y = max(0, int((sh - wh) / 2))
    return str(x), str(y)


def get_right_center_pygame_position(window_size):
    sw, sh = _get_screen_size_mac()
    ww, wh = int(window_size[0]), int(window_size[1])
    x = max(0, int((sw * 0.75) - ww / 2))
    y = max(0, int((sh - wh) / 2))
    return str(x), str(y)


def set_pygame_window_pos(x, y):
    """Set the SDL window position for the next pygame window creation."""
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{int(x)},{int(y)}"


def generateTunnelPath(width, curvature, start_x=0.0, end_x=0.46, y_base=0.13,
                        wavelength=0.23, step_size=0.002):
    """Generate a tunnel path using a sine wave curve.
    
    Args:
        width (float): Tunnel width (narrow=0.02 or wide=0.05) relative to screen size.
        curvature (float): Curvature amplitude (gentle=0.025 or sharp=0.05).
        start_x (float): Starting x-coordinate of tunnel. Default 0.0.
        end_x (float): Ending x-coordinate of tunnel. Default 0.46.
        y_base (float): Base y-coordinate for tunnel centerline. Default 0.13.
        wavelength (float): Wavelength for sine wave. Default 0.23.
        step_size (float): Step size along x-axis. Default 0.002.
    
    Returns:
        List of (x, y) tuples representing the tunnel centerline path.
        Tunnel width parameter (float).
    """
    # Generate x coordinates
    xs = np.arange(start_x, end_x, step_size)
    
    # Generate y coordinates using sine wave
    # amplitude = curvature parameter
    amplitude = curvature
    ys = y_base + amplitude * np.sin(2 * np.pi * xs / wavelength)
    
    # Combine into tunnel path
    tunnel_path = list(zip(xs.tolist(), ys.tolist()))
    
    return tunnel_path, width


def generateCornerPath(width, start_x=0.0, end_x=0.46, y_base=0.13, 
                       num_corners=3, corner_offset=0.05, step_size=0.002):
    """Generate a tunnel path with 90-degree corners.
    
    Creates a path that alternates between horizontal and vertical segments,
    forming a path with sharp 90-degree turns. Each segment (horizontal + vertical)
    has equal total length.
    
    Args:
        width (float): Tunnel width (consistent throughout).
        start_x (float): Starting x-coordinate of tunnel. Default 0.0.
        end_x (float): Ending x-coordinate of tunnel. Default 0.46.
        y_base (float): Base y-coordinate for tunnel centerline. Default 0.13.
        num_corners (int): Number of 90-degree corners. Default 2 (L-shaped).
        corner_offset (float): Vertical offset for each corner segment. Default 0.05.
        step_size (float): Step size along path segments. Default 0.002.
    
    Returns:
        List of (x, y) tuples representing the tunnel centerline path.
        Tunnel width parameter (float).
    """
    tunnel_path = []
    
    # Calculate total horizontal distance we need to cover
    total_x_distance = end_x - start_x
    
    # Make all horizontal segments have equal length
    # We have: num_corners horizontal segments (before corners) + 1 final horizontal segment
    # Total: (num_corners + 1) horizontal segments, each with equal length
    horizontal_length_per_segment = total_x_distance / (num_corners + 1)
    
    current_x = start_x
    current_y = y_base
    
    # Generate combined segments (horizontal + vertical)
    for corner_idx in range(num_corners):
        # Horizontal segment before corner - same length for all horizontal segments
        x_end = current_x + horizontal_length_per_segment
        x_coords = np.arange(current_x, x_end + step_size * 0.5, step_size)
        x_coords = x_coords[x_coords <= x_end]
        for x in x_coords:
            tunnel_path.append((x, current_y))
        # Ensure exact endpoint
        if len(tunnel_path) == 0 or abs(tunnel_path[-1][0] - x_end) > 1e-6:
            tunnel_path.append((x_end, current_y))
        else:
            tunnel_path[-1] = (x_end, current_y)
        current_x = x_end
        
        # Vertical segment (corner) - exactly corner_offset length
        direction = 1.0 if corner_idx % 2 == 0 else -1.0
        current_y_start = current_y
        current_y_end = current_y + direction * corner_offset
        
        # Generate vertical segment with exact corner_offset length
        if direction > 0:  # Moving up
            y_coords = np.arange(current_y_start + step_size, current_y_end + step_size * 0.5, step_size)
        else:  # Moving down
            y_coords = np.arange(current_y_start - step_size, current_y_end - step_size * 0.5, -step_size)
        
        # Ensure we include the end point
        if len(y_coords) == 0 or abs(y_coords[-1] - current_y_end) > step_size * 0.5:
            y_coords = np.append(y_coords, current_y_end)
        
        for y in y_coords:
            tunnel_path.append((current_x, y))
        
        current_y = current_y_end
    
    # Final horizontal segment - same length as all other horizontal segments
    final_x_end = current_x + horizontal_length_per_segment
    # But we must end at end_x, so clip if necessary
    final_x_end = min(final_x_end, end_x)
    x_coords = np.arange(current_x + step_size, final_x_end + step_size * 0.5, step_size)
    x_coords = x_coords[x_coords <= final_x_end]
    for x in x_coords:
        tunnel_path.append((x, current_y))
    
    # Ensure we end exactly at end_x
    if len(tunnel_path) == 0 or abs(tunnel_path[-1][0] - end_x) > 1e-6:
        tunnel_path.append((end_x, current_y))
    else:
        tunnel_path[-1] = (end_x, current_y)
    
    return tunnel_path, width


def _seg_intersect_2d(p0, p1, p2, p3):
    """Intersection point of segments p0–p1 and p2–p3, or None (interior only)."""
    d1 = p1 - p0
    d2 = p3 - p2
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-14:
        return None
    dp = p2 - p0
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    u = (dp[0] * d1[1] - dp[1] * d1[0]) / denom
    EPS = 1e-10
    if EPS < t < 1.0 - EPS and EPS < u < 1.0 - EPS:
        return p0 + t * d1
    return None


def _clean_offset_curve(pts: np.ndarray) -> np.ndarray:
    """Remove self-intersection loops from an offset curve.

    At a sharp turn the inner offset boundary can cross itself.  This walks
    the polyline and replaces each loop (segment i→i+1 crosses later segment
    j→j+1) with the single intersection point.
    """
    if len(pts) < 4:
        return pts
    out = []
    i, n = 0, len(pts)
    while i < n - 1:
        out.append(pts[i])
        best_j, best_pt = -1, None
        for j in range(i + 2, n - 1):
            ip = _seg_intersect_2d(pts[i], pts[i + 1], pts[j], pts[j + 1])
            if ip is not None:
                best_j, best_pt = j, ip
        if best_j >= 0:
            out.append(best_pt)
            i = best_j + 1
        else:
            i += 1
    if i == n - 1:
        out.append(pts[-1])
    return np.array(out, dtype=float)


def generateTunnelBoundaries(tunnel_path, tunnel_width):
    """Generate smooth left and right boundaries for a tunnel path.
    
    Creates smooth wave-like boundaries by:
    1. Computing normals at each point
    2. Offsetting by half_width along normals
    3. Removing self-intersection loops (at sharp turns the inner boundary
       can cross itself)
    4. Smoothing the boundaries using spline interpolation
    
    Args:
        tunnel_path: List of (x, y) tuples representing the centerline.
        tunnel_width: Width of the tunnel.
    
    Returns:
        Tuple of (left_boundary, right_boundary) as arrays of shape (N, 2).
    """
    from scipy.interpolate import splprep, splev
    
    path = np.asarray(tunnel_path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("tunnel_path must be of shape (N, 2)")
    
    n_pts = path.shape[0]
    if n_pts < 2:
        return path, path
    
    # Compute per-point tangents using central differences
    tangents = np.zeros_like(path)
    for i in range(n_pts):
        if i == 0:
            t_vec = path[i + 1] - path[i]
        elif i == n_pts - 1:
            t_vec = path[i] - path[i - 1]
        else:
            t_vec = path[i + 1] - path[i - 1]
        norm = np.linalg.norm(t_vec)
        if norm <= 1e-12:
            tangents[i] = np.array([1.0, 0.0])
        else:
            tangents[i] = t_vec / norm
    
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    
    half_widths = np.full(n_pts, tunnel_width / 2.0) if np.isscalar(tunnel_width) \
              else np.asarray(tunnel_width, dtype=float) / 2.0
    # error check
    if half_widths.shape[0] != n_pts:
        raise ValueError(f"width_profile length {half_widths.shape[0]} != path length {n_pts}")
    half_widths = half_widths[:, np.newaxis]

    # Compute raw offset boundaries
    left_boundary_raw  = path + half_widths * normals
    right_boundary_raw = path - half_widths * normals

    def smooth_boundary(boundary_points):
        """Smooth a boundary using spline interpolation."""
        boundary_points = np.asarray(boundary_points, dtype=float)
        if boundary_points.shape[0] < 4:
            return boundary_points
        
        try:
            tck, u = splprep([boundary_points[:, 0], boundary_points[:, 1]], 
                           s=0.0, k=min(3, boundary_points.shape[0] - 1))
            # Evaluate at a dense set of points so the subsequent loop
            # cleaning has enough resolution to detect crossings.
            n_eval = max(boundary_points.shape[0], 200)
            u_eval = np.linspace(0, 1, n_eval)
            smoothed = np.array(splev(u_eval, tck)).T
            return smoothed
        except (ValueError, TypeError):
            return boundary_points
    
    # Smooth first, then clean: the cubic spline can overshoot at sharp
    # corners and re-introduce the loops that raw-offset cleaning removed.
    left_boundary = _clean_offset_curve(smooth_boundary(left_boundary_raw))
    right_boundary = _clean_offset_curve(smooth_boundary(right_boundary_raw))
    
    return left_boundary, right_boundary


class DDMLivePlot:
    """Live visualization of DDM accumulation with saving support.

    Usage:
        viz = DDMLivePlot(threshold=1.0, title='DDM Accumulation', show=True)
        viz.update(A=0.1, Pr=0.05, step_idx=0, trigger=False)
        ...
        viz.finalize(save_path='results/figures/ddm_accumulation.png')
    """

    def __init__(self, threshold=1.0, title='DDM Accumulation', show=True, max_points=100000,
                 window_size=400, ylim=None):
        self.threshold = float(threshold)
        self.show = bool(show)
        self.max_points = int(max_points)
        self.window_size = int(window_size)

        self.ts = []
        self.As = []
        self.Prs = []            # combined evidence
        self.Prs_react = []      # reactive evidence only
        self.Prs_pred = []       # predictive risk only (normalized)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title(title)
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Accumulator A / Evidence Pr')
        self.ax.grid(True, alpha=0.3)

        # Lines
        (self.line_A,) = self.ax.plot([], [], color='tab:blue', label='A (accumulator)')
        (self.line_Pr,) = self.ax.plot([], [], color='tab:orange', alpha=0.9, linewidth=1, label='Pr total')
        (self.line_Pr_react,) = self.ax.plot([], [], color='tab:green', alpha=0.7, linestyle='--', label='Pr reactive', linewidth=1)
        (self.line_Pr_pred,) = self.ax.plot([], [], color='tab:red', alpha=0.7, linestyle='--', label='Pr predictive (z)', linewidth=1)

        # Thresholds (dynamic placeholders, updated per-step if T_eff provided)
        self.upper = self.ax.axhline(self.threshold, color='tab:red', linestyle=':', linewidth=1, label='+T')
        self.lower = self.ax.axhline(-self.threshold, color='tab:red', linestyle=':', linewidth=1, label='-T')
        
        # Place legend outside plot to avoid overlap with data
        self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True, fontsize=9)

        # Fixed axis ranges
        if ylim is None:
            margin = 0.75 * abs(self.threshold)
            self._ylim = (-3 * abs(self.threshold) - margin, 3 * abs(self.threshold) + margin)
        else:
            self._ylim = (float(ylim[0]), float(ylim[1]))
        self.ax.set_ylim(*self._ylim)
        self.ax.set_xlim(0, max(1, self.window_size))

        if self.show:
            plt.ion()
            self.fig.show()
            # Place the DDM window at left-center to avoid overlap with pygame
            try:
                position_matplotlib_window(self.fig, align="left_center")
            except Exception:
                pass
            self.fig.canvas.draw()

    def update(self, A, Pr=None, step_idx=None, trigger=False, Pr_react=None, Pr_pred=None):
        if step_idx is None:
            step_idx = len(self.ts)

        self.ts.append(step_idx)
        self.As.append(float(A))
        if Pr is not None:
            self.Prs.append(float(Pr))
        else:
            self.Prs.append(np.nan)
        if Pr_react is not None:
            self.Prs_react.append(float(Pr_react))
        else:
            self.Prs_react.append(np.nan)
        if Pr_pred is not None:
            self.Prs_pred.append(float(Pr_pred))
        else:
            self.Prs_pred.append(np.nan)

        # Limit stored points to avoid memory bloat
        if len(self.ts) > self.max_points:
            self.ts = self.ts[-self.max_points:]
            self.As = self.As[-self.max_points:]
            self.Prs = self.Prs[-self.max_points:]
            self.Prs_react = self.Prs_react[-self.max_points:]
            self.Prs_pred = self.Prs_pred[-self.max_points:]

        # Use a fixed-size sliding window for display
        start = max(0, len(self.ts) - self.window_size)
        xs = self.ts[start:]
        ys_A = self.As[start:]
        ys_Pr = self.Prs[start:]
        ys_Pr_react = self.Prs_react[start:]
        ys_Pr_pred = self.Prs_pred[start:]
        self.line_A.set_data(xs, ys_A)
        self.line_Pr.set_data(xs, ys_Pr)
        self.line_Pr_react.set_data(xs, ys_Pr_react)
        self.line_Pr_pred.set_data(xs, ys_Pr_pred)
        self.ax.set_xlim(start, start + max(self.window_size, 1))
        self.ax.set_ylim(*self._ylim)

        # Keep thresholds at fixed +/- self.threshold now that risk shaping is removed
        self.upper.set_ydata([self.threshold, self.threshold])
        self.lower.set_ydata([-self.threshold, -self.threshold])

        if trigger:
            self.ax.axvline(step_idx, color='tab:gray', linestyle=':', linewidth=1)

        if self.show:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def finalize(self, save_path=None):
        if save_path:
            # Render full process for saving
            self.line_A.set_data(self.ts, self.As)
            self.line_Pr.set_data(self.ts, self.Prs)
            self.line_Pr_react.set_data(self.ts, self.Prs_react)
            self.line_Pr_pred.set_data(self.ts, self.Prs_pred)
            total_steps = len(self.ts) if self.ts else self.window_size
            self.ax.set_xlim(0, max(1, total_steps))
            self.ax.set_ylim(*self._ylim)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # bbox_inches='tight' ensures legend outside plot is included
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if self.show:
            try:
                plt.ioff()
            except Exception:
                pass
        plt.close(self.fig)


# ============================================================================
# Lasso Selection Task Helper Functions
# ============================================================================

@dataclass
class Icon:
    """Represents an icon in the grid for lasso selection tasks."""
    x: float
    y: float
    radius: float
    is_target: bool  # True for "X", False for "O"
    is_empty: bool = False  # True for empty cells (gray background)
    invisible: bool = False  # True for "O" (empty space, no grid, no collision)


def compute_window_size_for_lasso(
    grid_layout: List[str],
    grid_origin: Tuple[float, float],
    icon_spacing: float,
    end_radius: float,
    default_width: float = 0.46,
    default_height: float = 0.26
) -> Tuple[float, float]:
    """
    Compute window size based on grid layout for lasso tasks.
    
    Args:
        grid_layout: List of strings representing grid rows
        grid_origin: (x, y) origin of grid
        icon_spacing: Spacing between icon centers (m)
        end_radius: Radius for end point detection
        default_width: Default window width if grid is empty
        default_height: Default window height if grid is empty
        
    Returns:
        Tuple of (window_width, window_height) in meters
    """
    if not grid_layout:
        return default_width, default_height
    
    rows = len(grid_layout)
    max_cols = max(len(row.split()) if ' ' in row else len(row) for row in grid_layout) if grid_layout else 0
    
    if rows == 0 or max_cols == 0:
        return default_width, default_height
    
    origin_x, origin_y = grid_origin
    grid_width = (max_cols - 1) * icon_spacing
    grid_height = (rows - 1) * icon_spacing
    margin = max(icon_spacing, end_radius * 2, 0.05)
    
    window_width = origin_x + grid_width + margin
    window_height = origin_y + grid_height + margin
    
    return max(window_width, 0.3), max(window_height, 0.3)


def parse_grid_layout(
    grid_layout: List[str],
    grid_origin: Tuple[float, float],
    icon_spacing: float,
    icon_radius: float
) -> List[Icon]:
    """
    Parse grid layout string into Icon objects.
    
    Args:
        grid_layout: List of strings representing grid rows
        grid_origin: (x, y) origin of grid
        icon_spacing: Spacing between icon centers (m)
        icon_radius: Radius of icons (m)
        
    Returns:
        List of Icon objects
    """
    icons = []
    if not grid_layout:
        return icons
    
    origin_x, origin_y = grid_origin
    rows = len(grid_layout)
    max_cols = max(len(row.split()) if ' ' in row else len(row) for row in grid_layout) if grid_layout else 0
    
    for row_idx, row in enumerate(grid_layout):
        if ' ' in row:
            cells = row.split()
        else:
            cells = list(row)
        
        for col_idx in range(max_cols):
            x = origin_x + col_idx * icon_spacing
            y = origin_y + row_idx * icon_spacing
            
            char = cells[col_idx] if col_idx < len(cells) else ' '
            
            if char == 'X':
                icons.append(Icon(x=x, y=y, radius=icon_radius, is_target=True, is_empty=False))
            elif char == 'O':
                icons.append(Icon(x=x, y=y, radius=icon_radius, is_target=False, is_empty=True, invisible=True))
            else:
                icons.append(Icon(x=x, y=y, radius=icon_radius, is_target=False, is_empty=True))
    
    return icons


def compute_start_position_for_lasso(
    icons: List[Icon],
    grid_origin: Tuple[float, float],
    icon_spacing: float,
    icon_radius: float,
    margin_size: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute start position based on upper left corner of target cluster.
    
    Args:
        icons: List of Icon objects
        grid_origin: (x, y) origin of grid
        icon_spacing: Spacing between icon centers (m)
        icon_radius: Radius of icons (m)
        margin_size: Safety margin size (computed if None)
        
    Returns:
        Start position (x, y)
    """
    if not icons:
        return (0.1, 0.1)
    
    targets = [icon for icon in icons if icon.is_target]
    if not targets:
        origin_x, origin_y = grid_origin
        return (origin_x - icon_spacing * 0.5, origin_y + icon_spacing * 0.5)
    
    target_ys = [icon.y for icon in targets]
    min_y = min(target_ys)
    target_xs = [icon.x for icon in targets if icon.y == min_y]
    min_x = min(target_xs)
    
    gap_offset = icon_spacing * 0.5
    corner_x = min_x - gap_offset
    corner_y = min_y - gap_offset
    
    if margin_size is None:
        margin_size = icon_spacing - 2 * icon_radius
    
    start_x = corner_x - margin_size
    start_y = corner_y
    
    return (start_x, start_y)


def compute_end_position_for_lasso(
    icons: List[Icon],
    grid_origin: Tuple[float, float],
    icon_spacing: float
) -> Tuple[float, float]:
    """
    Compute end position based on upper left corner of target cluster.
    
    Args:
        icons: List of Icon objects
        grid_origin: (x, y) origin of grid
        icon_spacing: Spacing between icon centers (m)
        
    Returns:
        End position (x, y)
    """
    if not icons:
        return (0.1, 0.1)
    
    targets = [icon for icon in icons if icon.is_target]
    if not targets:
        origin_x, origin_y = grid_origin
        return (origin_x - icon_spacing * 0.5, origin_y + icon_spacing * 0.5)
    
    target_ys = [icon.y for icon in targets]
    min_y = min(target_ys)
    target_xs = [icon.x for icon in targets if icon.y == min_y]
    min_x = min(target_xs)
    
    gap_offset = icon_spacing * 0.5
    corner_x = min_x - gap_offset
    corner_y = min_y - gap_offset
    
    end_x = corner_x
    end_y = corner_y + gap_offset * 0.5
    
    return (end_x, end_y)


def generate_lasso_boundary_waypoints(
    icons: List[Icon],
    grid_origin: Tuple[float, float],
    icon_spacing: float
) -> List[Tuple[float, float]]:
    """
    Generate waypoints that trace the boundary around target icons (clockwise loop).
    
    Args:
        icons: List of Icon objects
        grid_origin: (x, y) origin of grid
        icon_spacing: Spacing between icon centers (m)
        
    Returns:
        List of (x, y) waypoints tracing the boundary
    """
    if not icons:
        return []
    
    targets = [icon for icon in icons if icon.is_target]
    if not targets:
        return []
    
    origin_x, origin_y = grid_origin
    spacing = icon_spacing
    
    # Map targets to grid coordinates
    target_grid = set()
    for icon in targets:
        c = round((icon.x - origin_x) / spacing)
        r = round((icon.y - origin_y) / spacing)
        target_grid.add((r, c))
    
    start_r, start_c = min(target_grid)
    curr_r, curr_c = start_r, start_c
    dir_r, dir_c = 0, 1
    path_indices = [(curr_r, curr_c)]
    
    def is_target(r, c):
        return (r, c) in target_grid
    
    max_steps = len(target_grid) * 4 + 20
    
    for _ in range(max_steps):
        dirs = [
            (dir_c, -dir_r),   # Right
            (dir_r, dir_c),    # Straight
            (-dir_c, dir_r)    # Left
        ]
        
        found_next = False
        for next_dr, next_dc in dirs:
            if next_dr == -1 and next_dc == 0:
                valid = is_target(curr_r - 1, curr_c) and not is_target(curr_r - 1, curr_c - 1)
            elif next_dr == 0 and next_dc == 1:
                valid = is_target(curr_r, curr_c) and not is_target(curr_r - 1, curr_c)
            elif next_dr == 1 and next_dc == 0:
                valid = is_target(curr_r, curr_c - 1) and not is_target(curr_r, curr_c)
            elif next_dr == 0 and next_dc == -1:
                valid = is_target(curr_r - 1, curr_c - 1) and not is_target(curr_r, curr_c - 1)
            else:
                valid = False
            
            if valid:
                curr_r += next_dr
                curr_c += next_dc
                dir_r, dir_c = next_dr, next_dc
                path_indices.append((curr_r, curr_c))
                found_next = True
                break
        
        if not found_next:
            break
        
        if (curr_r, curr_c) == (start_r, start_c):
            break
    
    # Filter to corners only
    if len(path_indices) > 2:
        filtered_indices = [path_indices[0]]
        for i in range(1, len(path_indices) - 1):
            prev = path_indices[i-1]
            curr = path_indices[i]
            next_p = path_indices[i+1]
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_p[0] - curr[0], next_p[1] - curr[1])
            if v1 != v2:
                filtered_indices.append(curr)
        filtered_indices.append(path_indices[-1])
        path_indices = filtered_indices
    
    # Convert to world coordinates
    waypoints = []
    gap_offset = spacing / 2.0
    for r, c in path_indices:
        wx = origin_x + c * spacing - gap_offset
        wy = origin_y + r * spacing - gap_offset
        waypoints.append((wx, wy))
    
    if waypoints and waypoints[0] != waypoints[-1]:
        waypoints.append(waypoints[0])
    
    return waypoints


def check_icon_collision(
    pos: Tuple[float, float],
    icons: List[Icon]
) -> Tuple[bool, Optional[Icon]]:
    """
    Check if position collides with any icon boundary.
    
    Args:
        pos: Position (x, y) to check
        icons: List of Icon objects
        
    Returns:
        Tuple of (collision_detected, icon) where icon is the colliding icon or None
    """
    px, py = pos
    for icon in icons:
        if icon.invisible:
            continue
        dist = np.sqrt((px - icon.x)**2 + (py - icon.y)**2)
        if dist <= icon.radius:
            return True, icon
    return False, None


def check_line_segment_icon_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    icons: List[Icon],
    num_samples: int = 50
) -> Tuple[bool, Optional[Icon]]:
    """
    Check if a line segment intersects with any icon boundary.
    
    Args:
        p1: Start point of line segment (x, y)
        p2: End point of line segment (x, y)
        icons: List of Icon objects to check against
        num_samples: Number of points to sample along the segment
        
    Returns:
        Tuple of (intersection_detected, icon) where icon is the intersecting icon or None
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Sample points along the line segment
    for i in range(num_samples + 1):
        t = i / num_samples
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        collision, icon = check_icon_collision((x, y), icons)
        if collision:
            return True, icon
    
    return False, None


def has_nearby_distractor(
    waypoint: Tuple[float, float],
    icons: List[Icon],
    threshold: float = 0.05
) -> bool:
    """
    Check if there are any distractor icons (non-target, non-empty) nearby a waypoint.
    
    Args:
        waypoint: Waypoint position (x, y)
        icons: List of Icon objects
        threshold: Distance threshold to consider "nearby" (meters)
        
    Returns:
        True if there are nearby distractor icons, False otherwise
    """
    wx, wy = waypoint
    for icon in icons:
        if icon.invisible or icon.is_target or icon.is_empty:
            continue
        
        # This is a distractor icon
        dist = np.sqrt((wx - icon.x)**2 + (wy - icon.y)**2)
        if dist <= threshold:
            return True
    
    return False


def simplify_lasso_path(
    waypoints: List[Tuple[float, float]],
    icons: List[Icon]
) -> List[Tuple[float, float]]:
    """
    Simplify lasso path by removing interior turning points when safe.
    
    This function performs a second pass to make the path more convex by removing
    interior turning points, but only if:
    1. There are no nearby distractor icons
    2. Removing the point won't cause the path to intersect with any icons
    
    Args:
        waypoints: List of waypoints defining the path
        icons: List of Icon objects (targets, distractors, empty cells)
        
    Returns:
        Simplified list of waypoints
    """
    if len(waypoints) <= 3:
        # Need at least 3 points to form a path, can't simplify
        return waypoints
    
    # Make a copy to avoid modifying the original
    simplified = waypoints.copy()
    
    # Remove duplicate start/end if path is closed
    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    
    if len(simplified) <= 3:
        return simplified  # Can't simplify further, return the processed copy
    
    # Try to remove interior points (not first or last)
    # We iterate backwards to avoid index shifting issues when removing elements
    i = len(simplified) - 2  # Start from second-to-last (last is end, first is start)
    
    while i > 0:
        # Skip first and last points (start and end positions) - these should never be removed
        if i == 0 or i >= len(simplified) - 1:
            i -= 1
            continue
        
        current_point = simplified[i]
        
        # Check if there are nearby distractors
        if has_nearby_distractor(current_point, icons):
            # Keep this point if there are nearby distractors
            i -= 1
            continue
        
        # Try removing this point and check if the new segment intersects icons
        prev_point = simplified[i - 1]
        next_point = simplified[i + 1]
        
        # Check if the line segment from prev to next intersects any icons
        intersects, _ = check_line_segment_icon_intersection(
            prev_point,
            next_point,
            icons
        )
        
        if not intersects:
            # Safe to remove this point - it makes the path more convex
            simplified.pop(i)
            # Don't decrement i since we removed an element (next iteration will check i-1)
        else:
            # Keep this point, it's needed to avoid icon collision
            i -= 1
    
    # Note: We don't force the path to be closed here because start_pos and end_pos
    # might have been added, making it an open path. The caller will handle closing
    # if needed (e.g., for SharpReferencePath which can handle both open and closed paths).
    
    return simplified


# ============================================================================
# Constraint conversion functions
# ============================================================================

class CoordinateConverter:
    """Helper class to convert pixel coordinates to normalized coordinates."""
    def __init__(self, viewport_width: int, viewport_height: int):
        self.viewport_width = float(viewport_width)
        self.viewport_height = float(viewport_height)
    
    def convert_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Convert a point from pixel to normalized coordinates."""
        return (point[0] / self.viewport_width, point[1] / self.viewport_height)
    
    def convert_geometry(self, geometry):
        """Convert geometry from pixel to normalized coordinates."""
        from simulators.constraints import (
            PointConstraint, LineConstraint, PolygonConstraint,
            CircleConstraint, RectangleConstraint
        )
        
        if isinstance(geometry, PointConstraint):
            x, y = self.convert_point((geometry.x, geometry.y))
            radius = geometry.radius / self.viewport_width if geometry.radius else None
            return PointConstraint(x=x, y=y, radius=radius)
        
        elif isinstance(geometry, LineConstraint):
            start = self.convert_point(geometry.start)
            end = self.convert_point(geometry.end)
            width = geometry.width / self.viewport_width if geometry.width else None
            return LineConstraint(start=start, end=end, width=width)
        
        elif isinstance(geometry, PolygonConstraint):
            vertices = [self.convert_point(v) for v in geometry.vertices]
            return PolygonConstraint(vertices=vertices, closed=geometry.closed)
        
        elif isinstance(geometry, CircleConstraint):
            center = self.convert_point(geometry.center)
            radius = geometry.radius / self.viewport_width
            return CircleConstraint(center=center, radius=radius)
        
        elif isinstance(geometry, RectangleConstraint):
            x, y = self.convert_point((geometry.x, geometry.y))
            width = geometry.width / self.viewport_width
            height = geometry.height / self.viewport_height
            return RectangleConstraint(x=x, y=y, width=width, height=height)
        
        return geometry


def _distance_to_circle(
    point: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    normal: np.ndarray
) -> Optional[float]:
    """Compute signed distance from point to circle along normal direction.
    
    Returns:
        Negative if circle center is on left side of normal, positive if on right.
        None if circle doesn't intersect the normal line.
    """
    center_arr = np.array(center, dtype=float)
    vec_to_center = center_arr - point
    
    # Project onto normal to determine side
    lateral_component = np.dot(vec_to_center, normal)
    
    # Distance to circle center
    dist_to_center = np.linalg.norm(vec_to_center)
    
    # Distance to circle edge along normal
    if dist_to_center < radius:
        # Point is inside circle
        return lateral_component - radius
    else:
        # Closest point on circle along normal direction
        closest_on_normal = point + normal * lateral_component
        dist_to_closest = np.linalg.norm(closest_on_normal - center_arr)
        if dist_to_closest < radius:
            # Circle intersects normal line
            # Compute distance to edge
            edge_dist = np.sqrt(radius**2 - (dist_to_closest**2))
            return lateral_component - edge_dist
        else:
            # Circle doesn't intersect normal line
            return None


def _distance_to_rectangle(
    point: np.ndarray,
    rect_x: float,
    rect_y: float,
    rect_width: float,
    rect_height: float,
    normal: np.ndarray
) -> Optional[float]:
    """Compute signed distance from point to rectangle along normal direction."""
    # Rectangle corners
    corners = np.array([
        [rect_x, rect_y],
        [rect_x + rect_width, rect_y],
        [rect_x + rect_width, rect_y + rect_height],
        [rect_x, rect_y + rect_height]
    ], dtype=float)
    
    # Check if point is inside rectangle
    if (rect_x <= point[0] <= rect_x + rect_width and
        rect_y <= point[1] <= rect_y + rect_height):
        # Point is inside, find closest edge
        dist_left = point[0] - rect_x
        dist_right = rect_x + rect_width - point[0]
        dist_top = point[1] - rect_y
        dist_bottom = rect_y + rect_height - point[1]
        
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        
        # Determine which side the closest edge is on relative to normal
        if min_dist == dist_left:
            # Left edge - check if it's on the left side of normal
            edge_point = np.array([rect_x, point[1]])
            vec_to_edge = edge_point - point
            lateral = np.dot(vec_to_edge, normal)
            return -min_dist if lateral < 0 else min_dist
        elif min_dist == dist_right:
            edge_point = np.array([rect_x + rect_width, point[1]])
            vec_to_edge = edge_point - point
            lateral = np.dot(vec_to_edge, normal)
            return min_dist if lateral > 0 else -min_dist
        elif min_dist == dist_top:
            edge_point = np.array([point[0], rect_y])
            vec_to_edge = edge_point - point
            lateral = np.dot(vec_to_edge, normal)
            return -min_dist if lateral < 0 else min_dist
        else:  # dist_bottom
            edge_point = np.array([point[0], rect_y + rect_height])
            vec_to_edge = edge_point - point
            lateral = np.dot(vec_to_edge, normal)
            return min_dist if lateral > 0 else -min_dist
    
    # Point is outside rectangle
    # Find closest point on rectangle boundary
    closest_x = np.clip(point[0], rect_x, rect_x + rect_width)
    closest_y = np.clip(point[1], rect_y, rect_y + rect_height)
    closest_point = np.array([closest_x, closest_y])
    
    vec_to_closest = closest_point - point
    dist = np.linalg.norm(vec_to_closest)
    lateral = np.dot(vec_to_closest, normal)
    
    return lateral if abs(lateral) > 1e-9 else None


def _distance_to_polygon(
    point: np.ndarray,
    vertices: List[Tuple[float, float]],
    normal: np.ndarray,
    closed: bool = True
) -> Optional[float]:
    """Compute signed distance from point to polygon along normal direction."""
    vertices_arr = np.array(vertices, dtype=float)
    
    # Check if point is inside polygon (using ray casting)
    n_vertices = len(vertices)
    inside = False
    j = n_vertices - 1
    
    for i in range(n_vertices):
        vi = vertices_arr[i]
        vj = vertices_arr[j]
        if ((vi[1] > point[1]) != (vj[1] > point[1])) and \
           (point[0] < (vj[0] - vi[0]) * (point[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]):
            inside = not inside
        j = i
    
    if inside:
        # Point is inside polygon - find distance to nearest edge
        min_dist = float('inf')
        min_lateral = 0.0
        
        for i in range(n_vertices):
            if i == n_vertices - 1 and not closed:
                break
            v1 = vertices_arr[i]
            v2 = vertices_arr[(i + 1) % n_vertices]
            
            # Vector along edge
            edge_vec = v2 - v1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-9:
                continue
            
            # Vector from v1 to point
            to_point = point - v1
            
            # Project point onto edge
            t = np.clip(np.dot(to_point, edge_vec) / (edge_len**2), 0.0, 1.0)
            closest_on_edge = v1 + t * edge_vec
            
            # Distance to edge
            dist_vec = point - closest_on_edge
            dist = np.linalg.norm(dist_vec)
            
            if dist < min_dist:
                min_dist = dist
                lateral = np.dot(dist_vec, normal)
                min_lateral = -min_dist if lateral < 0 else min_dist
        
        return min_lateral if min_dist < float('inf') else None
    else:
        # Point is outside - find closest point on boundary
        min_dist = float('inf')
        min_lateral = None
        
        for i in range(n_vertices):
            if i == n_vertices - 1 and not closed:
                break
            v1 = vertices_arr[i]
            v2 = vertices_arr[(i + 1) % n_vertices]
            
            edge_vec = v2 - v1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-9:
                continue
            
            to_point = point - v1
            t = np.clip(np.dot(to_point, edge_vec) / (edge_len**2), 0.0, 1.0)
            closest_on_edge = v1 + t * edge_vec
            
            dist_vec = point - closest_on_edge
            dist = np.linalg.norm(dist_vec)
            lateral = np.dot(dist_vec, normal)
            
            if dist < min_dist:
                min_dist = dist
                min_lateral = lateral
        
        return min_lateral if min_lateral is not None else None


def _distance_to_polyline(
    point: np.ndarray,
    polyline: List[Tuple[float, float]],
    normal: np.ndarray
) -> Optional[float]:
    """Compute signed distance from point to polyline along normal direction."""
    if not polyline or len(polyline) < 2:
        return None
    
    polyline_arr = np.array(polyline, dtype=float)
    min_dist = float('inf')
    min_lateral = None
    
    for i in range(len(polyline) - 1):
        v1 = polyline_arr[i]
        v2 = polyline_arr[i + 1]
        
        edge_vec = v2 - v1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-9:
            continue
        
        to_point = point - v1
        t = np.clip(np.dot(to_point, edge_vec) / (edge_len**2), 0.0, 1.0)
        closest_on_edge = v1 + t * edge_vec
        
        dist_vec = point - closest_on_edge
        dist = np.linalg.norm(dist_vec)
        lateral = np.dot(dist_vec, normal)
        
        if dist < min_dist:
            min_dist = dist
            min_lateral = lateral
    
    return min_lateral if min_lateral is not None else None


def _distance_to_obstacle(
    point: np.ndarray,
    geometry,
    normal: np.ndarray
) -> Optional[float]:
    """Compute signed distance from point to obstacle along normal direction.
    
    Returns:
        Negative if obstacle is on left side, positive if on right side, None if not relevant.
    """
    from simulators.constraints import (
        PointConstraint, CircleConstraint, RectangleConstraint,
        PolygonConstraint, LineConstraint
    )
    
    if isinstance(geometry, (PointConstraint, CircleConstraint)):
        if isinstance(geometry, PointConstraint):
            center = (geometry.x, geometry.y)
            radius = geometry.radius if geometry.radius else 0.0
        else:
            center = geometry.center
            radius = geometry.radius
        
        return _distance_to_circle(point, center, radius, normal)
    
    elif isinstance(geometry, RectangleConstraint):
        return _distance_to_rectangle(
            point, geometry.x, geometry.y, geometry.width, geometry.height, normal
        )
    
    elif isinstance(geometry, PolygonConstraint):
        return _distance_to_polygon(point, geometry.vertices, normal, geometry.closed)
    
    elif isinstance(geometry, LineConstraint):
        # Treat line as a thin rectangle
        start = np.array(geometry.start)
        end = np.array(geometry.end)
        edge_vec = end - start
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-9:
            return None
        
        # Create a thin rectangle along the line
        width = geometry.width if geometry.width else 0.001
        perp = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        half_width = width / 2.0
        
        # Create rectangle corners
        corners = [
            start - perp * half_width,
            start + perp * half_width,
            end + perp * half_width,
            end - perp * half_width
        ]
        
        return _distance_to_polygon(point, [tuple(c) for c in corners], normal, closed=True)
    
    return None


def _distance_to_boundary(
    point: np.ndarray,
    geometry,
    normal: np.ndarray
) -> Optional[float]:
    """Compute signed distance from point to boundary along normal direction.
    
    For KEEP_IN constraints, returns positive if point is inside, negative if outside.
    """
    from simulators.constraints import (
        RectangleConstraint, PolygonConstraint, CircleConstraint
    )
    
    if isinstance(geometry, CircleConstraint):
        center = np.array(geometry.center)
        radius = geometry.radius
        vec_to_center = center - point
        dist_to_center = np.linalg.norm(vec_to_center)
        
        # Distance to boundary
        dist_to_boundary = radius - dist_to_center
        lateral = np.dot(vec_to_center, normal)
        
        # If inside, return positive distance; if outside, return negative
        return dist_to_boundary if dist_to_center < radius else -dist_to_boundary
    
    elif isinstance(geometry, RectangleConstraint):
        # Check if inside
        inside = (geometry.x <= point[0] <= geometry.x + geometry.width and
                 geometry.y <= point[1] <= geometry.y + geometry.height)
        
        dist = _distance_to_rectangle(
            point, geometry.x, geometry.y, geometry.width, geometry.height, normal
        )
        
        if dist is None:
            return None
        
        # For KEEP_IN, we want positive if inside, negative if outside
        return abs(dist) if inside else -abs(dist)
    
    elif isinstance(geometry, PolygonConstraint):
        # Check if inside
        vertices_arr = np.array(geometry.vertices)
        n_vertices = len(vertices_arr)
        inside = False
        j = n_vertices - 1
        
        for i in range(n_vertices):
            vi = vertices_arr[i]
            vj = vertices_arr[j]
            if ((vi[1] > point[1]) != (vj[1] > point[1])) and \
               (point[0] < (vj[0] - vi[0]) * (point[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]):
                inside = not inside
            j = i
        
        dist = _distance_to_polygon(point, geometry.vertices, normal, geometry.closed)
        
        if dist is None:
            return None
        
        # For KEEP_IN, positive if inside, negative if outside
        return abs(dist) if inside else -abs(dist)
    
    return None


def convert_constraints_to_corridor_bounds(
    constraints,
    reference_path,
    viewport_size: Optional[Tuple[int, int]] = None
) -> Tuple[Callable, Callable]:
    """
    Convert user-defined constraints to corridor_bounds functions.

    Args:
        constraints: ConstraintConfig with user-defined constraints
        reference_path: ReferencePath object for the trajectory
        viewport_size: (width, height) if coordinates are in pixels

    Returns:
        Tuple of (left_bound_func, right_bound_func) where each function
        takes path progress s and returns maximum allowed distance from path
    """
    from simulators.constraints import ConstraintConfig, ConstraintType
    
    if not isinstance(constraints, ConstraintConfig):
        raise TypeError("constraints must be a ConstraintConfig instance")
    
    # Convert pixel coordinates to normalized if needed
    converter = None
    coordinate_system = getattr(constraints, 'coordinate_system', 'normalized')
    if coordinate_system == "pixel" and viewport_size:
        converter = CoordinateConverter(viewport_size[0], viewport_size[1])
    
    # Sample reference path
    num_samples = max(50, int(reference_path.total_length / 0.01))
    s_samples = np.linspace(0, reference_path.total_length, num_samples)
    
    left_bounds = []
    right_bounds = []
    
    for s in s_samples:
        # Get point and normal on reference path
        path_point = np.array(reference_path(s), dtype=float)
        normal = reference_path.normal(s)  # Right-pointing normal
        
        # Initialize bounds (large = unconstrained)
        left_bound = 1e6
        right_bound = 1e6
        
        # Process each constraint region
        for region in constraints.regions:
            
            
            # Convert geometry to normalized if needed
            geometry = region.geometry
            if converter:
                geometry = converter.convert_geometry(geometry)
            
            # Compute distance based on constraint type
            if region.constraint_type == ConstraintType.KEEP_OUT:
                # Obstacle: cursor should stay away
                dist = _distance_to_obstacle(path_point, geometry, normal)
                if dist is not None:
                    # Obstacle on left side (negative normal direction)
                    if dist < 0:
                        margin = getattr(region, 'margin', 0.0)
                        safe_dist = abs(dist) - margin
                        left_bound = min(left_bound, max(0.0, safe_dist))
                    # Obstacle on right side (positive normal direction)
                    else:
                        margin = getattr(region, 'margin', 0.0)
                        safe_dist = dist - margin
                        right_bound = min(right_bound, max(0.0, safe_dist))
            
            elif region.constraint_type == ConstraintType.KEEP_IN:
                # Boundary: cursor should stay inside
                # For KEEP_IN, we compute distance to boundary edge and constrain both sides
                # We need to find the distance to the nearest boundary edge
                from simulators.constraints import (
                    RectangleConstraint, PolygonConstraint, CircleConstraint, PathConstraint
                )
                
                # Handle PathConstraint (tunnel centerline with width)
                if isinstance(geometry, PathConstraint):
                    # For PathConstraint, the path represents the centerline
                    # The width defines the corridor: cursor should stay within width/2 on each side
                    if geometry.width is not None and geometry.width > 0:
                        half_width = geometry.width / 2.0
                        # Use default_margin from constraint_config if available, otherwise 0
                        default_margin = getattr(constraints, 'default_margin', 0.0)
                        safe_dist = half_width - default_margin
                        # Apply to both sides (symmetric corridor)
                        left_bound = min(left_bound, max(0.0, safe_dist))
                        right_bound = min(right_bound, max(0.0, safe_dist))
                
                # Compute distance to boundary edge
                elif isinstance(geometry, CircleConstraint):
                    center = np.array(geometry.center)
                    radius = geometry.radius
                    vec_to_center = center - path_point
                    dist_to_center = np.linalg.norm(vec_to_center)
                    
                    if dist_to_center < radius:
                        # Inside circle - distance to edge
                        dist_to_edge = radius - dist_to_center
                        # Apply conservatively to both sides
                        margin = getattr(region, 'margin', 0.0)
                        safe_dist = dist_to_edge - margin
                        left_bound = min(left_bound, max(0.0, safe_dist))
                        right_bound = min(right_bound, max(0.0, safe_dist))
                
                elif isinstance(geometry, RectangleConstraint):
                    # Check if inside
                    inside = (geometry.x <= path_point[0] <= geometry.x + geometry.width and
                             geometry.y <= path_point[1] <= geometry.y + geometry.height)
                    
                    if inside:
                        # Find distance to nearest edge
                        dist_left = path_point[0] - geometry.x
                        dist_right = geometry.x + geometry.width - path_point[0]
                        dist_top = path_point[1] - geometry.y
                        dist_bottom = geometry.y + geometry.height - path_point[1]
                        
                        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
                        margin = getattr(region, 'margin', 0.0)
                        safe_dist = min_dist - margin
                        left_bound = min(left_bound, max(0.0, safe_dist))
                        right_bound = min(right_bound, max(0.0, safe_dist))
                
                elif isinstance(geometry, PolygonConstraint):
                    # Check if inside and compute distance to nearest edge
                    vertices_arr = np.array(geometry.vertices)
                    n_vertices = len(vertices_arr)
                    inside = False
                    j = n_vertices - 1
                    
                    for i in range(n_vertices):
                        vi = vertices_arr[i]
                        vj = vertices_arr[j]
                        if ((vi[1] > path_point[1]) != (vj[1] > path_point[1])) and \
                           (path_point[0] < (vj[0] - vi[0]) * (path_point[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]):
                            inside = not inside
                        j = i
                    
                    if inside:
                        # Find distance to nearest edge
                        min_dist = float('inf')
                        for i in range(n_vertices):
                            if i == n_vertices - 1 and not geometry.closed:
                                break
                            v1 = vertices_arr[i]
                            v2 = vertices_arr[(i + 1) % n_vertices]
                            
                            edge_vec = v2 - v1
                            edge_len = np.linalg.norm(edge_vec)
                            if edge_len < 1e-9:
                                continue
                            
                            to_point = path_point - v1
                            t = np.clip(np.dot(to_point, edge_vec) / (edge_len**2), 0.0, 1.0)
                            closest_on_edge = v1 + t * edge_vec
                            
                            dist = np.linalg.norm(path_point - closest_on_edge)
                            min_dist = min(min_dist, dist)
                        
                        if min_dist < float('inf'):
                            margin = getattr(region, 'margin', 0.0)
                            safe_dist = min_dist - margin
                            left_bound = min(left_bound, max(0.0, safe_dist))
                            right_bound = min(right_bound, max(0.0, safe_dist))
        
        # Handle explicit left/right boundaries (tunnel case - legacy support)
        left_boundary = getattr(constraints, 'left_boundary', None)
        right_boundary = getattr(constraints, 'right_boundary', None)
        if left_boundary and right_boundary:
            if converter:
                left_boundary = [converter.convert_point(p) for p in left_boundary]
                right_boundary = [converter.convert_point(p) for p in right_boundary]
            
            left_dist = _distance_to_polyline(path_point, left_boundary, -normal)
            right_dist = _distance_to_polyline(path_point, right_boundary, normal)
            
            default_margin = getattr(constraints, 'default_margin', 0.0)
            if left_dist is not None:
                left_bound = min(left_bound, max(0.0, left_dist - default_margin))
            if right_dist is not None:
                right_bound = min(right_bound, max(0.0, right_dist - default_margin))
        
        # Handle symmetric width (legacy support)
        corridor_width = getattr(constraints, 'corridor_width', None)
        if corridor_width:
            half_width = corridor_width / 2.0
            left_bound = min(left_bound, half_width)
            right_bound = min(right_bound, half_width)
        
        left_bounds.append(max(0.0, left_bound))
        right_bounds.append(max(0.0, right_bound))
    
    # Create interpolation functions
    left_func = interp1d(
        s_samples, left_bounds,
        kind='linear',
        bounds_error=False,
        fill_value=(left_bounds[0], left_bounds[-1])
    )
    
    right_func = interp1d(
        s_samples, right_bounds,
        kind='linear',
        bounds_error=False,
        fill_value=(right_bounds[0], right_bounds[-1])
    )
    
    # Wrap to ensure scalar return
    def left_bound_func(s):
        return float(left_func(s))
    
    def right_bound_func(s):
        return float(right_func(s))
    
    return (left_bound_func, right_bound_func)

