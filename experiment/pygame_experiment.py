"""
Unified pygame experiment runner for all task types.

Accepts command-line arguments to specify which environment configuration to run.
Generates trajectory via hcs_package.CursorSimulator, replays in pygame, and saves result plots.

Usage:
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/sigmoidal_tunnel.json
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/corner_tunnel.json
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/lasso_selection.json
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/cascading_menu.json
"""
import os
import sys
import json
import argparse
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    cwd = str(Path.cwd())
    if cwd in sys.path:
        sys.path.remove(cwd)
    from hcs_package import CursorSimulator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "hcs_package" / "src"))
    from hcs_package import CursorSimulator

import pygame
from experiment.utils import draw_speed_profile, draw_trajectory
from experiment.render import (
    render_tunnel_frame, render_lasso_frame, render_menu_frame,
    precompute_tunnel_boundaries, precompute_icon_rects,
    m_to_px, SCALE
)
from experiment.environment import (
    load_environment_config, create_environment, generate_task_config
)

# Reuse the evaluation pipeline's plotting so pygame output matches eval figures.
sys.path.insert(0, str(Path(__file__).parent.parent / "eval" / "utils"))
from plot_utils import (
    plot_enhanced_speed_profiles,
    plot_speeds_vs_progress_enhanced,
)

DEFAULT_USER_CONFIG = Path(__file__).parent / "user_configurations" / "customized.json"


def get_experiment_name(env_type: str) -> str:
    """Get experiment name from environment type."""
    if "tunnel" in env_type or "corner" in env_type or "turning" in env_type:
        if "corner" in env_type or "turning" in env_type:
            return "turning"
        return "steering"
    elif "lasso" in env_type:
        return "lasso"
    elif "menu" in env_type:
        return "menu"
    return "experiment"


def save_evaluation_speed_figures(trajectory, environment, interval, run_dir):
    """Render the evaluation-style speed figures for a single sim run.

    Produces speeds_enhanced_t0.png and speeds_progress_enhanced_t0.png using
    eval/utils/plot_utils, so the speed output is directly comparable with the
    evaluation results. (The trajectory figure is drawn separately by
    draw_trajectory so it keeps the target point + radius marker.)

    Trajectory points arrive in task-config pixels; the evaluation works in
    metres at a 0.001 scale (matching environment["centerline"]), so we apply
    the same conversion and speed convention as run_eval.

    Returns the per-step speed list (m/s).
    """
    scale = 0.001  # task-config pixels -> metres (matches run_eval)
    sim_traj_m = [[x * scale, y * scale] for x, y, _ in trajectory]

    # Per-step speed = displacement / interval, with speeds[0] duplicated so
    # len(speeds) == len(trajectory) (identical convention to run_eval).
    speeds = []
    for i in range(1, len(sim_traj_m)):
        dx = sim_traj_m[i][0] - sim_traj_m[i - 1][0]
        dy = sim_traj_m[i][1] - sim_traj_m[i - 1][1]
        speeds.append(float(np.hypot(dx, dy)) / interval)
    if speeds:
        speeds.insert(0, speeds[0])

    trial_id = 0
    all_results = {trial_id: {0: {'sim': {'trajectories': [sim_traj_m], 'speeds': [speeds]}}}}
    tunnel_paths = {trial_id: environment.get("centerline", [])}

    plot_enhanced_speed_profiles(all_results, run_dir, time_step=interval)
    if tunnel_paths[trial_id]:
        plot_speeds_vs_progress_enhanced(all_results, run_dir, tunnel_paths, bin_size=0.1)

    return speeds


def run_tunnel_experiment(
    environment: dict,
    task_config: dict,
    user_config_path: str,
    run_dir: str,
    fps: int = 20
):
    """Run tunnel/steering/turning experiment."""
    SCREEN_W_PX = environment["screen_width"]
    MAX_STEPS = environment["max_steps"]
    TARGET_RADIUS_M = environment["target_radius"]
    TUNNEL_WIDTH_M = environment.get("width_profile") or environment["tunnel_width"]
    window_width_m = environment["window_width_m"]
    window_height_m = environment["window_height_m"]
    tunnel_path_m = environment["centerline"]
    target_pos_m = environment["target_pos"]
    left_boundary_m = environment["left_boundary"]
    right_boundary_m = environment["right_boundary"]

    # Generate trajectory
    print("Generating trajectory …")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(task_config, tf)
        task_file_path = tf.name

    try:
        simulator = CursorSimulator(user_config_path)
        trajectory, reference_path = simulator.generate_trajectory_with_waypoints(
            task_file=task_file_path,
            max_steps=MAX_STEPS,
            target_radius=TARGET_RADIUS_M,
            use_optimal_path=True,
            return_timestamps=False,
            return_reference_path=True,
        )
    finally:
        os.unlink(task_file_path)

    print(f"Generated {len(trajectory)} points")

    # Conversion from task-config pixels to metres (uniform 0.001 scale)
    SCREEN_H_PX = environment["screen_height"]
    s2m_x = window_width_m / SCREEN_W_PX
    s2m_y = window_height_m / SCREEN_H_PX
    interval = simulator.interval

    # Pygame replay
    display_w = int(window_width_m * SCALE)
    display_h = int(window_height_m * SCALE)

    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption(f"Experiment: {environment.get('env_type', 'tunnel')}")
    clock = pygame.time.Clock()

    left_boundary_px = [(m_to_px(x, SCALE), m_to_px(y, SCALE)) for x, y in left_boundary_m]
    right_boundary_px = [(m_to_px(x, SCALE), m_to_px(y, SCALE)) for x, y in right_boundary_m]

    trail = []
    running = True
    frame_idx = 0

    while running and frame_idx < len(trajectory):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        # Convert from task config pixels to display pixels
        x_task, y_task, _ = trajectory[frame_idx]
        cursor_pos_px = (int(x_task * s2m_x * SCALE), int(y_task * s2m_y * SCALE))
        trail.append(cursor_pos_px)

        render_tunnel_frame(
            screen,
            tunnel_path_m, TUNNEL_WIDTH_M,
            cursor_pos_px,
            target_pos_m, TARGET_RADIUS_M,
            trail_px=trail,
            scale=SCALE,
            left_boundary_px=left_boundary_px,
            right_boundary_px=right_boundary_px,
        )

        pygame.display.flip()
        clock.tick(fps)
        frame_idx += 1

    pygame.quit()

    # Trajectory plot (original style): cursor path, tunnel corridor, and the
    # target drawn as a red point with the target-radius circle around it.
    start_pos_m = tunnel_path_m[0] if tunnel_path_m else (0, 0)
    cursor_xs_m = [start_pos_m[0]]
    cursor_ys_m = [start_pos_m[1]]
    for x_px, y_px, _ in trajectory:
        cursor_xs_m.append(x_px * s2m_x)
        cursor_ys_m.append(y_px * s2m_y)

    pause_coordinates = []
    for i in range(1, len(cursor_xs_m)):
        dx = cursor_xs_m[i] - cursor_xs_m[i - 1]
        dy = cursor_ys_m[i] - cursor_ys_m[i - 1]
        if np.sqrt(dx ** 2 + dy ** 2) / interval < 1e-3:
            pause_coordinates.append((cursor_xs_m[i], cursor_ys_m[i]))

    # Pre-convert reference path from pixels/1000 to metres
    ref_path_m = None
    if reference_path is not None:
        try:
            ref_to_m_x = window_width_m / (SCREEN_W_PX / 1000)
            ref_to_m_y = window_height_m / (SCREEN_H_PX / 1000)
            theta_samples = np.linspace(0, reference_path.total_length, 200)
            ref_path_m = []
            for theta in theta_samples:
                try:
                    ref_pos = reference_path(theta)
                    ref_path_m.append((ref_pos[0] * ref_to_m_x, ref_pos[1] * ref_to_m_y))
                except:
                    continue
        except Exception as e:
            print(f"Warning: Could not convert reference path: {e}")

    draw_trajectory(
        cursor_xs_m, cursor_ys_m,
        target_pos_m, TARGET_RADIUS_M,
        window_width_m, window_height_m,
        tunnel_path=tunnel_path_m,
        tunnel_width=TUNNEL_WIDTH_M,
        pause_coordinates=pause_coordinates,
        tunnel_wall_left=True,
        tunnel_wall_right=True,
        reference_path=ref_path_m,
        save_path=os.path.join(run_dir, "plot_trajectory.png"),
        title="Cursor Trajectory",
    )

    # Speed figures in the evaluation format (speeds_enhanced, speeds_progress_enhanced)
    speeds = save_evaluation_speed_figures(trajectory, environment, interval, run_dir)

    return trajectory, speeds


def run_lasso_experiment(
    environment: dict,
    task_config: dict,
    user_config_path: str,
    run_dir: str,
    fps: int = 20
):
    """Run lasso selection experiment."""
    SCREEN_W_PX = environment["screen_width"]
    MAX_STEPS = environment["max_steps"]
    TARGET_RADIUS_M = environment["target_radius"]
    window_width_m = environment["window_width_m"]
    window_height_m = environment["window_height_m"]
    icons = environment["icons"]
    start_pos_m = environment["start_pos"]
    end_pos_m = environment["end_pos"]
    centerline = environment["centerline"]

    # Generate trajectory
    print("Generating trajectory …")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(task_config, tf)
        task_file_path = tf.name

    try:
        simulator = CursorSimulator(user_config_path)
        trajectory, reference_path = simulator.generate_trajectory_with_waypoints(
            task_file=task_file_path,
            max_steps=MAX_STEPS,
            target_radius=TARGET_RADIUS_M,
            use_optimal_path=True,
            return_timestamps=False,
            return_reference_path=True,
        )
    finally:
        os.unlink(task_file_path)

    print(f"Generated {len(trajectory)} points")

    # Convert trajectory to metres (use separate scales for X and Y)
    SCREEN_H_PX = environment["screen_height"]
    s2m_x = window_width_m / SCREEN_W_PX
    s2m_y = window_height_m / SCREEN_H_PX
    cursor_xs_m = [start_pos_m[0]]
    cursor_ys_m = [start_pos_m[1]]
    speeds = []
    pause_coordinates = []
    interval = simulator.interval

    for x_px, y_px, _ in trajectory:
        cursor_xs_m.append(x_px * s2m_x)
        cursor_ys_m.append(y_px * s2m_y)

    for i in range(1, len(cursor_xs_m)):
        dx = cursor_xs_m[i] - cursor_xs_m[i - 1]
        dy = cursor_ys_m[i] - cursor_ys_m[i - 1]
        speed = np.sqrt(dx ** 2 + dy ** 2) / interval
        speeds.append(speed)
        if speed < 1e-3:
            pause_coordinates.append((cursor_xs_m[i], cursor_ys_m[i]))

    # Pygame replay
    display_w = int(window_width_m * SCALE)
    display_h = int(window_height_m * SCALE)

    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption("Lasso Selection Experiment")
    clock = pygame.time.Clock()

    icon_rects = precompute_icon_rects(icons, SCALE)

    trail = []
    running = True
    frame_idx = 0

    while running and frame_idx < len(trajectory):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        # Convert from task config pixels to display pixels
        x_task, y_task, _ = trajectory[frame_idx]
        cursor_pos_px = (int(x_task * s2m_x * SCALE), int(y_task * s2m_y * SCALE))
        trail.append(cursor_pos_px)

        render_lasso_frame(
            screen,
            icons,
            cursor_pos_px,
            trail_px=trail,
            start_pos_m=start_pos_m,
            end_pos_m=end_pos_m,
            end_radius_m=TARGET_RADIUS_M,
            scale=SCALE,
            icon_rects=icon_rects,
        )

        pygame.display.flip()
        clock.tick(fps)
        frame_idx += 1

    pygame.quit()

    # Save plots
    draw_speed_profile(
        speeds,
        save_path=os.path.join(run_dir, "plot_speed.png"),
        title="Speed Profile - Lasso Selection",
    )

    # Lasso trajectory plot
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Draw PathConstraint corridor (matches actual constraint enforcement)
    corridor_width = environment.get("corridor_width", 0.01)
    if centerline and len(centerline) >= 2:
        from experiment.utils import generateTunnelBoundaries
        try:
            left_boundary, right_boundary = generateTunnelBoundaries(centerline, corridor_width)
            if left_boundary is not None and right_boundary is not None:
                poly_x = np.concatenate([left_boundary[:, 0], right_boundary[::-1, 0]])
                poly_y = np.concatenate([left_boundary[:, 1], right_boundary[::-1, 1]])
                ax.fill(poly_x, poly_y, color='blue', alpha=0.15, label='Constraint Corridor', zorder=1)
                ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'b--', linewidth=0.8, alpha=0.4)
                ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'b--', linewidth=0.8, alpha=0.4)
        except Exception as e:
            print(f"Warning: Could not draw constraint corridor: {e}")

    for icon in icons:
        if icon.get('is_empty', False) and not icon.get('is_target', False):
            color = 'lightgrey'
        elif icon.get('is_target', False):
            color = 'orange'
        else:
            color = 'gray'
        r = icon['radius']
        sq = Rectangle((icon['x'] - r, icon['y'] - r), r * 2, r * 2,
                        facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(sq)

    ax.add_patch(Circle(start_pos_m, 0.001, color='green', fill=True))
    ax.add_patch(Circle(end_pos_m, 0.001, color='red', fill=True))
    ax.add_patch(Circle(end_pos_m, TARGET_RADIUS_M,
                        color='red', fill=False, linestyle='--', alpha=0.5))

    if centerline:
        ref_xs = [w[0] for w in centerline]
        ref_ys = [w[1] for w in centerline]
        ax.plot(ref_xs, ref_ys, 'k--', linewidth=0.6, alpha=0.4, label='Centerline')

    # Draw optimized reference path
    # Reference path is in (pixels / 1000) format from hcs_package
    # Convert to meters: ref * window_m / (screen_px / 1000)
    if reference_path is not None:
        try:
            ref_to_m_x = window_width_m / (SCREEN_W_PX / 1000)
            ref_to_m_y = window_height_m / (SCREEN_H_PX / 1000)
            theta_samples = np.linspace(0, reference_path.total_length, 200)
            opt_xs, opt_ys = [], []
            for theta in theta_samples:
                try:
                    ref_pos = reference_path(theta)
                    opt_xs.append(ref_pos[0] * ref_to_m_x)
                    opt_ys.append(ref_pos[1] * ref_to_m_y)
                except:
                    continue
            if opt_xs and opt_ys:
                ax.plot(opt_xs, opt_ys, color='gray', linewidth=2.0, alpha=0.4,
                       label='Optimal Reference Path', linestyle='-')
        except Exception as e:
            print(f"Warning: Could not plot reference path: {e}")

    ax.plot(cursor_xs_m, cursor_ys_m, 'green', linewidth=1.0, label='Selection Loop', alpha=0.8)
    ax.scatter(cursor_xs_m[0], cursor_ys_m[0], color='green', zorder=5, label='Start')
    ax.scatter(cursor_xs_m[-1], cursor_ys_m[-1], color='blue', zorder=5, label='End')

    ax.set_xlim(0, window_width_m)
    ax.set_ylim(0, window_height_m)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Lasso Selection Trajectory')

    reached_end = False
    if len(cursor_xs_m) > 1:
        d2 = (cursor_xs_m[-1] - end_pos_m[0]) ** 2 + (cursor_ys_m[-1] - end_pos_m[1]) ** 2
        reached_end = d2 < TARGET_RADIUS_M ** 2
    status = "SUCCESS: Reached end" if reached_end else "INCOMPLETE"
    info_lines = [status, f"Steps: {len(cursor_xs_m)}", f"Time: {len(trajectory) * interval:.2f}s"]
    ax.text(0.02, 0.98, '\n'.join(info_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return trajectory, speeds


def run_menu_experiment(
    environment: dict,
    task_config: dict,
    user_config_path: str,
    run_dir: str,
    fps: int = 20
):
    """Run cascading menu experiment with macOS-style rendering."""
    SCREEN_W_PX = environment["screen_width"]
    MAX_STEPS = environment["max_steps"]
    TARGET_RADIUS_M = environment["target_radius"]
    window_width_m = environment["window_width_m"]
    window_height_m = environment["window_height_m"]
    start_pos_m = environment["start_pos"]
    target_pos_m = environment["target_pos"]
    main_menu_rect = environment["main_menu_rect"]
    sub_menu_rect = environment["sub_menu_rect"]
    main_menu_items = environment["main_menu_items"]
    sub_menu_items = environment["sub_menu_items"]

    # Generate trajectory
    print("Generating trajectory …")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(task_config, tf)
        task_file_path = tf.name

    try:
        simulator = CursorSimulator(user_config_path)
        trajectory, reference_path = simulator.generate_trajectory_with_waypoints(
            task_file=task_file_path,
            max_steps=MAX_STEPS,
            target_radius=TARGET_RADIUS_M,
            use_optimal_path=True,
            return_timestamps=False,
            return_reference_path=True,
        )
    finally:
        os.unlink(task_file_path)

    print(f"Generated {len(trajectory)} points")

    # Convert trajectory to metres (use separate scales for X and Y)
    SCREEN_H_PX = environment["screen_height"]
    s2m_x = window_width_m / SCREEN_W_PX
    s2m_y = window_height_m / SCREEN_H_PX
    cursor_xs_m = [start_pos_m[0]]
    cursor_ys_m = [start_pos_m[1]]
    speeds = []
    interval = simulator.interval

    for x_px, y_px, _ in trajectory:
        cursor_xs_m.append(x_px * s2m_x)
        cursor_ys_m.append(y_px * s2m_y)

    for i in range(1, len(cursor_xs_m)):
        dx = cursor_xs_m[i] - cursor_xs_m[i - 1]
        dy = cursor_ys_m[i] - cursor_ys_m[i - 1]
        speed = np.sqrt(dx ** 2 + dy ** 2) / interval
        speeds.append(speed)

    # Pygame replay
    display_w = int(window_width_m * SCALE)
    display_h = int(window_height_m * SCALE)

    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption("Cascading Menu Experiment")
    clock = pygame.time.Clock()

    trail = []
    running = True
    frame_idx = 0

    while running and frame_idx < len(trajectory):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        # Convert from task config pixels to display pixels and meters
        x_task, y_task, _ = trajectory[frame_idx]
        cursor_pos_m = (x_task * s2m_x, y_task * s2m_y)
        # Display pixels: convert via meters * SCALE
        cursor_pos_px = (int(cursor_pos_m[0] * SCALE), int(cursor_pos_m[1] * SCALE))
        trail.append(cursor_pos_px)

        render_menu_frame(
            screen,
            main_menu_rect,
            sub_menu_rect,
            main_menu_items,
            sub_menu_items,
            cursor_pos_m,
            cursor_pos_px,
            target_pos_m,
            TARGET_RADIUS_M,
            trail_px=trail,
            scale=SCALE,
        )

        pygame.display.flip()
        clock.tick(fps)
        frame_idx += 1

    pygame.quit()

    # Save plots
    draw_speed_profile(
        speeds,
        save_path=os.path.join(run_dir, "plot_speed.png"),
        title="Speed Profile - Cascading Menu",
    )

    # Detailed trajectory plot (matching original style)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Colors
    menu_bg_color = 'white'
    menu_border_color = '#C0C0C0'
    target_color = '#FF6B6B'
    shadow_color = 'black'
    shadow_alpha = 0.15
    shadow_offset = 0.003

    # Draw main menu shadow + background
    main_rect_dict = main_menu_rect
    shadow_rect = Rectangle(
        (main_rect_dict['x'] + shadow_offset, main_rect_dict['y'] + shadow_offset),
        main_rect_dict['width'], main_rect_dict['height'],
        facecolor=shadow_color, edgecolor='none', alpha=shadow_alpha, zorder=1
    )
    ax.add_patch(shadow_rect)

    main_bg = Rectangle(
        (main_rect_dict['x'], main_rect_dict['y']),
        main_rect_dict['width'], main_rect_dict['height'],
        facecolor=menu_bg_color, edgecolor=menu_border_color, linewidth=1.5, zorder=2
    )
    ax.add_patch(main_bg)

    # Draw main menu items
    for item in main_menu_items:
        item_color = target_color if item['is_target'] else menu_bg_color
        item_rect = Rectangle(
            (item['x'], item['y']), item['width'], item['height'],
            facecolor=item_color, edgecolor='none', alpha=0.9 if item['is_target'] else 1.0, zorder=3
        )
        ax.add_patch(item_rect)

    # Draw submenu shadow + background
    sub_rect_dict = sub_menu_rect
    shadow_rect = Rectangle(
        (sub_rect_dict['x'] + shadow_offset, sub_rect_dict['y'] + shadow_offset),
        sub_rect_dict['width'], sub_rect_dict['height'],
        facecolor=shadow_color, edgecolor='none', alpha=shadow_alpha, zorder=1
    )
    ax.add_patch(shadow_rect)

    sub_bg = Rectangle(
        (sub_rect_dict['x'], sub_rect_dict['y']),
        sub_rect_dict['width'], sub_rect_dict['height'],
        facecolor=menu_bg_color, edgecolor=menu_border_color, linewidth=1.5, zorder=2
    )
    ax.add_patch(sub_bg)

    # Draw submenu items
    for item in sub_menu_items:
        item_color = target_color if item['is_target'] else menu_bg_color
        item_rect = Rectangle(
            (item['x'], item['y']), item['width'], item['height'],
            facecolor=item_color, edgecolor='none', alpha=0.9 if item['is_target'] else 1.0, zorder=3
        )
        ax.add_patch(item_rect)

    # Draw constrained region (L-shaped polygon) as blue semi-transparent overlay
    from matplotlib.patches import Polygon as MplPolygon
    mx, my = main_rect_dict['x'], main_rect_dict['y']
    mw, mh = main_rect_dict['width'], main_rect_dict['height']
    sx, sy = sub_rect_dict['x'], sub_rect_dict['y']
    sw, sh = sub_rect_dict['width'], sub_rect_dict['height']

    # Build L-shaped polygon vertices (clockwise)
    top_main, bot_main = my, my + mh
    top_sub, bot_sub = sy, sy + sh

    constraint_vertices = [
        (mx, top_main),           # top-left of main
        (sx, top_main),           # top-right of main (junction)
    ]
    if top_sub > top_main:
        constraint_vertices.append((sx, top_sub))
    constraint_vertices.append((sx + sw, top_sub if top_sub > top_main else top_main))
    constraint_vertices.append((sx + sw, bot_sub))  # bottom-right of sub
    constraint_vertices.append((sx, bot_sub))       # bottom-left of sub
    if bot_sub != bot_main:
        constraint_vertices.append((sx, bot_main))
    constraint_vertices.append((mx, bot_main))      # bottom-left of main

    constraint_poly = MplPolygon(
        constraint_vertices,
        facecolor='blue', edgecolor='blue', alpha=0.15,
        linewidth=2, linestyle='--', zorder=4,
        label='Constraint Region'
    )
    ax.add_patch(constraint_poly)

    # Draw reference path (convert from pixels/1000 to meters)
    if reference_path is not None:
        try:
            SCREEN_H_PX = environment["screen_height"]
            ref_to_m_x = window_width_m / (SCREEN_W_PX / 1000)
            ref_to_m_y = window_height_m / (SCREEN_H_PX / 1000)
            theta_samples = np.linspace(0, reference_path.total_length, 100)
            ref_xs, ref_ys = [], []
            for theta in theta_samples:
                try:
                    ref_pos = reference_path(theta)
                    ref_xs.append(ref_pos[0] * ref_to_m_x)
                    ref_ys.append(ref_pos[1] * ref_to_m_y)
                except:
                    continue
            if ref_xs and ref_ys:
                ax.plot(ref_xs, ref_ys, 'g--', linewidth=1.5, alpha=0.6,
                       label='Reference Path', zorder=9, dashes=(5, 5))
        except Exception as e:
            print(f"Warning: Could not plot reference path: {e}")

    # Draw target position and arrival radius
    target_x, target_y = target_pos_m
    target_r = TARGET_RADIUS_M
    target_circle = plt.Circle((target_x, target_y), target_r,
                               color='red', fill=False, linewidth=1.5,
                               linestyle='--', zorder=12, label=f'Target (r={target_r*1000:.0f}mm)')
    ax.add_patch(target_circle)
    ax.plot(target_x, target_y, 'rx', markersize=10, markeredgewidth=2, zorder=12)

    # Annotate distance from cursor end to target
    end_x, end_y = cursor_xs_m[-1], cursor_ys_m[-1]
    dist_to_target = np.sqrt((end_x - target_x)**2 + (end_y - target_y)**2)
    reached = dist_to_target < target_r
    status = "REACHED" if reached else f"{dist_to_target*1000:.1f}mm away"
    ax.annotate(status,
                xy=(end_x, end_y), xytext=(15, -15),
                textcoords='offset points', fontsize=9,
                color='green' if reached else 'red',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                zorder=13)

    # Draw trajectory
    ax.plot(cursor_xs_m, cursor_ys_m, 'b-', linewidth=1.2, label='Trajectory', zorder=10)
    ax.plot(cursor_xs_m[0], cursor_ys_m[0], 'go', markersize=10, label='Start', zorder=11,
            markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax.plot(cursor_xs_m[-1], cursor_ys_m[-1], 'ro', markersize=12, label='End', zorder=11,
            markeredgecolor='darkred', markeredgewidth=1.5)

    ax.set_xlim(0, window_width_m)
    ax.set_ylim(0, window_height_m)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'Cascading Menu Trajectory ({len(trajectory)} steps, {len(trajectory)*interval:.1f}s)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plot_trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return trajectory, speeds


BUNDLED_PERSONAS_DIR = Path(__file__).parent.parent / "hcs_package" / "src" / "hcs_package" / "user_configurations"


def validate_user_config(user_config_path: Path) -> None:
    """Warn if the user config is missing fields required by the simulator.

    The hcs_package simulator now requires a ``speed_model`` block. Configs
    written before the speed-model refactor (e.g., experiment/user_configurations/
    customized.json) will crash deep inside the trajectory generator; surface
    the problem up-front with an actionable hint.
    """
    try:
        with open(user_config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: Could not parse user config {user_config_path}: {e}")
        sys.exit(1)

    if 'speed_model' not in cfg:
        bundled = sorted(p.stem for p in BUNDLED_PERSONAS_DIR.glob("*.json"))
        print(
            f"Warning: user config {user_config_path} has no 'speed_model' block.\n"
            f"  The simulator requires one and will raise ValueError mid-run.\n"
            f"  Fix: add\n"
            f'    "speed_model": {{"type": "gam", "path": "<path-to-gam.pkl>"}}\n'
            f"  to the config (paths resolve relative to the config file), OR\n"
            f"  pass --user-config with a bundled persona, e.g.\n"
            f"    --user-config {BUNDLED_PERSONAS_DIR / 'office_worker.json'}\n"
            f"  Available bundled personas: {bundled}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Run pygame experiment with specified environment configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/sigmoidal_tunnel.json
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/corner_tunnel.json
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/lasso_selection.json
    python -m experiment.pygame_experiment --env-config experiment/environment_configurations/cascading_menu.json
        """
    )
    parser.add_argument('--env-config', '--env_config', type=str, required=True,
                        help='Path to environment configuration JSON')
    parser.add_argument('--user-config', '--user_config', type=str,
                        default=str(DEFAULT_USER_CONFIG),
                        help='Path to user configuration JSON')
    parser.add_argument('--save-path', '--save_path', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--max-steps', '--max_steps', type=int, default=None,
                        help='Override max steps from config')
    parser.add_argument('--fps', type=int, default=20,
                        help='Frames per second for pygame replay')
    args = parser.parse_args()

    # Load environment config
    env_config_path = Path(args.env_config)
    if not env_config_path.exists():
        print(f"Error: Environment config not found: {env_config_path}")
        sys.exit(1)

    user_config_path = Path(args.user_config)
    if not user_config_path.exists():
        print(f"Error: User config not found: {user_config_path}")
        sys.exit(1)

    validate_user_config(user_config_path)

    print(f"Using environment config: {env_config_path}")
    print(f"Using user config: {user_config_path}")

    # Create environment
    env_config = load_environment_config(str(env_config_path))
    environment = create_environment(env_config)

    if args.max_steps is not None:
        environment["max_steps"] = args.max_steps

    # Generate task config for hcs_package
    task_config = generate_task_config(environment)

    # Setup save directory
    exp_name = get_experiment_name(environment.get("env_type", "experiment"))
    if args.save_path:
        run_dir = args.save_path
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        exp_dir = f"./results/{date_str}_{exp_name}_experiment"
        os.makedirs(exp_dir, exist_ok=True)
        run_dir = os.path.join(exp_dir, f"run_{datetime.now().strftime('%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving results to: {run_dir}")

    # Save configs
    with open(os.path.join(run_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=2)
    with open(os.path.join(run_dir, "task_config.json"), "w") as f:
        json.dump(task_config, f, indent=2)
    with open(os.path.join(run_dir, "user_config.json"), "w") as f:
        with open(user_config_path) as uf:
            json.dump(json.load(uf), f, indent=2)

    # Run appropriate experiment based on env_type
    env_type = environment.get("env_type", "")

    if "tunnel" in env_type or "corner" in env_type or "turning" in env_type:
        trajectory, speeds = run_tunnel_experiment(
            environment, task_config, str(user_config_path), run_dir, args.fps
        )
    elif "lasso" in env_type:
        trajectory, speeds = run_lasso_experiment(
            environment, task_config, str(user_config_path), run_dir, args.fps
        )
    elif "menu" in env_type:
        trajectory, speeds = run_menu_experiment(
            environment, task_config, str(user_config_path), run_dir, args.fps
        )
    else:
        print(f"Unknown environment type: {env_type}")
        sys.exit(1)

    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
