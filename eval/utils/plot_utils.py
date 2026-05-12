"""
Plotting utilities for steering experiment results.

This module contains all plotting functions for visualizing experiment results,
including trajectories, speeds, progress, and density heatmaps.
"""

import os
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import seaborn as sns
from scipy.interpolate import interp1d
import pandas as pd

# Suppress matplotlib debug output
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
import sys
project_root = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, project_root)

WINDOW_WIDTH = 0.46
WINDOW_HEIGHT = 0.26

from experiment.utils import generateTunnelBoundaries
from matplotlib.patches import Rectangle, Circle, Polygon


def resample_to_common_time(speeds_list, time_step=0.05, max_time=None):
    """
    Resample multiple speed profiles to a common time grid.
    
    Args:
        speeds_list: List of speed arrays (may have different lengths)
        time_step: Time step in seconds
        max_time: Maximum time (if None, uses longest trajectory)
    
    Returns:
        tuple: (common_time_grid, resampled_speeds_list)
    """
    if not speeds_list:
        return np.array([]), []
    
    # Find maximum length
    max_len = max(len(speeds) for speeds in speeds_list)
    if max_time is None:
        max_time = max_len * time_step
    
    # Create common time grid
    common_time = np.arange(0, max_time, time_step)
    
    # Resample each speed profile
    resampled_list = []
    for speeds in speeds_list:
        original_time = np.arange(0, len(speeds) * time_step, time_step)
        if len(original_time) > len(speeds):
            original_time = original_time[:len(speeds)]
        
        if len(speeds) == 0:
            resampled = np.zeros_like(common_time)
        elif len(speeds) == 1:
            resampled = np.full_like(common_time, speeds[0])
        else:
            # Interpolate to common time grid
            f = interp1d(original_time, speeds, kind='linear', 
                        bounds_error=False, fill_value=(speeds[0], speeds[-1]))
            resampled = f(common_time)
        
        resampled_list.append(resampled)
    
    return common_time, resampled_list


def compute_progress_along_path(trajectory, tunnel_path):
    """
    Compute progress (0 to 1) along tunnel path for each point in trajectory.
    
    Args:
        trajectory: List of (x, y) points
        tunnel_path: List of (x, y) points representing tunnel centerline
    
    Returns:
        List of progress values (0.0 to 1.0) for each trajectory point
    """
    if not trajectory or not tunnel_path:
        return [0.0] * len(trajectory)
    
    path = np.asarray(tunnel_path, dtype=float)
    traj = np.asarray(trajectory, dtype=float)
    
    # Compute cumulative arc length along tunnel path
    path_lengths = [0.0]
    cumulative_length = 0.0
    for i in range(len(path) - 1):
        seg_len = np.linalg.norm(path[i + 1] - path[i])
        cumulative_length += seg_len
        path_lengths.append(cumulative_length)
    
    total_path_length = path_lengths[-1]
    if total_path_length < 1e-6:
        return [0.0] * len(trajectory)
    
    # For each trajectory point, find its projection onto the path and compute progress
    progress_values = []
    for point in traj:
        # Find nearest point on path
        best_dist2 = float('inf')
        best_i = 0
        best_t = 0.0
        
        for i in range(len(path) - 1):
            p0 = path[i]
            p1 = path[i + 1]
            seg = p1 - p0
            seg_len2 = float(np.dot(seg, seg))
            
            if seg_len2 <= 1e-18:
                cand = p0
                dist2 = float(np.dot(point - cand, point - cand))
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_i = i
                    best_t = 0.0
            else:
                t = float(np.clip(np.dot(point - p0, seg) / seg_len2, 0.0, 1.0))
                cand = p0 + t * seg
                dist2 = float(np.dot(point - cand, point - cand))
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_i = i
                    best_t = t
        
        # Compute arc length to this point
        arc_length = path_lengths[best_i]
        if best_i < len(path) - 1:
            seg_len = np.linalg.norm(path[best_i + 1] - path[best_i])
            arc_length += best_t * seg_len
        
        # Normalize to progress (0 to 1)
        progress = arc_length / total_path_length
        progress_values.append(np.clip(progress, 0.0, 1.0))
    
    return progress_values


def plot_enhanced_speed_profiles(all_results, output_path_base, time_step=0.05):
    """
    Plot speed profiles with mean and shaded error bands using seaborn.
    
    Args:
        all_results: dict {trial_id: {segment_id: {human: {...}, sim: {...}}}}
        output_path_base: Folder path for saving plots
        time_step: Time step in seconds (default: 0.05)
    """
    trial_ids = sorted(all_results.keys())
    n_trials = len(trial_ids)
    
    if n_trials == 0:
        return
    
    first_trial = all_results[trial_ids[0]]
    n_segments = len(first_trial)
    
    fig = plt.figure(figsize=(6 * n_segments, 4 * n_trials))
    gs = GridSpec(n_trials, n_segments, figure=fig, hspace=0.3, wspace=0.3)
    
    for trial_idx, trial_id in enumerate(trial_ids):
        trial_results = all_results[trial_id]
        
        for segment_id in sorted(trial_results.keys()):
            segment_results = trial_results[segment_id]
            ax = fig.add_subplot(gs[trial_idx, segment_id])
            
            # Prepare data for seaborn
            all_speed_data = []
            
            # Human speeds
            if 'human' in segment_results:
                human_speeds_list = segment_results['human']['speeds']
                common_time, resampled_human = resample_to_common_time(human_speeds_list, time_step)
                
                for speeds in resampled_human:
                    for t, s in zip(common_time, speeds):
                        all_speed_data.append({'Time (s)': t, 'Speed (m/s)': s, 'Type': 'Human'})
            
            # Simulator speeds
            if 'sim' in segment_results:
                sim_speeds_list = segment_results['sim']['speeds']
                common_time, resampled_sim = resample_to_common_time(sim_speeds_list, time_step)

                for speeds in resampled_sim:
                    for t, s in zip(common_time, speeds):
                        all_speed_data.append({'Time (s)': t, 'Speed (m/s)': s, 'Type': 'Simulator'})

            # Baseline speeds
            if 'baseline' in segment_results:
                baseline_speeds_list = segment_results['baseline']['speeds']
                common_time, resampled_bl = resample_to_common_time(baseline_speeds_list, time_step)

                for speeds in resampled_bl:
                    for t, s in zip(common_time, speeds):
                        all_speed_data.append({'Time (s)': t, 'Speed (m/s)': s, 'Type': 'Baseline'})

            if all_speed_data:
                df = pd.DataFrame(all_speed_data)
                palette = {'Human': 'tab:blue', 'Simulator': 'tab:orange', 'Baseline': 'tab:green'}

                # Use seaborn lineplot with error bands
                sns.lineplot(data=df, x='Time (s)', y='Speed (m/s)', hue='Type',
                           ax=ax, errorbar='sd', alpha=0.8, palette=palette)
            
            ax.grid(False)
    
    # Create filename with trial IDs
    trial_ids_str = '_'.join([f't{tid}' for tid in trial_ids])
    speed_path = Path(output_path_base) / f'speeds_enhanced_{trial_ids_str}.png'
    fig.savefig(speed_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Enhanced speed plot saved to: {speed_path}")


def plot_speeds_vs_progress(all_results, output_path_base, tunnel_paths, bin_size=0.1):
    """
    Plot speeds vs progress (0 to 1) with discrete bins.
    
    Args:
        all_results: dict {trial_id: {segment_id: {human: {...}, sim: {...}}}}
        output_path_base: Folder path for saving plots
        tunnel_paths: dict {trial_id: tunnel_path}
        bin_size: Size of progress bins (default: 0.1)
    """
    trial_ids = sorted(all_results.keys())
    n_trials = len(trial_ids)
    
    if n_trials == 0:
        return
    
    first_trial = all_results[trial_ids[0]]
    n_segments = len(first_trial)
    
    # Progress bin size
    progress_bins = np.arange(0.0, 1.0 + bin_size, bin_size)
    progress_centers = (progress_bins[:-1] + progress_bins[1:]) / 2.0
    
    fig = plt.figure(figsize=(6 * n_segments, 4 * n_trials))
    gs = GridSpec(n_trials, n_segments, figure=fig, hspace=0.3, wspace=0.3)
    
    for trial_idx, trial_id in enumerate(trial_ids):
        trial_results = all_results[trial_id]
        tunnel_path = tunnel_paths.get(trial_id, [])
        
        for segment_id in sorted(trial_results.keys()):
            segment_results = trial_results[segment_id]
            ax = fig.add_subplot(gs[trial_idx, segment_id])
            
            # Process human trajectories - plot each one individually
            if 'human' in segment_results and tunnel_path:
                human_trajs = segment_results['human']['trajectories']
                human_speeds_list = segment_results['human']['speeds']
                
                for traj_idx, (traj, speeds) in enumerate(zip(human_trajs, human_speeds_list)):
                    if len(traj) != len(speeds) or len(traj) == 0:
                        continue
                    
                    # Compute progress for each point
                    progress_values = compute_progress_along_path(traj, tunnel_path)
                    
                    # Bin speeds by progress for this trajectory
                    binned_speeds = [[] for _ in range(len(progress_centers))]
                    for progress, speed in zip(progress_values, speeds):
                        bin_idx = int(np.clip(progress / bin_size, 0, len(progress_centers) - 1))
                        binned_speeds[bin_idx].append(speed)
                    
                    # Compute mean speed for each bin (for this trajectory)
                    mean_speeds = []
                    for bin_idx in range(len(progress_centers)):
                        if binned_speeds[bin_idx]:
                            mean_speeds.append(np.mean(binned_speeds[bin_idx]))
                        else:
                            mean_speeds.append(np.nan)
                    
                    # Plot line for this human trajectory
                    ax.plot(progress_centers, mean_speeds, 'b-', linewidth=1.5, 
                           marker='o', markersize=3, alpha=0.7)
            
            # Process simulator trajectories - plot each run individually
            if 'sim' in segment_results and tunnel_path:
                sim_trajs = segment_results['sim']['trajectories']
                sim_speeds_list = segment_results['sim']['speeds']
                
                for run_idx, (traj, speeds) in enumerate(zip(sim_trajs, sim_speeds_list)):
                    if len(traj) != len(speeds) or len(traj) == 0:
                        continue
                    
                    # Compute progress for each point
                    progress_values = compute_progress_along_path(traj, tunnel_path)
                    
                    # Bin speeds by progress for this run
                    binned_speeds = [[] for _ in range(len(progress_centers))]
                    for progress, speed in zip(progress_values, speeds):
                        bin_idx = int(np.clip(progress / bin_size, 0, len(progress_centers) - 1))
                        binned_speeds[bin_idx].append(speed)
                    
                    # Compute mean speed for each bin (for this run)
                    mean_speeds = []
                    for bin_idx in range(len(progress_centers)):
                        if binned_speeds[bin_idx]:
                            mean_speeds.append(np.mean(binned_speeds[bin_idx]))
                        else:
                            mean_speeds.append(np.nan)
                    
                    # Plot line for this simulator run
                    ax.plot(progress_centers, mean_speeds, 'orange', linewidth=1.5, 
                           marker='s', markersize=3, alpha=0.7)
            
            # Convert progress to percentage (0-100%) - data is 0-1, labels show 0-100%
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_ylabel('Speed (m/s)', fontsize=10)
            ax.grid(False)
            ax.set_xlim(0, 1)
    
    # Create filename with trial IDs
    trial_ids_str = '_'.join([f't{tid}' for tid in trial_ids])
    progress_path = Path(output_path_base) / f'speeds_progress_{trial_ids_str}.png'
    fig.savefig(progress_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Speed vs progress plot saved to: {progress_path}")


def plot_speeds_vs_progress_enhanced(all_results, output_path_base, tunnel_paths, bin_size=0.1):
    """
    Plot speeds vs progress (0 to 1) with mean and standard deviation error bands.
    Uses seaborn for enhanced visualization with error bands.
    
    Args:
        all_results: dict {trial_id: {segment_id: {human: {...}, sim: {...}}}}
        output_path_base: Folder path for saving plots
        tunnel_paths: dict {trial_id: tunnel_path}
        bin_size: Size of progress bins (default: 0.1)
    """
    trial_ids = sorted(all_results.keys())
    n_trials = len(trial_ids)
    
    if n_trials == 0:
        return
    
    first_trial = all_results[trial_ids[0]]
    n_segments = len(first_trial)
    
    # Progress bins
    progress_bins = np.arange(0.0, 1.0 + bin_size, bin_size)
    progress_centers = (progress_bins[:-1] + progress_bins[1:]) / 2.0
    
    fig = plt.figure(figsize=(6 * n_segments, 4 * n_trials))
    gs = GridSpec(n_trials, n_segments, figure=fig, hspace=0.3, wspace=0.3)
    
    for trial_idx, trial_id in enumerate(trial_ids):
        trial_results = all_results[trial_id]
        tunnel_path = tunnel_paths.get(trial_id, [])
        
        for segment_id in sorted(trial_results.keys()):
            segment_results = trial_results[segment_id]
            ax = fig.add_subplot(gs[trial_idx, segment_id])
            
            # Prepare data for seaborn
            all_speed_data = []
            
            # Process human trajectories
            if 'human' in segment_results and tunnel_path:
                human_trajs = segment_results['human']['trajectories']
                human_speeds_list = segment_results['human']['speeds']
                
                for traj, speeds in zip(human_trajs, human_speeds_list):
                    if len(traj) != len(speeds) or len(traj) == 0:
                        continue
                    
                    # Compute progress for each point
                    progress_values = compute_progress_along_path(traj, tunnel_path)
                    
                    # Bin speeds by progress for this trajectory
                    binned_speeds = [[] for _ in range(len(progress_centers))]
                    for progress, speed in zip(progress_values, speeds):
                        bin_idx = int(np.clip(progress / bin_size, 0, len(progress_centers) - 1))
                        binned_speeds[bin_idx].append(speed)
                    
                    # Add mean speed for each bin to dataframe
                    for bin_idx, progress_center in enumerate(progress_centers):
                        if binned_speeds[bin_idx]:
                            mean_speed = np.mean(binned_speeds[bin_idx])
                            all_speed_data.append({
                                'Progress': progress_center,
                                'Speed (m/s)': mean_speed,
                                'Type': 'Human'
                            })
            
            # Process simulator trajectories
            if 'sim' in segment_results and tunnel_path:
                sim_trajs = segment_results['sim']['trajectories']
                sim_speeds_list = segment_results['sim']['speeds']

                for traj, speeds in zip(sim_trajs, sim_speeds_list):
                    if len(traj) != len(speeds) or len(traj) == 0:
                        continue

                    # Compute progress for each point
                    progress_values = compute_progress_along_path(traj, tunnel_path)

                    # Bin speeds by progress for this run
                    binned_speeds = [[] for _ in range(len(progress_centers))]
                    for progress, speed in zip(progress_values, speeds):
                        bin_idx = int(np.clip(progress / bin_size, 0, len(progress_centers) - 1))
                        binned_speeds[bin_idx].append(speed)

                    # Add mean speed for each bin to dataframe
                    for bin_idx, progress_center in enumerate(progress_centers):
                        if binned_speeds[bin_idx]:
                            mean_speed = np.mean(binned_speeds[bin_idx])
                            all_speed_data.append({
                                'Progress': progress_center,
                                'Speed (m/s)': mean_speed,
                                'Type': 'Simulator'
                            })

            # Process baseline trajectories
            if 'baseline' in segment_results and tunnel_path:
                bl_trajs = segment_results['baseline']['trajectories']
                bl_speeds_list = segment_results['baseline']['speeds']

                for traj, speeds in zip(bl_trajs, bl_speeds_list):
                    if len(traj) != len(speeds) or len(traj) == 0:
                        continue

                    progress_values = compute_progress_along_path(traj, tunnel_path)

                    binned_speeds = [[] for _ in range(len(progress_centers))]
                    for progress, speed in zip(progress_values, speeds):
                        bin_idx = int(np.clip(progress / bin_size, 0, len(progress_centers) - 1))
                        binned_speeds[bin_idx].append(speed)

                    for bin_idx, progress_center in enumerate(progress_centers):
                        if binned_speeds[bin_idx]:
                            mean_speed = np.mean(binned_speeds[bin_idx])
                            all_speed_data.append({
                                'Progress': progress_center,
                                'Speed (m/s)': mean_speed,
                                'Type': 'Baseline'
                            })

            # Plot using seaborn with error bands
            if all_speed_data:
                df = pd.DataFrame(all_speed_data)
                palette = {'Human': 'tab:blue', 'Simulator': 'tab:orange', 'Baseline': 'tab:green'}

                # Use seaborn lineplot with error bands (without legend)
                sns.lineplot(data=df, x='Progress', y='Speed (m/s)', hue='Type',
                           ax=ax, errorbar='sd', alpha=0.8, legend=False, palette=palette)
            
            # Show only the 10%-90% range used for speed metrics
            ax.set_xticks([0.1, 0.25, 0.5, 0.75, 0.9])
            ax.set_xticklabels(['10%', '25%', '50%', '75%', '90%'])
            ax.set_xlabel('Progress', fontsize=10)
            ax.set_ylabel('Speed (m/s)', fontsize=10)
            ax.grid(False)
            ax.set_xlim(0.1, 0.9)
    
    # Create filename with trial IDs
    trial_ids_str = '_'.join([f't{tid}' for tid in trial_ids])
    progress_path = Path(output_path_base) / f'speeds_progress_enhanced_{trial_ids_str}.png'
    fig.savefig(progress_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Enhanced speed vs progress plot saved to: {progress_path}")


def plot_trajectory_density(all_results, output_path_base, tunnel_paths, tunnel_widths, trial_metadata=None):
    """
    Plot 2D density heatmaps for trajectories with standardized color scales.
    
    Args:
        all_results: dict {trial_id: {segment_id: {human: {...}, sim: {...}}}}
        output_path_base: Folder path for saving plots
        tunnel_paths: dict {trial_id: tunnel_path}
        tunnel_widths: dict {trial_id: tunnel_width}
    """
    trial_ids = sorted(all_results.keys())
    n_trials = len(trial_ids)
    
    if n_trials == 0:
        return
    
    first_trial = all_results[trial_ids[0]]
    n_segments = len(first_trial)
    
    # First pass: collect all points and compute global maximum density
    gridsize = 50  # Increased for better resolution
    global_max_density = 0.0
    
    # Temporary figure to compute densities
    temp_fig, temp_ax = plt.subplots(1, 1, figsize=(1, 1))
    
    for trial_id in trial_ids:
        trial_results = all_results[trial_id]
        for segment_id in sorted(trial_results.keys()):
            segment_results = trial_results[segment_id]
            
            # Check human data
            if 'human' in segment_results:
                human_trajs = segment_results['human']['trajectories']
                if human_trajs:
                    all_points = []
                    for traj in human_trajs:
                        if len(traj) > 0:
                            all_points.extend(traj)
                    
                    if all_points:
                        points_array = np.array(all_points)
                        # Compute hexbin to get density values
                        hb = temp_ax.hexbin(points_array[:, 0], points_array[:, 1], 
                                          gridsize=gridsize, extent=(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT))
                        counts = hb.get_array()
                        if counts is not None and len(counts) > 0:
                            max_count = np.max(counts)
                            global_max_density = max(global_max_density, max_count)
                        temp_ax.clear()
            
            # Check simulator data
            if 'sim' in segment_results:
                sim_trajs = segment_results['sim']['trajectories']
                if sim_trajs:
                    all_points = []
                    for traj in sim_trajs:
                        if len(traj) > 0:
                            all_points.extend(traj)
                    
                    if all_points:
                        points_array = np.array(all_points)
                        # Compute hexbin to get density values
                        hb = temp_ax.hexbin(points_array[:, 0], points_array[:, 1], 
                                          gridsize=gridsize, extent=(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT))
                        counts = hb.get_array()
                        if counts is not None and len(counts) > 0:
                            max_count = np.max(counts)
                            global_max_density = max(global_max_density, max_count)
                        temp_ax.clear()
    
    plt.close(temp_fig)
    
    # If no data found, set a default
    if global_max_density == 0.0:
        global_max_density = 1.0
    
    # Second pass: create actual plots with standardized scales
    fig = plt.figure(figsize=(6 * n_segments, 4 * n_trials * 2))  # *2 for human and sim side by side
    gs = GridSpec(n_trials * 2, n_segments, figure=fig, hspace=0.3, wspace=0.3)
    
    for trial_idx, trial_id in enumerate(trial_ids):
        trial_results = all_results[trial_id]
        tunnel_path = tunnel_paths.get(trial_id, [])
        tunnel_width = tunnel_widths.get(trial_id, 0.02)
        
        for segment_id in sorted(trial_results.keys()):
            segment_results = trial_results[segment_id]
            
            # Human density plot
            ax_human = fig.add_subplot(gs[trial_idx * 2, segment_id])
            
            # Get task type from trial metadata
            task_type = None
            trial_condition = None
            if trial_metadata and trial_id in trial_metadata:
                # Get condition from first participant and first round
                for participant_id in trial_metadata[trial_id].keys():
                    for round_num in trial_metadata[trial_id][participant_id].keys():
                        trial_condition = trial_metadata[trial_id][participant_id][round_num].get('condition', {})
                        if trial_condition:
                            task_type = trial_condition.get('tunnelType', 'wave')
                            break
                    if trial_condition:
                        break
            
            # Calculate bounds first (needed for hexbin extent and axis limits)
            hexbin_extent = (0, WINDOW_WIDTH, 0, WINDOW_HEIGHT)  # Default extent
            if task_type == 'lasso' and trial_condition:
                # Calculate grid bounds with padding
                grid_layout = trial_condition.get('grid_layout') or trial_condition.get('gridLayout', [])
                icon_spacing = trial_condition.get('icon_spacing') or trial_condition.get('iconSpacing', 0.05)
                grid_origin = trial_condition.get('grid_origin') or trial_condition.get('gridOrigin', [0.1, 0.1])
                icon_radius = trial_condition.get('icon_radius') or trial_condition.get('iconRadius', 0.01)
                
                if grid_layout:
                    origin_x, origin_y = grid_origin
                    rows = len(grid_layout)
                    max_cols = max(len(row.split()) if ' ' in row else len(row) for row in grid_layout) if grid_layout else 0
                    
                    # Calculate grid bounds (tight to grid groups, no padding)
                    grid_min_x = origin_x - icon_radius
                    grid_max_x = origin_x + icon_radius
                    grid_min_y = origin_y - icon_radius
                    grid_max_y = origin_y + icon_radius
                    
                    # Use grid bounds directly (no padding)
                    x_min = grid_min_x
                    x_max = grid_max_x
                    y_min = grid_min_y
                    y_max = grid_max_y
                    
                    hexbin_extent = (x_min, x_max, y_min, y_max)
            elif task_type == 'cascading_menu' and trial_condition:
                # Calculate menu bounds (include entire menu, submenu, and shadows)
                main_menu_window_size = trial_condition.get('mainMenuWindowSize') or trial_condition.get('main_menu_window_size', [0.12, 0.12])
                sub_menu_window_size = trial_condition.get('subMenuWindowSize') or trial_condition.get('sub_menu_window_size', [0.1, 0.12])
                main_menu_origin = trial_condition.get('mainMenuOrigin') or trial_condition.get('main_menu_origin', [0.1, 0.02])
                shadow_offset = 0.003  # Shadow offset from _draw_cascading_menu
                
                main_menu_x, main_menu_y = main_menu_origin
                main_menu_w, main_menu_h = main_menu_window_size
                sub_menu_w, sub_menu_h = sub_menu_window_size
                
                # Calculate submenu position
                target_main_idx = trial_condition.get('targetMainMenuIndex') or trial_condition.get('target_main_menu_index', 11)
                main_menu_size = trial_condition.get('mainMenuSize') or trial_condition.get('main_menu_size', 12)
                main_item_height = main_menu_h / main_menu_size
                target_main_item_top = main_menu_y + target_main_idx * main_item_height
                sub_menu_x = main_menu_x + main_menu_w
                sub_menu_y = target_main_item_top
                
                # Calculate bounds including shadows
                # Main menu shadow extends to: main_menu_x + shadow_offset, main_menu_y + shadow_offset
                # Submenu shadow extends to: sub_menu_x + shadow_offset, sub_menu_y + shadow_offset
                x_min = main_menu_x + shadow_offset - 0.005  # Include shadow and small padding
                x_max = sub_menu_x + sub_menu_w + shadow_offset + 0.005  # Include submenu and shadow
                y_min = min(main_menu_y + shadow_offset, sub_menu_y + shadow_offset) - 0.005
                y_max = max(main_menu_y + main_menu_h + shadow_offset, sub_menu_y + sub_menu_h + shadow_offset) + 0.005
                
                hexbin_extent = (x_min, x_max, y_min, y_max)
            
            # Draw task-specific visualizations (before density)
            if task_type == 'lasso' and trial_condition:
                _draw_lasso_grid(ax_human, trial_condition, zorder_base=1)
            elif task_type == 'cascading_menu' and trial_condition:
                _draw_cascading_menu(ax_human, trial_condition, zorder_base=1)
            
            # Draw tunnel fill and boundaries first (before density hexbin)
            # zorder=1: boundaries (lowest)
            # zorder=2: fill (above boundaries, below hexbin)
            if task_type != 'lasso' and task_type != 'cascading_menu':
                if tunnel_path is not None:
                    path = np.asarray(tunnel_path, dtype=float)
                    if path.ndim == 2 and path.shape[0] >= 2 and path.shape[1] == 2:
                        left_boundary, right_boundary = generateTunnelBoundaries(tunnel_path, tunnel_width)
                        # Fill polygon: left forward, right reversed
                        poly_x = np.concatenate([left_boundary[:, 0], right_boundary[::-1, 0]])
                        poly_y = np.concatenate([left_boundary[:, 1], right_boundary[::-1, 1]])
                        ax_human.fill(poly_x, poly_y, color='lightgray', alpha=1.0, zorder=2)
                        # Boundary lines
                        ax_human.plot(left_boundary[:, 0], left_boundary[:, 1], 
                                    color='gray', linestyle='--', linewidth=0.8, zorder=1)
                        ax_human.plot(right_boundary[:, 0], right_boundary[:, 1], 
                                    color='gray', linestyle='--', linewidth=0.8, zorder=1)
            
            # Draw density hexbin (will appear on top due to later drawing order and default zorder ~3)
            hb_human = None
            if 'human' in segment_results:
                human_trajs = segment_results['human']['trajectories']
                if human_trajs:
                    all_points = []
                    for traj in human_trajs:
                        if len(traj) > 0:
                            all_points.extend(traj)
                    
                    if all_points:
                        points_array = np.array(all_points)
                        hb_human = ax_human.hexbin(points_array[:, 0], points_array[:, 1], 
                                                  gridsize=gridsize, cmap='Blues', 
                                                  vmin=0, vmax=global_max_density,
                                                  extent=hexbin_extent,
                                                  mincnt=1, linewidths=0.1)
            
            # Set limits based on task type
            if task_type == 'lasso' and trial_condition:
                # Calculate grid bounds with padding
                grid_layout = trial_condition.get('grid_layout') or trial_condition.get('gridLayout', [])
                icon_spacing = trial_condition.get('icon_spacing') or trial_condition.get('iconSpacing', 0.05)
                grid_origin = trial_condition.get('grid_origin') or trial_condition.get('gridOrigin', [0.1, 0.1])
                icon_radius = trial_condition.get('icon_radius') or trial_condition.get('iconRadius', 0.01)
                
                if grid_layout:
                    origin_x, origin_y = grid_origin
                    rows = len(grid_layout)
                    max_cols = max(len(row.split()) if ' ' in row else len(row) for row in grid_layout) if grid_layout else 0
                    
                    # Calculate grid bounds (tight to grid groups, no padding)
                    # Squares are drawn at origin_x + col_idx * icon_spacing where col_idx = 0 to max_cols-1
                    # So the right edge is at origin_x + (max_cols - 1) * icon_spacing + icon_radius
                    grid_min_x = origin_x - icon_radius
                    grid_max_x = origin_x + (max_cols - 1) * icon_spacing + icon_radius
                    grid_min_y = origin_y - icon_radius
                    grid_max_y = origin_y + (rows - 1) * icon_spacing + icon_radius
                    
                    # Use grid bounds directly (no padding)
                    x_min = grid_min_x
                    x_max = grid_max_x
                    y_min = grid_min_y
                    y_max = grid_max_y
                    
                    ax_human.set_xlim(x_min, x_max)
                    ax_human.set_ylim(y_min, y_max)
                else:
                    ax_human.set_xlim(0, WINDOW_WIDTH)
                    ax_human.set_ylim(0, WINDOW_HEIGHT)
                
                # Remove axes frames
                ax_human.spines['top'].set_visible(False)
                ax_human.spines['right'].set_visible(False)
                ax_human.spines['bottom'].set_visible(False)
                ax_human.spines['left'].set_visible(False)
            elif task_type == 'cascading_menu' and trial_condition:
                # Calculate menu bounds (include entire menu, submenu, and shadows)
                main_menu_window_size = trial_condition.get('mainMenuWindowSize') or trial_condition.get('main_menu_window_size', [0.12, 0.12])
                sub_menu_window_size = trial_condition.get('subMenuWindowSize') or trial_condition.get('sub_menu_window_size', [0.1, 0.12])
                main_menu_origin = trial_condition.get('mainMenuOrigin') or trial_condition.get('main_menu_origin', [0.1, 0.02])
                shadow_offset = 0.003  # Shadow offset from _draw_cascading_menu
                
                main_menu_x, main_menu_y = main_menu_origin
                main_menu_w, main_menu_h = main_menu_window_size
                sub_menu_w, sub_menu_h = sub_menu_window_size
                
                # Calculate submenu position
                target_main_idx = trial_condition.get('targetMainMenuIndex') or trial_condition.get('target_main_menu_index', 11)
                main_menu_size = trial_condition.get('mainMenuSize') or trial_condition.get('main_menu_size', 12)
                main_item_height = main_menu_h / main_menu_size
                target_main_item_top = main_menu_y + target_main_idx * main_item_height
                sub_menu_x = main_menu_x + main_menu_w
                sub_menu_y = target_main_item_top
                
                # Calculate bounds including shadows
                # Main menu shadow extends to: main_menu_x + shadow_offset, main_menu_y + shadow_offset
                # Submenu shadow extends to: sub_menu_x + shadow_offset, sub_menu_y + shadow_offset
                x_min = main_menu_x + shadow_offset - 0.005  # Include shadow and small padding
                x_max = sub_menu_x + sub_menu_w + shadow_offset + 0.005  # Include submenu and shadow
                y_min = min(main_menu_y + shadow_offset, sub_menu_y + shadow_offset) - 0.005
                y_max = max(main_menu_y + main_menu_h + shadow_offset, sub_menu_y + sub_menu_h + shadow_offset) + 0.005
                
                ax_human.set_xlim(x_min, x_max)
                ax_human.set_ylim(y_min, y_max)
                
                # Remove axes frames
                ax_human.spines['top'].set_visible(False)
                ax_human.spines['right'].set_visible(False)
                ax_human.spines['bottom'].set_visible(False)
                ax_human.spines['left'].set_visible(False)
            else:
                ax_human.set_xlim(0, WINDOW_WIDTH)
                ax_human.set_ylim(0, WINDOW_HEIGHT)
            
            ax_human.invert_yaxis()
            ax_human.set_aspect('equal')
            ax_human.set_xticks([])
            ax_human.set_yticks([])
            ax_human.grid(False)
            
            # Simulator density plot
            ax_sim = fig.add_subplot(gs[trial_idx * 2 + 1, segment_id])
            
            # Calculate bounds first (needed for hexbin extent and axis limits)
            hexbin_extent = (0, WINDOW_WIDTH, 0, WINDOW_HEIGHT)  # Default extent
            if task_type == 'lasso' and trial_condition:
                # Calculate grid bounds with padding
                grid_layout = trial_condition.get('grid_layout') or trial_condition.get('gridLayout', [])
                icon_spacing = trial_condition.get('icon_spacing') or trial_condition.get('iconSpacing', 0.05)
                grid_origin = trial_condition.get('grid_origin') or trial_condition.get('gridOrigin', [0.1, 0.1])
                icon_radius = trial_condition.get('icon_radius') or trial_condition.get('iconRadius', 0.01)
                
                if grid_layout:
                    origin_x, origin_y = grid_origin
                    rows = len(grid_layout)
                    max_cols = max(len(row.split()) if ' ' in row else len(row) for row in grid_layout) if grid_layout else 0
                    
                    # Calculate grid bounds
                    grid_min_x = origin_x - icon_radius
                    grid_max_x = origin_x + max_cols * icon_spacing + icon_radius
                    grid_min_y = origin_y - icon_radius
                    grid_max_y = origin_y + rows * icon_spacing + icon_radius
                    
                    # Use grid bounds directly (no padding, tight to grid groups)
                    x_min = grid_min_x
                    x_max = grid_max_x
                    y_min = grid_min_y
                    y_max = grid_max_y
                    
                    hexbin_extent = (x_min, x_max, y_min, y_max)
            elif task_type == 'cascading_menu' and trial_condition:
                # Calculate menu bounds (include entire menu, submenu, and shadows)
                main_menu_window_size = trial_condition.get('mainMenuWindowSize') or trial_condition.get('main_menu_window_size', [0.12, 0.12])
                sub_menu_window_size = trial_condition.get('subMenuWindowSize') or trial_condition.get('sub_menu_window_size', [0.1, 0.12])
                main_menu_origin = trial_condition.get('mainMenuOrigin') or trial_condition.get('main_menu_origin', [0.1, 0.02])
                shadow_offset = 0.003  # Shadow offset from _draw_cascading_menu
                
                main_menu_x, main_menu_y = main_menu_origin
                main_menu_w, main_menu_h = main_menu_window_size
                sub_menu_w, sub_menu_h = sub_menu_window_size
                
                # Calculate submenu position
                target_main_idx = trial_condition.get('targetMainMenuIndex') or trial_condition.get('target_main_menu_index', 11)
                main_menu_size = trial_condition.get('mainMenuSize') or trial_condition.get('main_menu_size', 12)
                main_item_height = main_menu_h / main_menu_size
                target_main_item_top = main_menu_y + target_main_idx * main_item_height
                sub_menu_x = main_menu_x + main_menu_w
                sub_menu_y = target_main_item_top
                
                # Calculate bounds including shadows
                # Main menu shadow extends to: main_menu_x + shadow_offset, main_menu_y + shadow_offset
                # Submenu shadow extends to: sub_menu_x + shadow_offset, sub_menu_y + shadow_offset
                x_min = main_menu_x + shadow_offset - 0.005  # Include shadow and small padding
                x_max = sub_menu_x + sub_menu_w + shadow_offset + 0.005  # Include submenu and shadow
                y_min = min(main_menu_y + shadow_offset, sub_menu_y + shadow_offset) - 0.005
                y_max = max(main_menu_y + main_menu_h + shadow_offset, sub_menu_y + sub_menu_h + shadow_offset) + 0.005
                
                hexbin_extent = (x_min, x_max, y_min, y_max)
            
            # Draw task-specific visualizations (before density)
            if task_type == 'lasso' and trial_condition:
                _draw_lasso_grid(ax_sim, trial_condition, zorder_base=1)
            elif task_type == 'cascading_menu' and trial_condition:
                _draw_cascading_menu(ax_sim, trial_condition, zorder_base=1)
            
            # Draw tunnel fill and boundaries first (before density hexbin)
            # zorder=1: boundaries (lowest)
            # zorder=2: fill (above boundaries, below hexbin)
            if task_type != 'lasso' and task_type != 'cascading_menu':
                if tunnel_path is not None:
                    path = np.asarray(tunnel_path, dtype=float)
                    if path.ndim == 2 and path.shape[0] >= 2 and path.shape[1] == 2:
                        left_boundary, right_boundary = generateTunnelBoundaries(tunnel_path, tunnel_width)
                        # Fill polygon: left forward, right reversed
                        poly_x = np.concatenate([left_boundary[:, 0], right_boundary[::-1, 0]])
                        poly_y = np.concatenate([left_boundary[:, 1], right_boundary[::-1, 1]])
                        ax_sim.fill(poly_x, poly_y, color='lightgray', alpha=1.0, zorder=2)
                        # Boundary lines
                        ax_sim.plot(left_boundary[:, 0], left_boundary[:, 1], 
                                color='gray', linestyle='--', linewidth=0.8, zorder=1)
                        ax_sim.plot(right_boundary[:, 0], right_boundary[:, 1], 
                                color='gray', linestyle='--', linewidth=0.8, zorder=1)
            
            # Draw density hexbin (will appear on top due to later drawing order and default zorder ~3)
            hb_sim = None
            if 'sim' in segment_results:
                sim_trajs = segment_results['sim']['trajectories']
                if sim_trajs:
                    all_points = []
                    for traj in sim_trajs:
                        if len(traj) > 0:
                            all_points.extend(traj)
                    
                    if all_points:
                        points_array = np.array(all_points)
                        hb_sim = ax_sim.hexbin(points_array[:, 0], points_array[:, 1], 
                                              gridsize=gridsize, cmap='Oranges', 
                                              vmin=0, vmax=global_max_density,
                                              extent=hexbin_extent,
                                              mincnt=1, linewidths=0.1)
            
            # Set limits based on task type
            if task_type == 'lasso' and trial_condition:
                # Calculate grid bounds with padding
                grid_layout = trial_condition.get('grid_layout') or trial_condition.get('gridLayout', [])
                icon_spacing = trial_condition.get('icon_spacing') or trial_condition.get('iconSpacing', 0.05)
                grid_origin = trial_condition.get('grid_origin') or trial_condition.get('gridOrigin', [0.1, 0.1])
                icon_radius = trial_condition.get('icon_radius') or trial_condition.get('iconRadius', 0.01)
                
                if grid_layout:
                    origin_x, origin_y = grid_origin
                    rows = len(grid_layout)
                    max_cols = max(len(row.split()) if ' ' in row else len(row) for row in grid_layout) if grid_layout else 0
                    
                    # Calculate grid bounds (tight to grid groups, no padding)
                    # Squares are drawn at origin_x + col_idx * icon_spacing where col_idx = 0 to max_cols-1
                    # So the right edge is at origin_x + (max_cols - 1) * icon_spacing + icon_radius
                    grid_min_x = origin_x - icon_radius
                    grid_max_x = origin_x + (max_cols - 1) * icon_spacing + icon_radius
                    grid_min_y = origin_y - icon_radius
                    grid_max_y = origin_y + (rows - 1) * icon_spacing + icon_radius
                    
                    # Use grid bounds directly (no padding)
                    x_min = grid_min_x
                    x_max = grid_max_x
                    y_min = grid_min_y
                    y_max = grid_max_y
                    
                    ax_sim.set_xlim(x_min, x_max)
                    ax_sim.set_ylim(y_min, y_max)
                else:
                    ax_sim.set_xlim(0, WINDOW_WIDTH)
                    ax_sim.set_ylim(0, WINDOW_HEIGHT)
                
                # Remove axes frames
                ax_sim.spines['top'].set_visible(False)
                ax_sim.spines['right'].set_visible(False)
                ax_sim.spines['bottom'].set_visible(False)
                ax_sim.spines['left'].set_visible(False)
            elif task_type == 'cascading_menu' and trial_condition:
                # Calculate menu bounds (include entire menu, submenu, and shadows)
                main_menu_window_size = trial_condition.get('mainMenuWindowSize') or trial_condition.get('main_menu_window_size', [0.12, 0.12])
                sub_menu_window_size = trial_condition.get('subMenuWindowSize') or trial_condition.get('sub_menu_window_size', [0.1, 0.12])
                main_menu_origin = trial_condition.get('mainMenuOrigin') or trial_condition.get('main_menu_origin', [0.1, 0.02])
                shadow_offset = 0.003  # Shadow offset from _draw_cascading_menu
                
                main_menu_x, main_menu_y = main_menu_origin
                main_menu_w, main_menu_h = main_menu_window_size
                sub_menu_w, sub_menu_h = sub_menu_window_size
                
                # Calculate submenu position
                target_main_idx = trial_condition.get('targetMainMenuIndex') or trial_condition.get('target_main_menu_index', 11)
                main_menu_size = trial_condition.get('mainMenuSize') or trial_condition.get('main_menu_size', 12)
                main_item_height = main_menu_h / main_menu_size
                target_main_item_top = main_menu_y + target_main_idx * main_item_height
                sub_menu_x = main_menu_x + main_menu_w
                sub_menu_y = target_main_item_top
                
                # Calculate bounds including shadows
                # Main menu shadow extends to: main_menu_x + shadow_offset, main_menu_y + shadow_offset
                # Submenu shadow extends to: sub_menu_x + shadow_offset, sub_menu_y + shadow_offset
                x_min = main_menu_x + shadow_offset - 0.005  # Include shadow and small padding
                x_max = sub_menu_x + sub_menu_w + shadow_offset + 0.005  # Include submenu and shadow
                y_min = min(main_menu_y + shadow_offset, sub_menu_y + shadow_offset) - 0.005
                y_max = max(main_menu_y + main_menu_h + shadow_offset, sub_menu_y + sub_menu_h + shadow_offset) + 0.005
                
                ax_sim.set_xlim(x_min, x_max)
                ax_sim.set_ylim(y_min, y_max)
                
                # Remove axes frames
                ax_sim.spines['top'].set_visible(False)
                ax_sim.spines['right'].set_visible(False)
                ax_sim.spines['bottom'].set_visible(False)
                ax_sim.spines['left'].set_visible(False)
            else:
                ax_sim.set_xlim(0, WINDOW_WIDTH)
                ax_sim.set_ylim(0, WINDOW_HEIGHT)
            
            ax_sim.invert_yaxis()
            ax_sim.set_aspect('equal')
            ax_sim.set_xticks([])
            ax_sim.set_yticks([])
            ax_sim.grid(False)
    
    # Create filename with trial IDs
    trial_ids_str = '_'.join([f't{tid}' for tid in trial_ids])
    traj_path = Path(output_path_base) / f'trajectories_density_{trial_ids_str}.png'
    fig.savefig(traj_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Trajectory density plot saved to: {traj_path} (global max density: {global_max_density:.1f})")


def _draw_lasso_grid(ax, trial_condition, zorder_base=1):
    """
    Draw lasso grid visualization (targets in orange).
    
    Grid layout characters:
        - 'X': Target square (orange)
        - 'O': Empty placeholder grid cell (not drawn)
        - Other: Empty cell (light grey)
    
    Args:
        ax: Matplotlib axes
        trial_condition: Dictionary with grid_layout, icon_radius, icon_spacing, grid_origin
        zorder_base: Base z-order for drawing (grids should be below trajectories)
    """
    # Get grid configuration
    grid_layout = trial_condition.get('grid_layout') or trial_condition.get('gridLayout', [])
    icon_radius = trial_condition.get('icon_radius') or trial_condition.get('iconRadius', 0.01)
    icon_spacing = trial_condition.get('icon_spacing') or trial_condition.get('iconSpacing', 0.05)
    grid_origin = trial_condition.get('grid_origin') or trial_condition.get('gridOrigin', [0.1, 0.1])
    
    if not grid_layout:
        return
    
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
            
            # Skip empty placeholder cells ('O')
            if char == 'O':
                continue
            
            if char == 'X':
                # Target square (orange)
                color = 'lightgreen'
            else:
                # Empty cell or other character (light grey)
                color = 'lightgrey'
            
            square = Rectangle(
                (x - icon_radius, y - icon_radius),
                icon_radius * 2,
                icon_radius * 2,
                facecolor=color,
                edgecolor='black',
                linewidth=0.0,
                alpha=0.6,
                zorder=zorder_base
            )
            ax.add_patch(square)


def _draw_cascading_menu(ax, trial_condition, zorder_base=1):
    """
    Draw cascading menu visualization.
    
    Args:
        ax: Matplotlib axes
        trial_condition: Dictionary with menu configuration
        zorder_base: Base z-order for drawing (menus should be below trajectories)
    """
    # Get menu configuration
    main_menu_size = trial_condition.get('mainMenuSize') or trial_condition.get('main_menu_size', 12)
    sub_menu_size = trial_condition.get('subMenuSize') or trial_condition.get('sub_menu_size', 12)
    target_main_idx = trial_condition.get('targetMainMenuIndex') or trial_condition.get('target_main_menu_index', 11)
    target_sub_idx = trial_condition.get('targetSubMenuIndex') or trial_condition.get('target_sub_menu_index', 11)
    main_menu_window_size = trial_condition.get('mainMenuWindowSize') or trial_condition.get('main_menu_window_size', [0.12, 0.12])
    sub_menu_window_size = trial_condition.get('subMenuWindowSize') or trial_condition.get('sub_menu_window_size', [0.1, 0.12])
    main_menu_origin = trial_condition.get('mainMenuOrigin') or trial_condition.get('main_menu_origin', [0.1, 0.02])
    
    # Colors
    menu_bg_color = 'white'
    menu_border_color = '#C0C0C0'
    target_color = '#FF6B6B'
    shadow_color = 'black'
    shadow_alpha = 0.15
    shadow_offset = 0.003
    
    main_menu_x, main_menu_y = main_menu_origin
    main_menu_w, main_menu_h = main_menu_window_size
    sub_menu_w, sub_menu_h = sub_menu_window_size
    
    main_item_height = main_menu_h / main_menu_size
    sub_item_height = sub_menu_h / sub_menu_size
    
    # Draw main menu shadow
    shadow_rect = Rectangle(
        (main_menu_x + shadow_offset, main_menu_y + shadow_offset),
        main_menu_w,
        main_menu_h,
        facecolor=shadow_color,
        edgecolor='none',
        alpha=shadow_alpha,
        zorder=zorder_base
    )
    ax.add_patch(shadow_rect)
    
    # Draw main menu background
    main_menu_rect = Rectangle(
        (main_menu_x, main_menu_y),
        main_menu_w,
        main_menu_h,
        facecolor=menu_bg_color,
        edgecolor=menu_border_color,
        linewidth=1.5,
        zorder=zorder_base + 1
    )
    ax.add_patch(main_menu_rect)
    
    # Draw main menu items
    for i in range(main_menu_size):
        item_y = main_menu_y + i * main_item_height
        item_color = target_color if i == target_main_idx else menu_bg_color
        item_alpha = 0.9 if i == target_main_idx else 1.0
        
        item_rect = Rectangle(
            (main_menu_x, item_y),
            main_menu_w,
            main_item_height,
            facecolor=item_color,
            edgecolor='none',
            alpha=item_alpha,
            zorder=zorder_base + 2
        )
        ax.add_patch(item_rect)
        
        # Draw separator
        if i < main_menu_size - 1:
            separator_y = item_y + main_item_height
            ax.plot([main_menu_x, main_menu_x + main_menu_w], 
                   [separator_y, separator_y],
                   color=menu_border_color, linewidth=0.5, zorder=zorder_base + 3)
        
        # Add arrow on target item
        if i == target_main_idx:
            arrow_x = main_menu_x + main_menu_w - 0.003
            arrow_y = item_y + main_item_height / 2
            arrow_size = main_item_height * 0.15
            arrow_points = np.array([
                [arrow_x, arrow_y - arrow_size/2],
                [arrow_x, arrow_y + arrow_size/2],
                [arrow_x + arrow_size, arrow_y]
            ])
            arrow_patch = Polygon(arrow_points, facecolor='white', 
                                edgecolor='none', zorder=zorder_base + 4)
            ax.add_patch(arrow_patch)
    
    # Draw submenu
    target_main_item_top = main_menu_y + target_main_idx * main_item_height
    sub_menu_x = main_menu_x + main_menu_w
    sub_menu_y = target_main_item_top
    
    # Draw submenu shadow
    shadow_rect = Rectangle(
        (sub_menu_x + shadow_offset, sub_menu_y + shadow_offset),
        sub_menu_w,
        sub_menu_h,
        facecolor=shadow_color,
        edgecolor='none',
        alpha=shadow_alpha,
        zorder=zorder_base
    )
    ax.add_patch(shadow_rect)
    
    # Draw submenu background
    sub_menu_rect = Rectangle(
        (sub_menu_x, sub_menu_y),
        sub_menu_w,
        sub_menu_h,
        facecolor=menu_bg_color,
        edgecolor=menu_border_color,
        linewidth=1.5,
        zorder=zorder_base + 1
    )
    ax.add_patch(sub_menu_rect)
    
    # Draw submenu items
    for i in range(sub_menu_size):
        item_y = sub_menu_y + i * sub_item_height
        item_color = target_color if i == target_sub_idx else menu_bg_color
        item_alpha = 0.9 if i == target_sub_idx else 1.0
        
        item_rect = Rectangle(
            (sub_menu_x, item_y),
            sub_menu_w,
            sub_item_height,
            facecolor=item_color,
            edgecolor='none',
            alpha=item_alpha,
            zorder=zorder_base + 2
        )
        ax.add_patch(item_rect)
        
        # Draw separator
        if i < sub_menu_size - 1:
            separator_y = item_y + sub_item_height
            ax.plot([sub_menu_x, sub_menu_x + sub_menu_w], 
                   [separator_y, separator_y],
                   color=menu_border_color, linewidth=0.5, zorder=zorder_base + 3)


def plot_experiment_results(all_results, output_path_base, tunnel_paths, tunnel_widths, trial_metadata=None):
    """
    Plot all trajectories and speeds for all trials and segments.
    Creates two separate plots: one for trajectories, one for speeds.
    
    Args:
        all_results: dict {trial_id: {segment_id: {human: {...}, sim: {...}}}}
        output_path_base: Folder path for saving plots
        tunnel_paths: dict {trial_id: tunnel_path}
        tunnel_widths: dict {trial_id: tunnel_width}
        trial_metadata: dict {trial_id: {participant_id: {round: {condition, ...}}}} - optional, for task-specific visualizations
    """
    trial_ids = sorted(all_results.keys())
    n_trials = len(trial_ids)
    
    if n_trials == 0:
        print("No results to plot")
        return
    
    # Get number of segments (assume same for all trials)
    first_trial = all_results[trial_ids[0]]
    n_segments = len(first_trial)
    
    # Create trajectory plot
    fig_traj = plt.figure(figsize=(6 * n_segments, 4 * n_trials))
    gs_traj = GridSpec(n_trials, n_segments, figure=fig_traj, hspace=0.3, wspace=0.3)
    
    # Create speed plot
    fig_speed = plt.figure(figsize=(6 * n_segments, 4 * n_trials))
    gs_speed = GridSpec(n_trials, n_segments, figure=fig_speed, hspace=0.3, wspace=0.3)
    
    for trial_idx, trial_id in enumerate(trial_ids):
        trial_results = all_results[trial_id]
        tunnel_path = tunnel_paths.get(trial_id, [])
        tunnel_width = tunnel_widths.get(trial_id, 0.02)
        
        for segment_id in sorted(trial_results.keys()):
            segment_results = trial_results[segment_id]
            
            # Trajectory subplot
            ax_traj = fig_traj.add_subplot(gs_traj[trial_idx, segment_id])
            
            # Get task type from trial metadata
            task_type = None
            trial_condition = None
            if trial_metadata and trial_id in trial_metadata:
                # Get condition from first participant and first round
                for participant_id in trial_metadata[trial_id].keys():
                    for round_num in trial_metadata[trial_id][participant_id].keys():
                        trial_condition = trial_metadata[trial_id][participant_id][round_num].get('condition', {})
                        if trial_condition:
                            task_type = trial_condition.get('tunnelType', 'wave')
                            break
                    if trial_condition:
                        break
            
            # Draw task-specific visualizations (before trajectories)
            if task_type == 'lasso' and trial_condition:
                _draw_lasso_grid(ax_traj, trial_condition, zorder_base=1)
            elif task_type == 'cascading_menu' and trial_condition:
                _draw_cascading_menu(ax_traj, trial_condition, zorder_base=1)
            
            # Draw tunnel boundaries (like in draw_trajectory) - only for tunnel tasks
            if task_type != 'lasso' and task_type != 'cascading_menu':
                if tunnel_path is not None and tunnel_width is not None:
                    path = np.asarray(tunnel_path, dtype=float)
                    if path.ndim == 2 and path.shape[0] >= 2 and path.shape[1] == 2:
                        # Generate tunnel boundaries
                        left_boundary, right_boundary = generateTunnelBoundaries(tunnel_path, tunnel_width)
                        
                        # Fill polygon: left forward, right reversed
                        # zorder=2: above boundaries (1) but below trajectories (10)
                        poly_x = np.concatenate([left_boundary[:, 0], right_boundary[::-1, 0]])
                        poly_y = np.concatenate([left_boundary[:, 1], right_boundary[::-1, 1]])
                        ax_traj.fill(poly_x, poly_y, color='lightgray', alpha=1.0, zorder=2)
                        
                        # Boundary lines
                        # zorder=1: lowest layer (boundaries)
                        ax_traj.plot(left_boundary[:, 0], left_boundary[:, 1], 
                                color='gray', linestyle='--', linewidth=0.8, zorder=1)
                        ax_traj.plot(right_boundary[:, 0], right_boundary[:, 1], 
                                color='gray', linestyle='--', linewidth=0.8, zorder=1)
            
            # Get target position (end of segment - use first human trajectory or first sim trajectory)
            # For lasso and cascading menu, also draw start/end markers
            target_pos = None
            target_radius = 0.01
            if 'human' in segment_results and segment_results['human']['trajectories']:
                last_traj = segment_results['human']['trajectories'][0]
                if len(last_traj) > 0:
                    target_pos = (last_traj[-1][0], last_traj[-1][1])
            elif 'sim' in segment_results and segment_results['sim']['trajectories']:
                last_traj = segment_results['sim']['trajectories'][0]
                if len(last_traj) > 0:
                    target_pos = (last_traj[-1][0], last_traj[-1][1])
            
            # Draw target
            if target_pos:
                # target_circle = Circle(target_pos, target_radius, color='red', alpha=0.5, zorder=5)
                # ax_traj.add_patch(target_circle)
                ax_traj.scatter([target_pos[0]], [target_pos[1]], color='red', 
                              edgecolor='black', s=20, zorder=5)
            
            # Plot human trajectories (blue) - can be multiple
            if 'human' in segment_results:
                human_trajs = segment_results['human']['trajectories']
                for i, human_traj in enumerate(human_trajs):
                    if len(human_traj) == 0:
                        continue
                    human_xs = [p[0] for p in human_traj]
                    human_ys = [p[1] for p in human_traj]
                    ax_traj.plot(human_xs, human_ys, 'b-', linewidth=0.5, alpha=0.4, zorder=10)
                    # Draw start point
                    if i == 0:  # Only label start for first trajectory
                        ax_traj.scatter(human_xs[0], human_ys[0], color='green', s=30, 
                                      zorder=11)
            
            # Plot simulator trajectories (orange)
            if 'sim' in segment_results:
                sim_trajs = segment_results['sim']['trajectories']
                for i, sim_traj in enumerate(sim_trajs):
                    if len(sim_traj) == 0:
                        continue
                    sim_xs = [p[0] for p in sim_traj]
                    sim_ys = [p[1] for p in sim_traj]
                    ax_traj.plot(sim_xs, sim_ys, 'orange', linewidth=0.5, alpha=0.7, zorder=10)

            # Plot baseline trajectories (green)
            if 'baseline' in segment_results:
                bl_trajs = segment_results['baseline']['trajectories']
                for i, bl_traj in enumerate(bl_trajs):
                    if len(bl_traj) == 0:
                        continue
                    bl_xs = [p[0] for p in bl_traj]
                    bl_ys = [p[1] for p in bl_traj]
                    ax_traj.plot(bl_xs, bl_ys, 'tab:green', linewidth=0.5, alpha=0.7, zorder=10)
            
            # Set environment limits and aspect (like in draw_trajectory)
            if task_type == 'lasso' and trial_condition:
                # Calculate grid bounds with padding
                grid_layout = trial_condition.get('grid_layout') or trial_condition.get('gridLayout', [])
                icon_spacing = trial_condition.get('icon_spacing') or trial_condition.get('iconSpacing', 0.05)
                grid_origin = trial_condition.get('grid_origin') or trial_condition.get('gridOrigin', [0.1, 0.1])
                icon_radius = trial_condition.get('icon_radius') or trial_condition.get('iconRadius', 0.01)
                
                if grid_layout:
                    origin_x, origin_y = grid_origin
                    rows = len(grid_layout)
                    max_cols = max(len(row.split()) if ' ' in row else len(row) for row in grid_layout) if grid_layout else 0
                    
                    # Calculate grid bounds (tight to grid groups, no padding)
                    # Squares are drawn at origin_x + col_idx * icon_spacing where col_idx = 0 to max_cols-1
                    # So the right edge is at origin_x + (max_cols - 1) * icon_spacing + icon_radius
                    grid_min_x = origin_x - icon_radius
                    grid_max_x = origin_x + (max_cols - 1) * icon_spacing + icon_radius
                    grid_min_y = origin_y - icon_radius
                    grid_max_y = origin_y + (rows - 1) * icon_spacing + icon_radius
                    
                    # Use grid bounds directly (no padding)
                    x_min = grid_min_x
                    x_max = grid_max_x
                    y_min = grid_min_y
                    y_max = grid_max_y
                    
                    ax_traj.set_xlim(x_min, x_max)
                    ax_traj.set_ylim(y_min, y_max)
                else:
                    ax_traj.set_xlim(0, WINDOW_WIDTH)
                    ax_traj.set_ylim(0, WINDOW_HEIGHT)
                
                # Remove axes frames
                ax_traj.spines['top'].set_visible(False)
                ax_traj.spines['right'].set_visible(False)
                ax_traj.spines['bottom'].set_visible(False)
                ax_traj.spines['left'].set_visible(False)
            elif task_type == 'cascading_menu' and trial_condition:
                # Calculate menu bounds (include entire menu, submenu, and shadows)
                main_menu_window_size = trial_condition.get('mainMenuWindowSize') or trial_condition.get('main_menu_window_size', [0.12, 0.12])
                sub_menu_window_size = trial_condition.get('subMenuWindowSize') or trial_condition.get('sub_menu_window_size', [0.1, 0.12])
                main_menu_origin = trial_condition.get('mainMenuOrigin') or trial_condition.get('main_menu_origin', [0.1, 0.02])
                shadow_offset = 0.003  # Shadow offset from _draw_cascading_menu
                
                main_menu_x, main_menu_y = main_menu_origin
                main_menu_w, main_menu_h = main_menu_window_size
                sub_menu_w, sub_menu_h = sub_menu_window_size
                
                # Calculate submenu position
                target_main_idx = trial_condition.get('targetMainMenuIndex') or trial_condition.get('target_main_menu_index', 11)
                main_menu_size = trial_condition.get('mainMenuSize') or trial_condition.get('main_menu_size', 12)
                main_item_height = main_menu_h / main_menu_size
                target_main_item_top = main_menu_y + target_main_idx * main_item_height
                sub_menu_x = main_menu_x + main_menu_w
                sub_menu_y = target_main_item_top
                
                # Calculate bounds including shadows
                # Main menu shadow extends to: main_menu_x + shadow_offset, main_menu_y + shadow_offset
                # Submenu shadow extends to: sub_menu_x + shadow_offset, sub_menu_y + shadow_offset
                x_min = main_menu_x + shadow_offset - 0.005  # Include shadow and small padding
                x_max = sub_menu_x + sub_menu_w + shadow_offset + 0.005  # Include submenu and shadow
                y_min = min(main_menu_y + shadow_offset, sub_menu_y + shadow_offset) - 0.005
                y_max = max(main_menu_y + main_menu_h + shadow_offset, sub_menu_y + sub_menu_h + shadow_offset) + 0.005
                
                ax_traj.set_xlim(x_min, x_max)
                ax_traj.set_ylim(y_min, y_max)
                
                # Remove axes frames
                ax_traj.spines['top'].set_visible(False)
                ax_traj.spines['right'].set_visible(False)
                ax_traj.spines['bottom'].set_visible(False)
                ax_traj.spines['left'].set_visible(False)
            else:
                ax_traj.set_xlim(0, WINDOW_WIDTH)
                ax_traj.set_ylim(0, WINDOW_HEIGHT)
            
            ax_traj.invert_yaxis()
            ax_traj.set_aspect('equal')
            
            ax_traj.set_xticks([])
            ax_traj.set_yticks([])
            ax_traj.grid(False)
            
            # # Speed subplot
            # ax_speed = fig_speed.add_subplot(gs_speed[trial_idx, segment_id])
            
            # # Plot human speeds (blue)
            # if 'human' in segment_results:
            #     human_speeds_list = segment_results['human']['speeds']
            #     for i, speeds in enumerate(human_speeds_list):
            #         if len(speeds) == 0:
            #             continue
            #         time_points = np.arange(0, len(speeds) * 0.05, 0.05)
            #         if len(time_points) > len(speeds):
            #             time_points = time_points[:len(speeds)]
            #         ax_speed.plot(time_points, speeds, 'b-', linewidth=0.5, alpha=0.7)
            
            # # Plot simulator speeds (orange)
            # if 'sim' in segment_results:
            #     sim_speeds_list = segment_results['sim']['speeds']
            #     for i, speeds in enumerate(sim_speeds_list):
            #         if len(speeds) == 0:
            #             continue
            #         time_points = np.arange(0, len(speeds) * 0.05, 0.05)
            #         if len(time_points) > len(speeds):
            #             time_points = time_points[:len(speeds)]
            #         ax_speed.plot(time_points, speeds, 'orange', linewidth=0.5, alpha=1.0)
            
            # ax_speed.set_xlabel('Time (s)')
            # ax_speed.set_ylabel('Speed (m/s)')
            # ax_speed.grid(False)
    
    # Create filename with trial IDs
    trial_ids_str = '_'.join([f't{tid}' for tid in trial_ids])
    
    # Save trajectory plot
    traj_path = Path(output_path_base) / f'trajectories_{trial_ids_str}.png'
    fig_traj.savefig(traj_path, dpi=300, bbox_inches='tight')
    plt.close(fig_traj)
    print(f"Trajectory plot saved to: {traj_path}")
    
    # Save speed plot
    # speed_path = Path(output_path_base) / f'speeds_{trial_ids_str}.png'
    # fig_speed.savefig(speed_path, dpi=300, bbox_inches='tight')
    # plt.close(fig_speed)
    # print(f"Speed plot saved to: {speed_path}")
