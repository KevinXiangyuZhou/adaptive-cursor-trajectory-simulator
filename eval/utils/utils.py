import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directories to path
project_root = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, project_root)

from simulators.steering_simulator import Env
import simulators.point_and_click_modules.mouse_module as mouse
from evaluation.steering.process import compute_speeds_from_trajectory

# Import metrics (use relative import from current directory)
sys.path.insert(0, os.path.dirname(__file__))
from metrics.compute_metrics import (
    compute_rmse_contouring,
    compute_rmse_lagging,
    compute_constraint_violation_rate,
    compute_path_length,
    compute_completion_time
)


def compute_hand_pos_from_cursor(cursor_pos, cursor_vel, prev_hand_pos, forearm, interval, nc):
    """
    Reverse function: Compute hand position from cursor position and velocity.
    
    This is an approximation that reverses the mouse noise and motor noise effects.
    The forward function is: hand -> motor noise -> mouse noise -> cursor
    This reverse function approximates: cursor -> mouse noise reverse -> motor noise reverse -> hand
    
    Args:
        cursor_pos: Current cursor position (x, y)
        cursor_vel: Current cursor velocity (vx, vy)
        prev_hand_pos: Previous hand position (x, y) - needed for orientation
        forearm: Forearm length
        interval: Time step
        nc: Motor noise parameters [nc_dir, nc_per]
    
    Returns:
        Estimated hand position (x, y)
    """
    # Get cursor speed
    cursor_speed = np.sqrt(cursor_vel[0]**2 + cursor_vel[1]**2)
    
    # Get mouse gain (approximate, using average gain for small speeds)
    if cursor_speed > 1e-6:
        mouse_gain_val = mouse.gain_func_can(cursor_speed)
    else:
        mouse_gain_val = 1.0
    
    # Reverse mouse gain: device-frame velocity
    inv_gain = 1.0 / max(mouse_gain_val, 1e-6)
    vx_dev = cursor_vel[0] * inv_gain
    vy_dev = cursor_vel[1] * inv_gain
    
    # Approximate reverse of motor noise (simplified - assumes noise is small)
    # In practice, motor noise adds random components, so we can't perfectly reverse it
    # We'll use the device-frame velocity as an approximation
    vx_hand = vx_dev
    vy_hand = vy_dev
    
    # Reverse mouse rotation and compute hand displacement
    # Get previous hand orientation
    prev_hand_ori = mouse.get_hand_orientation(np.array(prev_hand_pos), forearm)
    
    # Approximate cursor displacement from velocity
    cursor_dx = cursor_vel[0] * interval
    cursor_dy = cursor_vel[1] * interval
    
    # Reverse the mouse rotation effect
    # The forward function rotates hand displacement, so we reverse-rotate cursor displacement
    hand_dx, hand_dy = mouse.rot_mat(cursor_dx, cursor_dy, prev_hand_ori)
    
    # Estimate hand position (simplified - doesn't fully account for rotation effects)
    # This is an approximation; the exact reverse would require solving a nonlinear system
    hand_x = prev_hand_pos[0] + hand_dx
    hand_y = prev_hand_pos[1] + hand_dy
    
    return np.array([hand_x, hand_y], dtype=float)


def normalize_trajectory_format(trajectory):
    """
    Normalize trajectory to [[x, y], ...] format.
    
    Handles both list format [[x, y], ...] and dict format [{"x": ..., "y": ...}, ...].
    
    Args:
        trajectory: Trajectory in either format
    
    Returns:
        List of [x, y] points
    """
    if not trajectory:
        return []
    
    # Check if first element is a dict
    if isinstance(trajectory[0], dict):
        return [[p['x'], p['y']] for p in trajectory]
    else:
        # Already in list format
        return trajectory


def load_human_trajectory(trial_num):
    """
    Load human trajectory data for a given trial from processed directory.
    
    Args:
        trial_num: Trial number (1-4)
    
    Returns:
        tuple: (trajectory, speeds, completion_time, tunnel_width, tunnel_curvature, description)
    """
    # Load processed trajectory data
    processed_dir = Path(__file__).parent / "human_data" / "processed" / str(trial_num)
    
    # Find first available processed data file
    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No processed human data files found in {processed_dir}")
    
    # Load processed data
    with open(json_files[0], 'r') as f:
        processed_data = json.load(f)
    
    # Extract trajectory and normalize format
    trajectory = processed_data.get('trajectory', [])
    trajectory = normalize_trajectory_format(trajectory)
    # Extract speeds
    speeds = processed_data.get('speed', [])
    
    # Get tunnel configuration from raw data files
    # Try to find condition info from raw files
    raw_dir = Path(__file__).parent / "human_data" / "raw"
    raw_files = list(raw_dir.glob("*.json")) if raw_dir.exists() else []
    
    tunnel_width = None
    tunnel_curvature = None
    description = ''
    completion_time = 0.0
    
    # Search raw files for trial data with matching trial_id
    for raw_file in raw_files:
        try:
            with open(raw_file, 'r') as f:
                raw_data = json.load(f)
            
            trial_data_array = raw_data.get('trialData', [])
            for trial_data in trial_data_array:
                if trial_data.get('trial_id') == trial_num:
                    # Found matching trial
                    condition = trial_data.get('condition', {})
                    tunnel_width = condition.get('tunnelWidth')
                    tunnel_curvature = condition.get('curvature')
                    description = condition.get('description', '')
                    completion_time = trial_data.get('completionTime', 0.0)
                    break
            
            if tunnel_width is not None:
                break
        except:
            continue
    
    # If no condition found in raw files, use trial config defaults
    # Import TRIAL_CONFIGS from run_eval to avoid circular import
    if tunnel_width is None or tunnel_curvature is None:
        # Import here to avoid circular import
        from evaluation.steering.run_eval import TRIAL_CONFIGS
        config = TRIAL_CONFIGS.get(trial_num, {})
        tunnel_width = tunnel_width or config.get('width')
        tunnel_curvature = tunnel_curvature or config.get('curvature')
        description = description or config.get('description', '')
    
    return trajectory, speeds, completion_time, tunnel_width, tunnel_curvature, description


def load_all_human_trajectories(trial_num):
    """
    Load all human trajectory data for a given trial.
    
    Args:
        trial_num: Trial number (1-4)
    
    Returns:
        list: List of (trajectory, speeds) tuples for each human
    """
    processed_dir = Path(__file__).parent / "human_data" / "processed" / str(trial_num)
    json_files = sorted(list(processed_dir.glob("*.json")))
    
    human_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            processed_data = json.load(f)
        trajectory = processed_data.get('trajectory', [])
        trajectory = normalize_trajectory_format(trajectory)
        speeds = processed_data.get('speed', [])
        human_data.append((trajectory, speeds))
    
    return human_data


def run_simulator(config_path, tunnel_width, tunnel_curvature, max_steps=1000,
                  cursor_pos=None, cursor_vel=None, hand_pos=None, target_pos=None,
                  random_seed=None):
    """
    Run steering simulator with given configuration.
    
    Args:
        config_path: Path to simulator configuration JSON
        tunnel_width: Tunnel width
        tunnel_curvature: Tunnel curvature
        max_steps: Maximum simulation steps
        cursor_pos: Optional initial cursor position (x, y)
        cursor_vel: Optional initial cursor velocity (vx, vy)
        hand_pos: Optional initial hand position (x, y)
        target_pos: Optional target position to stop at (x, y). If None, runs until done.
        random_seed: Optional random seed to override config value
    
    Returns:
        tuple: (trajectory, speeds, time_step, tunnel_path)
    """
    # Load simulation configuration
    with open(config_path, 'r') as f:
        sim_config = json.load(f)
    
    # Load task configuration (try to find it in the same directory)
    import os
    config_dir = os.path.dirname(config_path) if os.path.dirname(config_path) else '.'
    task_config_path = os.path.join(config_dir, 'task_configuration.json')
    
    if os.path.exists(task_config_path):
        with open(task_config_path, 'r') as f:
            task_config = json.load(f)
    else:
        # Create minimal task config from parameters
        task_config = {
            'task_type': 'tunnel',
            'tunnel_type': 'wave',
            'tunnel_width': tunnel_width,
            'tunnel_curvature': tunnel_curvature
        }
    
    # Update tunnel parameters in task_config
    task_config['tunnel_width'] = tunnel_width
    task_config['tunnel_curvature'] = tunnel_curvature
    
    # Ensure required fields are present
    if 'target_pos' not in task_config:
        task_config['target_pos'] = [0.46, 0.13]  # Default target at end
    if 'target_radius' not in task_config:
        task_config['target_radius'] = 0.01
    if 'window_width' not in task_config:
        task_config['window_width'] = 0.46
    if 'window_height' not in task_config:
        task_config['window_height'] = 0.26
    
    # Generate reference_path from task_config
    from experiment.utils import generate_reference_path_from_task_config
    reference_path = generate_reference_path_from_task_config(task_config)
    
    # Update random seed if provided (in sim_config)
    if random_seed is not None:
        sim_config['random_seed'] = random_seed
        # Also update other random seeds if present
        if 'ddm_seed' in sim_config:
            sim_config['ddm_seed'] = random_seed
        if 'primitive_seed' in sim_config:
            sim_config['primitive_seed'] = random_seed
    
    # Create environment with separate configs and reference_path
    env = Env(sim_config=sim_config, task_config=task_config, reference_path=reference_path)
    
    # Reset environment with optional initial state
    observation = env.reset(cursor_pos=cursor_pos, cursor_vel=cursor_vel, hand_pos=hand_pos)
    
    # Run simulation
    trajectory = [[observation[0], observation[1]]]
    time_step = env.Interval
    
    for t in range(max_steps):
        # Step environment (action=0 means no external action)
        observation, done, info = env.step(0)
        
        # Record position
        cursor_x = observation[0]
        cursor_y = observation[1]
        trajectory.append([cursor_x, cursor_y])
        
        # Check if reached target position (if specified)
        if target_pos is not None:
            dist_to_target = np.sqrt((cursor_x - target_pos[0])**2 + (cursor_y - target_pos[1])**2)
            if dist_to_target < 0.01:  # Within 1cm of target
                break
        
        if done:
            break
    
    # Compute speeds from trajectory using the same method as user speeds
    # Create timestamps from time_step (in milliseconds)
    timestamps = [i * time_step * 1000.0 for i in range(len(trajectory))]
    
    # Use same window size as user speeds (5)
    speed_window_size = 5
    
    # Compute speeds from trajectory (returns empty list if trajectory too short)
    if len(trajectory) >= 2:
        speeds = compute_speeds_from_trajectory(trajectory, timestamps, window_size=speed_window_size)
        
        # Ensure speeds array matches trajectory length
        if len(speeds) != len(trajectory):
            if len(speeds) < len(trajectory):
                # Pad with last value (or 0.0 if speeds is empty)
                if speeds:
                    speeds.extend([speeds[-1]] * (len(trajectory) - len(speeds)))
                else:
                    speeds = [0.0] * len(trajectory)
            else:
                # Truncate
                speeds = speeds[:len(trajectory)]
    else:
        # Too few points, return zeros
        speeds = [0.0] * len(trajectory)
    
    tunnel_path = env.tunnel_path
    
    return trajectory, speeds, time_step, tunnel_path


def compute_all_metrics(sim_trajectory, sim_speeds, sim_time_step, sim_tunnel_path,
                       human_trajectory, human_completion_time, tunnel_width):
    """
    Compute all evaluation metrics.
    
    Returns:
        dict with all metric values
    """
    # RMSE contouring and lagging
    rmse_c = compute_rmse_contouring(sim_trajectory, human_trajectory)
    rmse_l = compute_rmse_lagging(sim_trajectory, human_trajectory)
    
    # Constraint violation rate
    violation_rate = compute_constraint_violation_rate(
        sim_trajectory, sim_tunnel_path, tunnel_width
    )
    
    # Completion time difference
    sim_completion_time = compute_completion_time(sim_trajectory, sim_time_step)
    completion_time_diff = sim_completion_time - human_completion_time
    
    # Path length difference
    sim_path_length = compute_path_length(sim_trajectory)
    human_path_length = compute_path_length(human_trajectory)
    path_length_diff = abs(sim_path_length - human_path_length)
    
    return {
        'RMSE_c': rmse_c,
        'RMSE_l': rmse_l,
        'constraint_violation_rate': violation_rate,
        'completion_time_difference': completion_time_diff,
        'path_length_difference': path_length_diff,
    }
