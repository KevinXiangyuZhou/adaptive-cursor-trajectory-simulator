"""
Process raw human steering data.

This module processes raw human trial data by:
1. Loading raw JSON files from human_data/raw/
2. Extracting trajectory data (assumed to be already sampled at constant interval)
3. Computing speeds using central difference and moving average
4. Saving processed data to human_data/processed_{time_step}s/
"""

import json
import os
import math
from pathlib import Path
import sys

# Add parent directory to path to import from project root
project_root = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, project_root)


def moving_average(data, window_size):
    """
    Apply moving average filter to a list of values.
    
    Args:
        data: List of values
        window_size: Size of the moving average window (must be odd)
    
    Returns:
        List of smoothed values (same length as input)
    """
    if len(data) == 0:
        return []
    
    if window_size <= 1:
        return data
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed = []
    n = len(data)
    
    for i in range(n):
        # Determine the window boundaries
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        
        # Compute average over the window
        window_data = data[start_idx:end_idx]
        avg = sum(window_data) / len(window_data)
        smoothed.append(avg)
    
    return smoothed


def compute_speeds_from_trajectory(trajectory, timestamps, window_size=5):
    """
    Compute speeds from trajectory positions and timestamps using central difference
    and apply moving average smoothing.
    
    Uses central difference for interior points (more accurate) and forward/backward
    difference for boundary points, then applies a moving average filter.
    
    Args:
        trajectory: List of [x, y] points
        timestamps: List of timestamps in milliseconds
        window_size: Size of moving average window (default: 5)
    
    Returns:
        List of computed and smoothed speeds (same length as trajectory)
    """
    if len(trajectory) < 2 or len(timestamps) < 2:
        return []
    
    if len(trajectory) != len(timestamps):
        return []
    
    # First compute raw speeds using central difference
    raw_speeds = []
    n = len(trajectory)
    
    for i in range(n):
        if i == 0:
            # First point: use forward difference
            p0 = trajectory[i]
            p1 = trajectory[i + 1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dist = math.sqrt(dx**2 + dy**2)
            dt_ms = timestamps[i + 1] - timestamps[i]
            dt_sec = dt_ms / 1000.0
        elif i == n - 1:
            # Last point: use backward difference
            p0 = trajectory[i - 1]
            p1 = trajectory[i]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dist = math.sqrt(dx**2 + dy**2)
            dt_ms = timestamps[i] - timestamps[i - 1]
            dt_sec = dt_ms / 1000.0
        else:
            # Interior points: use central difference
            p_prev = trajectory[i - 1]
            p_next = trajectory[i + 1]
            dx = p_next[0] - p_prev[0]
            dy = p_next[1] - p_prev[1]
            dist = math.sqrt(dx**2 + dy**2)
            dt_ms = timestamps[i + 1] - timestamps[i - 1]
            dt_sec = dt_ms / 1000.0
        
        # Compute speed (avoid division by zero)
        if dt_sec > 0:
            speed = dist / dt_sec
        else:
            speed = 0.0
        
        raw_speeds.append(float(speed))
    
    # Apply moving average smoothing
    smoothed_speeds = moving_average(raw_speeds, window_size)
    if window_size <= 1:
        assert smoothed_speeds == raw_speeds, "Smoothed speeds should be equal to raw speeds if window size is 1"
    
    return smoothed_speeds


def extract_raw_trajectory(trial_data):
    """
    Extract raw trajectory from trial data.
    
    Args:
        trial_data: Dict containing trial data with 'trajectory' key
    
    Returns:
        trajectory as [[x, y], ...]
    """
    trajectory = trial_data.get('trajectory', [])
    
    # Convert trajectory from dict format [{"x": ..., "y": ...}] to list format [[x, y], ...]
    if trajectory and isinstance(trajectory[0], dict):
        trajectory = [[p['x'], p['y']] for p in trajectory]
    
    return trajectory


def process_trial_from_raw_data(trial_data, speed_window_size=5):
    """
    Process a single trial from raw data structure.
    
    Assumes trajectory is already sampled at constant interval.
    
    Args:
        trial_data: Dict containing trial data (from trialData array)
        speed_window_size: Window size for moving average smoothing of speeds (default: 5)
    
    Returns:
        tuple: (processed_data, trial_id) or (None, None) if failed
    """
    try:
        timestamps = trial_data.get('timestamps', [])
        trajectory = trial_data.get('trajectory', [])
        trial_id = trial_data.get('trial_id')
        
        if not timestamps or not trajectory:
            return None, None
        
        # Extract and normalize trajectory format
        trajectory = extract_raw_trajectory(trial_data)
        
        if len(trajectory) != len(timestamps):
            return None, None
        
        # Compute speeds from trajectory using central difference + moving average
        # (trajectory is assumed to be already at constant interval)
        computed_speeds = compute_speeds_from_trajectory(
            trajectory, 
            timestamps, 
            window_size=speed_window_size
        )
        
        # Ensure speeds array matches trajectory length
        if len(computed_speeds) != len(trajectory):
            # If mismatch, pad or truncate to match
            if len(computed_speeds) < len(trajectory):
                # Pad with last value
                computed_speeds.extend([computed_speeds[-1]] * (len(trajectory) - len(computed_speeds)))
            else:
                # Truncate
                computed_speeds = computed_speeds[:len(trajectory)]
        
        processed_data = {
            'trajectory': trajectory,
            'speed': computed_speeds
        }
        
        return processed_data, trial_id
        
    except Exception as e:
        print(f"  ✗ Error processing trial: {e}")
        return None, None


def organize_by_trial(raw_data_dir, processed_base_dir, speed_window_size=5):
    """
    Process raw data files and organize them by trial number.
    
    This function:
    1. Reads raw JSON files (which contain trialData arrays)
    2. Extracts each trial from trialData array
    3. Processes and saves to trial-specific directories
    
    Args:
        raw_data_dir: Directory containing raw JSON files
        processed_base_dir: Base directory for processed output
        speed_window_size: Window size for moving average smoothing of speeds
    
    Returns:
        dict: Statistics about processing (counts, etc.)
    """
    raw_dir = Path(raw_data_dir)
    processed_dir = Path(processed_base_dir)
    
    if not raw_dir.exists():
        print(f"Error: Raw data directory does not exist: {raw_dir}")
        return {'processed': 0, 'failed': 0, 'skipped': 0}
    
    # Find all JSON files in raw directory
    json_files = sorted(list(raw_dir.glob("*.json")))
    
    if not json_files:
        print(f"No JSON files found in {raw_dir}")
        return {'processed': 0, 'failed': 0, 'skipped': 0}
    
    print(f"Found {len(json_files)} raw data files")
    print(f"Speed smoothing window size: {speed_window_size}")
    print(f"Output directory: {processed_dir}")
    
    stats = {'processed': 0, 'failed': 0, 'skipped': 0}
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            # Load file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract trialData array
            trial_data_array = data.get('trialData', [])
            
            if not trial_data_array:
                print(f"  ⚠ No trialData found in {json_file.name}")
                stats['skipped'] += 1
                continue
            
            print(f"  Found {len(trial_data_array)} trials in file")
            
            # Process each trial
            for trial_idx, trial_data in enumerate(trial_data_array):
                # Process the trial
                processed_data, trial_id = process_trial_from_raw_data(
                    trial_data,
                    speed_window_size
                )
                
                if processed_data is None or trial_id is None:
                    print(f"    ⚠ Skipped trial {trial_idx + 1}")
                    stats['skipped'] += 1
                    continue
                
                # Create trial-specific output directory
                trial_output_dir = processed_dir / str(trial_id)
                trial_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create output filename: participantId_trialId.json
                participant_id = data.get('participantId', 'unknown')
                output_filename = f"{participant_id}_trial_{trial_id}.json"
                output_file = trial_output_dir / output_filename
                
                # Save processed data
                output_data = {
                    'trajectory': processed_data['trajectory'],
                    'speed': processed_data['speed']
                }
                
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                point_count = len(processed_data['trajectory'])
                
                print(f"    ✓ Trial {trial_id}: {point_count} points")
                print(f"      Saved to: {output_file}")
                stats['processed'] += 1
                
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
            stats['failed'] += 1
            import traceback
            traceback.print_exc()
    
    return stats


def process_all_raw_data(raw_data_dir=None, processed_base_dir=None, speed_window_size=5):
    """
    Main function to process all raw human data.
    
    Args:
        raw_data_dir: Directory containing raw JSON files (default: human_data/raw/)
        processed_base_dir: Base directory for processed output (default: human_data/processed/)
        speed_window_size: Window size for moving average smoothing of speeds (default: 5)
    
    Returns:
        dict: Processing statistics
    """
    # Set default paths relative to this script
    base_dir = Path(__file__).parent
    
    if raw_data_dir is None:
        raw_data_dir = base_dir / "human_data" / "raw"
    
    if processed_base_dir is None:
        processed_base_dir = base_dir / "human_data" / "processed"
    
    print("="*60)
    print("Processing Raw Human Steering Data")
    print("="*60)
    print(f"Input directory: {raw_data_dir}")
    print(f"Output directory: {processed_base_dir}")
    print(f"Speed smoothing window size: {speed_window_size}")
    print("="*60)
    
    # Process all files
    stats = organize_by_trial(
        str(raw_data_dir),
        str(processed_base_dir),
        speed_window_size
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"  Successfully processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Output directory: {processed_base_dir}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    # Process all raw data
    stats = process_all_raw_data(speed_window_size=5)
    
    if stats['processed'] > 0:
        print("\n✓ Processing complete!")
    else:
        print("\n⚠ No files were processed. Check input directory and file format.")
        exit(1)

