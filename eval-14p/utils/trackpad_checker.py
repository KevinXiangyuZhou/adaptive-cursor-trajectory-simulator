"""
Trackpad Checker: Detect whether users are using mouse or trackpad based on raw movement data.

This module analyzes trajectory and speed characteristics to distinguish between trackpad and mouse input.
Trackpads typically exhibit:
- Smoother movements (lower jerk)
- More continuous motion (fewer zero-velocity periods)
- More consistent acceleration patterns
- Higher frequency of micro-movements
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add parent directory to path
project_root = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, project_root)

from process import extract_raw_trajectory, compute_speeds_from_trajectory


def compute_acceleration(trajectory: List, timestamps: List) -> np.ndarray:
    """
    Compute acceleration from trajectory and timestamps.
    
    Args:
        trajectory: List of [x, y] points
        timestamps: List of timestamps in milliseconds
    
    Returns:
        Array of acceleration magnitudes
    """
    if len(trajectory) < 3 or len(timestamps) < 3:
        return np.array([])
    
    # Compute velocities first
    velocities = []
    for i in range(len(trajectory) - 1):
        dx = trajectory[i + 1][0] - trajectory[i][0]
        dy = trajectory[i + 1][1] - trajectory[i][1]
        dt_ms = timestamps[i + 1] - timestamps[i]
        dt_sec = dt_ms / 1000.0 if dt_ms > 0 else 1e-6
        
        vx = dx / dt_sec
        vy = dy / dt_sec
        velocities.append([vx, vy])
    
    # Compute accelerations from velocities
    accelerations = []
    for i in range(len(velocities) - 1):
        dvx = velocities[i + 1][0] - velocities[i][0]
        dvy = velocities[i + 1][1] - velocities[i][1]
        dt_ms = timestamps[i + 2] - timestamps[i + 1]
        dt_sec = dt_ms / 1000.0 if dt_ms > 0 else 1e-6
        
        ax = dvx / dt_sec
        ay = dvy / dt_sec
        accel_mag = np.sqrt(ax**2 + ay**2)
        accelerations.append(accel_mag)
    
    return np.array(accelerations)


def compute_jerk(trajectory: List, timestamps: List) -> np.ndarray:
    """
    Compute jerk (rate of change of acceleration) from trajectory.
    
    Args:
        trajectory: List of [x, y] points
        timestamps: List of timestamps in milliseconds
    
    Returns:
        Array of jerk magnitudes
    """
    if len(trajectory) < 4 or len(timestamps) < 4:
        return np.array([])
    
    # Compute accelerations first
    accelerations = []
    velocities = []
    
    for i in range(len(trajectory) - 1):
        dx = trajectory[i + 1][0] - trajectory[i][0]
        dy = trajectory[i + 1][1] - trajectory[i][1]
        dt_ms = timestamps[i + 1] - timestamps[i]
        dt_sec = dt_ms / 1000.0 if dt_ms > 0 else 1e-6
        
        vx = dx / dt_sec
        vy = dy / dt_sec
        velocities.append([vx, vy])
    
    for i in range(len(velocities) - 1):
        dvx = velocities[i + 1][0] - velocities[i][0]
        dvy = velocities[i + 1][1] - velocities[i][1]
        dt_ms = timestamps[i + 2] - timestamps[i + 1]
        dt_sec = dt_ms / 1000.0 if dt_ms > 0 else 1e-6
        
        ax = dvx / dt_sec
        ay = dvy / dt_sec
        accelerations.append([ax, ay])
    
    # Compute jerk from accelerations
    jerks = []
    for i in range(len(accelerations) - 1):
        dax = accelerations[i + 1][0] - accelerations[i][0]
        day = accelerations[i + 1][1] - accelerations[i][1]
        dt_ms = timestamps[i + 3] - timestamps[i + 2]
        dt_sec = dt_ms / 1000.0 if dt_ms > 0 else 1e-6
        
        jx = dax / dt_sec
        jy = day / dt_sec
        jerk_mag = np.sqrt(jx**2 + jy**2)
        jerks.append(jerk_mag)
    
    return np.array(jerks)


def extract_features(trajectory: List, timestamps: List, speeds: Optional[List] = None) -> Dict:
    """
    Extract features from trajectory that help distinguish trackpad from mouse.
    
    Args:
        trajectory: List of [x, y] points
        timestamps: List of timestamps in milliseconds
        speeds: Optional pre-computed speeds array
    
    Returns:
        Dictionary of extracted features
    """
    if len(trajectory) < 3 or len(timestamps) < 3:
        return {}
    
    # Normalize trajectory format
    if trajectory and isinstance(trajectory[0], dict):
        trajectory = [[p['x'], p['y']] for p in trajectory]
    
    # Compute speeds if not provided
    if speeds is None:
        speeds = compute_speeds_from_trajectory(trajectory, timestamps, window_size=1)
    
    speeds = np.array(speeds)
    
    # Feature 1: Zero-velocity periods
    # For steering tasks: trackpads have more (finger repositioning), mice have fewer (continuous movement)
    zero_velocity_threshold = 0.001  # Very small speed threshold
    zero_velocity_ratio = np.sum(speeds < zero_velocity_threshold) / len(speeds) if len(speeds) > 0 else 0
    
    # Feature 2: Speed variance (trackpads have more consistent speeds due to continuous finger movement)
    speed_variance = np.var(speeds) if len(speeds) > 0 else 0
    speed_std = np.std(speeds) if len(speeds) > 0 else 0
    speed_mean = np.mean(speeds) if len(speeds) > 0 else 0
    speed_cv = speed_std / speed_mean if speed_mean > 0 else 0  # Coefficient of variation
    
    # Feature 3: Jerk (rate of change of acceleration) - trackpads have lower jerk (smoother finger movements)
    jerks = compute_jerk(trajectory, timestamps)
    mean_jerk = np.mean(jerks) if len(jerks) > 0 else 0
    std_jerk = np.std(jerks) if len(jerks) > 0 else 0
    max_jerk = np.max(jerks) if len(jerks) > 0 else 0
    
    # Feature 4: Acceleration smoothness (trackpads have lower variance - smoother acceleration changes)
    accelerations = compute_acceleration(trajectory, timestamps)
    accel_variance = np.var(accelerations) if len(accelerations) > 0 else 0
    accel_std = np.std(accelerations) if len(accelerations) > 0 else 0
    
    # Feature 5: Micro-movements (small movements < threshold) - trackpads have more fine-grained movements
    micro_movement_threshold = 0.01  # 1cm threshold
    movement_distances = []
    for i in range(len(trajectory) - 1):
        dx = trajectory[i + 1][0] - trajectory[i][0]
        dy = trajectory[i + 1][1] - trajectory[i][1]
        dist = np.sqrt(dx**2 + dy**2)
        movement_distances.append(dist)
    
    movement_distances = np.array(movement_distances)
    micro_movement_ratio = np.sum(movement_distances < micro_movement_threshold) / len(movement_distances) if len(movement_distances) > 0 else 0
    
    # Feature 6: Movement continuity (consecutive non-zero movements)
    # For steering tasks: mice have higher continuity (continuous steering), trackpads have lower (finger repositioning)
    non_zero_movements = movement_distances > 1e-6
    if len(non_zero_movements) > 0:
        # Count consecutive non-zero movements
        consecutive_non_zero = 0
        max_consecutive = 0
        current_streak = 0
        for is_moving in non_zero_movements:
            if is_moving:
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0
        movement_continuity = max_consecutive / len(non_zero_movements) if len(non_zero_movements) > 0 else 0
    else:
        movement_continuity = 0
    
    # Feature 7: Speed change rate (how quickly speed changes)
    # Trackpads have lower rate (smoother transitions), mice have higher (more abrupt changes)
    if len(speeds) > 1:
        speed_changes = np.abs(np.diff(speeds))
        mean_speed_change = np.mean(speed_changes) if len(speed_changes) > 0 else 0
        speed_change_rate = mean_speed_change / (speed_mean + 1e-6) if speed_mean > 0 else 0
    else:
        speed_change_rate = 0
    
    return {
        'zero_velocity_ratio': zero_velocity_ratio,
        'speed_variance': speed_variance,
        'speed_cv': speed_cv,
        'mean_jerk': mean_jerk,
        'std_jerk': std_jerk,
        'max_jerk': max_jerk,
        'accel_variance': accel_variance,
        'accel_std': accel_std,
        'micro_movement_ratio': micro_movement_ratio,
        'movement_continuity': movement_continuity,
        'speed_change_rate': speed_change_rate,
        'num_points': len(trajectory)
    }


def classify_device_type(features: Dict) -> Tuple[str, float]:
    """
    Classify device type (trackpad or mouse) based on features.
    
    Args:
        features: Dictionary of extracted features
    
    Returns:
        Tuple of (device_type, confidence_score)
        device_type: 'trackpad' or 'mouse'
        confidence_score: Float between 0 and 1 (higher = more confident)
    """
    if not features:
        return 'unknown', 0.0
    
    # Initialize scores (positive = trackpad, negative = mouse)
    trackpad_score = 0.0
    mouse_score = 0.0
    
    # Rule 1: Lower jerk suggests trackpad
    mean_jerk = features.get('mean_jerk', 0)
    if mean_jerk < 50:  # Threshold for low jerk
        trackpad_score += 2.0
    elif mean_jerk > 200:  # High jerk suggests mouse
        mouse_score += 2.0
    
    # Rule 2: For steering tasks, higher zero-velocity ratio suggests trackpad
    # (trackpads require finger lifting/repositioning, mice allow continuous movement)
    zero_vel_ratio = features.get('zero_velocity_ratio', 0)
    if zero_vel_ratio > 0.3:  # More than 30% zero velocity (finger repositioning)
        trackpad_score += 1.5
    elif zero_vel_ratio < 0.1:  # Less than 10% zero velocity (continuous movement)
        mouse_score += 1.5
    
    # Rule 3: Higher micro-movement ratio suggests trackpad
    micro_movement_ratio = features.get('micro_movement_ratio', 0)
    if micro_movement_ratio > 0.3:  # More than 30% micro-movements
        trackpad_score += 1.0
    elif micro_movement_ratio < 0.1:  # Less than 10% micro-movements
        mouse_score += 1.0
    
    # Rule 4: For steering tasks, higher movement continuity suggests mouse
    # (mice allow continuous steering without repositioning)
    movement_continuity = features.get('movement_continuity', 0)
    if movement_continuity > 0.5:  # High continuity (continuous steering)
        mouse_score += 1.0
    elif movement_continuity < 0.2:  # Low continuity (frequent repositioning)
        trackpad_score += 1.0
    
    # Rule 5: Lower speed change rate suggests trackpad (smoother)
    speed_change_rate = features.get('speed_change_rate', 0)
    if speed_change_rate < 0.5:  # Low change rate
        trackpad_score += 0.5
    elif speed_change_rate > 1.5:  # High change rate
        mouse_score += 0.5
    
    # Rule 6: Lower acceleration variance suggests trackpad (smoother)
    accel_variance = features.get('accel_variance', 0)
    if accel_variance < 0.1:  # Low variance
        trackpad_score += 0.5
    elif accel_variance > 1.0:  # High variance
        mouse_score += 0.5
    
    # Determine device type
    total_score = trackpad_score + mouse_score
    if total_score == 0:
        return 'unknown', 0.0
    
    if trackpad_score > mouse_score:
        confidence = trackpad_score / total_score
        return 'trackpad', confidence
    else:
        confidence = mouse_score / total_score
        return 'mouse', confidence


def check_participant(raw_data_file: Path) -> Dict:
    """
    Check a single participant's data file to determine device type.
    
    Args:
        raw_data_file: Path to raw JSON data file
    
    Returns:
        Dictionary with participant_id, device_type, confidence, and features
    """
    try:
        with open(raw_data_file, 'r') as f:
            data = json.load(f)
        
        participant_id = data.get('participantId', 'unknown')
        
        # Extract sessions
        sessions = data.get('sessions', [])
        if not sessions:
            trial_data_array = data.get('trialData', [])
            sessions = [{'trialData': trial_data_array}] if trial_data_array else []
        
        # Collect features from all trials
        all_features = []
        
        for session in sessions:
            trial_data_array = session.get('trialData', [])
            
            for trial_data in trial_data_array:
                trajectory = trial_data.get('trajectory', [])
                timestamps = trial_data.get('timestamps', [])
                speeds = trial_data.get('speeds', None)
                
                if not trajectory or not timestamps:
                    continue
                
                # Extract features from this trial
                features = extract_features(trajectory, timestamps, speeds)
                if features:
                    all_features.append(features)
        
        if not all_features:
            return {
                'participant_id': participant_id,
                'device_type': 'unknown',
                'confidence': 0.0,
                'num_trials': 0,
                'features': None
            }
        
        # Aggregate features across all trials (use mean)
        aggregated_features = {}
        for key in all_features[0].keys():
            if key == 'num_points':
                aggregated_features[key] = sum(f[key] for f in all_features)
            else:
                aggregated_features[key] = np.mean([f[key] for f in all_features])
        
        # Classify device type
        device_type, confidence = classify_device_type(aggregated_features)
        
        return {
            'participant_id': participant_id,
            'device_type': device_type,
            'confidence': confidence,
            'num_trials': len(all_features),
            'features': aggregated_features
        }
        
    except Exception as e:
        return {
            'participant_id': raw_data_file.stem if raw_data_file else 'unknown',
            'device_type': 'error',
            'confidence': 0.0,
            'error': str(e),
            'num_trials': 0,
            'features': None
        }


def check_all_participants(raw_data_dir: Optional[Path] = None) -> List[Dict]:
    """
    Check all participants in a directory to determine their device types.
    
    Args:
        raw_data_dir: Directory containing raw JSON files (default: human_data/raw/)
    
    Returns:
        List of dictionaries with participant results
    """
    if raw_data_dir is None:
        base_dir = Path(__file__).parent
        raw_data_dir = base_dir / "human_data" / "raw"
    
    raw_data_dir = Path(raw_data_dir)
    
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")
    
    json_files = sorted(list(raw_data_dir.glob("*.json")))
    
    if not json_files:
        print(f"No JSON files found in {raw_data_dir}")
        return []
    
    results = []
    for json_file in json_files:
        result = check_participant(json_file)
        results.append(result)
    
    return results


def print_results(results: List[Dict]):
    """
    Print formatted results of device type detection.
    
    Args:
        results: List of result dictionaries from check_all_participants
    """
    print("=" * 80)
    print("Trackpad vs Mouse Detection Results")
    print("=" * 80)
    print(f"{'Participant ID':<40} {'Device Type':<15} {'Confidence':<12} {'Trials':<8}")
    print("-" * 80)
    
    trackpad_count = 0
    mouse_count = 0
    unknown_count = 0
    
    for result in results:
        participant_id = result.get('participant_id', 'unknown')
        device_type = result.get('device_type', 'unknown')
        confidence = result.get('confidence', 0.0)
        num_trials = result.get('num_trials', 0)
        
        # Truncate participant ID if too long
        display_id = participant_id[:37] + "..." if len(participant_id) > 40 else participant_id
        
        print(f"{display_id:<40} {device_type:<15} {confidence:<12.3f} {num_trials:<8}")
        
        if device_type == 'trackpad':
            trackpad_count += 1
        elif device_type == 'mouse':
            mouse_count += 1
        else:
            unknown_count += 1
    
    print("-" * 80)
    print(f"Summary: {len(results)} participants")
    print(f"  Trackpad: {trackpad_count}")
    print(f"  Mouse: {mouse_count}")
    print(f"  Unknown/Error: {unknown_count}")
    print("=" * 80)


def get_device_type_for_participant(participant_id: str, raw_data_dir: Optional[Path] = None) -> Optional[str]:
    """
    Get device type for a specific participant.
    
    Args:
        participant_id: Participant ID to look up
        raw_data_dir: Directory containing raw JSON files (default: human_data/raw/)
    
    Returns:
        Device type ('trackpad', 'mouse', or None if not found)
    """
    if raw_data_dir is None:
        base_dir = Path(__file__).parent
        raw_data_dir = base_dir / "human_data" / "raw"
    
    raw_data_dir = Path(raw_data_dir)
    json_files = list(raw_data_dir.glob("*.json"))
    
    for json_file in json_files:
        result = check_participant(json_file)
        if result.get('participant_id') == participant_id:
            return result.get('device_type')
    
    return None


if __name__ == "__main__":
    # Check all participants in the default directory
    results = check_all_participants()
    print_results(results)
    
    # Optionally save results to JSON
    output_file = Path(__file__).parent / "device_type_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
