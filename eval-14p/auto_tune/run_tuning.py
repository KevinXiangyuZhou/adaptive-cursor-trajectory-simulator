"""
Automatic Parameter Tuning for Cursor Steering Model.

Tunes parameters in customized.json to match human behavior in:
1. Steering Law experiment (speed adaptation to tunnel width)
2. Stop-and-Go experiment (speed variability at corners)

Uses Bayesian optimization (Optuna) with joint tuning of 6 parameters:
- Steering Law: desired_speed, speed_alpha, speed_floor, speed_ceil
- Stop-and-Go: rate_percentile, rate_alpha

Fixed parameters (not tuned): speed_reference, rate_scale, rate_floor, rate_median_max

Usage:
    python run_tuning.py [--trials N] [--save]
"""

import argparse
import json
import os
import sys
import copy
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

# Try to import optuna
OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "utils"))

DEFAULT_CONFIG = PROJECT_ROOT / "experiment" / "user_configurations" / "customized.json"
STEERING_LAW_SCRIPT = PROJECT_ROOT / "eval" / "experiment-steering-law" / "run_eval.py"
STOP_AND_GO_SCRIPT = PROJECT_ROOT / "eval" / "experiment-stop-and-go" / "run_eval.py"
STRATEGY_SCRIPT = PROJECT_ROOT / "eval" / "experiment-strategy" / "run_eval.py"

# Human data directories
STEERING_LAW_DATA_DIR = PROJECT_ROOT / "eval" / "human_data" / "raw"
STOP_AND_GO_DATA_DIR = PROJECT_ROOT / "eval" / "human_data" / "raw"

# Default trackpad confidence threshold
DEFAULT_TRACKPAD_CONFIDENCE = 0  # Set to 0 to disable filtering, >0 to filter by trackpad

# ---------------------------------------------------------------------------
# Human Target Computation (computed dynamically from human data)
# ---------------------------------------------------------------------------

# Global variable to store computed human targets
HUMAN_TARGETS = None


def _compute_speed_cv(speeds):
    """Coefficient of variation of speed (std/mean)."""
    if not speeds or len(speeds) < 2:
        return 0.0
    speeds_arr = np.array(speeds)
    mean_spd = np.mean(speeds_arr)
    if mean_spd <= 0:
        return 0.0
    return float(np.std(speeds_arr) / mean_spd)


def _compute_min_speed_ratio(speeds):
    """Ratio of minimum speed to mean speed."""
    if not speeds or len(speeds) < 2:
        return 1.0
    speeds_arr = np.array(speeds)
    mean_spd = np.mean(speeds_arr)
    min_spd = np.min(speeds_arr)
    if mean_spd <= 0:
        return 1.0
    return float(min_spd / mean_spd)


def _get_trackpad_participants(data_dir, min_confidence=0.7):
    """Get set of participant IDs that are trackpad users."""
    try:
        from trackpad_checker import check_participant
    except ImportError:
        print("Warning: trackpad_checker not available, using all participants")
        return None
    
    trackpad_pids = set()
    for fpath in sorted(data_dir.glob("*.json")):
        result = check_participant(fpath)
        if (result.get("device_type") == "trackpad" and 
            result.get("confidence", 0) >= min_confidence):
            trackpad_pids.add(result.get("participant_id"))
    return trackpad_pids


def compute_human_targets(trackpad_confidence=None):
    """
    Compute human target metrics from actual human data.
    
    Returns dict with structure:
        {
            "steering_law": {
                "mt_ratio_wide_narrow": float,
                "speed_ratio_wide_narrow": float,
            },
            "stop_and_go": {
                "speed_cv_diff_narrow": float,
                "speed_cv_diff_wide": float,
                "min_ratio_corner_narrow": float,
                "min_ratio_corner_wide": float,
            },
            "speed_stats": {
                "global_min": float,
                "global_max": float,
                "global_mean": float,
                "speed_floor_ratio": float,  # min/mean
                "speed_ceil_ratio": float,   # max/mean
            }
        }
    """
    targets = {
        "steering_law": {},
        "stop_and_go": {},
        "speed_stats": {},
    }
    
    # Collect all speeds across all trials for computing speed_floor/speed_ceil ranges
    all_speeds = []
    
    # Get trackpad participants
    trackpad_pids = None
    if trackpad_confidence is not None and trackpad_confidence > 0:
        trackpad_pids = _get_trackpad_participants(STEERING_LAW_DATA_DIR, trackpad_confidence)
        if trackpad_pids:
            print(f"  Filtering to {len(trackpad_pids)} trackpad participants (confidence >= {trackpad_confidence})")
    
    # -------------------------------------------------------------------------
    # Steering Law targets (trial_id 1 & 2: wave tunnels with W=0.02 and W=0.04)
    # -------------------------------------------------------------------------
    steering_records = []
    for fpath in sorted(STEERING_LAW_DATA_DIR.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        pid = data.get("participantId", fpath.stem)
        
        if trackpad_pids is not None and pid not in trackpad_pids:
            continue
        
        sessions = data.get("sessions", [])
        trial_list = sessions[0].get("trialData", []) if sessions else data.get("trialData", [])
        
        for trial in trial_list:
            tid = trial.get("trial_id")
            if tid not in [1, 2]:  # Only wave tunnel trials
                continue
            
            cond = trial.get("condition", {})
            width = cond.get("tunnelWidth", 0.02 if tid == 1 else 0.04)
            traj_raw = trial.get("trajectory", [])
            timestamps = trial.get("timestamps", [])
            ct = trial.get("completionTime", 0.0)
            
            if not traj_raw or not timestamps or ct <= 0:
                continue
            
            # Compute path length and speeds
            traj = [[p["x"], p["y"]] if isinstance(p, dict) else p for p in traj_raw]
            pl = 0.0
            trial_speeds = []
            for i in range(1, len(traj)):
                dx = traj[i][0] - traj[i - 1][0]
                dy = traj[i][1] - traj[i - 1][1]
                dist = (dx * dx + dy * dy) ** 0.5
                pl += dist
                if i < len(timestamps):
                    dt = (timestamps[i] - timestamps[i-1]) / 1000.0
                    if dt > 0:
                        trial_speeds.append(dist / dt)
            
            # Collect speeds for global statistics
            all_speeds.extend(trial_speeds)
            
            steering_records.append({
                "width": width,
                "completion_time": ct,
                "avg_speed": pl / ct if ct > 0 else 0.0,
            })
    
    # Compute steering law targets
    by_width = defaultdict(list)
    for r in steering_records:
        by_width[r["width"]].append(r)
    
    narrow_mt = np.mean([r["completion_time"] for r in by_width.get(0.02, [])]) if by_width.get(0.02) else 10.0
    wide_mt = np.mean([r["completion_time"] for r in by_width.get(0.04, [])]) if by_width.get(0.04) else 5.0
    narrow_speed = np.mean([r["avg_speed"] for r in by_width.get(0.02, [])]) if by_width.get(0.02) else 0.06
    wide_speed = np.mean([r["avg_speed"] for r in by_width.get(0.04, [])]) if by_width.get(0.04) else 0.11
    
    targets["steering_law"]["mt_ratio_wide_narrow"] = wide_mt / narrow_mt if narrow_mt > 0 else 0.5
    targets["steering_law"]["speed_ratio_wide_narrow"] = wide_speed / narrow_speed if narrow_speed > 0 else 1.84
    
    # -------------------------------------------------------------------------
    # Stop-and-Go targets (trial_id 1-4: sigmoidal and corner tunnels)
    # -------------------------------------------------------------------------
    # Trial conditions: 1=narrow sigmoidal, 2=wide sigmoidal, 3=narrow corner, 4=wide corner
    STOP_GO_CONDITIONS = {
        1: {"width": 0.02, "type": "sigmoidal"},
        2: {"width": 0.04, "type": "sigmoidal"},
        3: {"width": 0.02, "type": "corner"},
        4: {"width": 0.04, "type": "corner"},
    }
    
    stopgo_records = []
    for fpath in sorted(STOP_AND_GO_DATA_DIR.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        pid = data.get("participantId", fpath.stem)
        
        if trackpad_pids is not None and pid not in trackpad_pids:
            continue
        
        sessions = data.get("sessions", [])
        trial_list = sessions[0].get("trialData", []) if sessions else data.get("trialData", [])
        
        for trial in trial_list:
            tid = trial.get("trial_id")
            if tid not in STOP_GO_CONDITIONS:
                continue
            
            cond_info = STOP_GO_CONDITIONS[tid]
            traj_raw = trial.get("trajectory", [])
            timestamps = trial.get("timestamps", [])
            
            if not traj_raw or not timestamps or len(traj_raw) < 5:
                continue
            
            # Compute speeds
            traj = [[p["x"], p["y"]] if isinstance(p, dict) else p for p in traj_raw]
            speeds = []
            for i in range(1, len(traj)):
                if i >= len(timestamps):
                    break
                dt = (timestamps[i] - timestamps[i-1]) / 1000.0
                if dt > 0:
                    dx = traj[i][0] - traj[i - 1][0]
                    dy = traj[i][1] - traj[i - 1][1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    speeds.append(dist / dt)
            
            if len(speeds) < 5:
                continue
            
            # Collect speeds for global statistics
            all_speeds.extend(speeds)
            
            stopgo_records.append({
                "width": cond_info["width"],
                "tunnel_type": cond_info["type"],
                "speed_cv": _compute_speed_cv(speeds),
                "min_speed_ratio": _compute_min_speed_ratio(speeds),
            })
    
    # Compute stop-and-go targets by condition
    by_condition = defaultdict(list)
    for r in stopgo_records:
        key = (r["width"], r["tunnel_type"])
        by_condition[key].append(r)
    
    # SpeedCV differences
    sig_narrow_cv = np.mean([r["speed_cv"] for r in by_condition.get((0.02, "sigmoidal"), [])]) if by_condition.get((0.02, "sigmoidal")) else 0.44
    cor_narrow_cv = np.mean([r["speed_cv"] for r in by_condition.get((0.02, "corner"), [])]) if by_condition.get((0.02, "corner")) else 0.62
    sig_wide_cv = np.mean([r["speed_cv"] for r in by_condition.get((0.04, "sigmoidal"), [])]) if by_condition.get((0.04, "sigmoidal")) else 0.45
    cor_wide_cv = np.mean([r["speed_cv"] for r in by_condition.get((0.04, "corner"), [])]) if by_condition.get((0.04, "corner")) else 0.60
    
    targets["stop_and_go"]["speed_cv_diff_narrow"] = cor_narrow_cv - sig_narrow_cv
    targets["stop_and_go"]["speed_cv_diff_wide"] = cor_wide_cv - sig_wide_cv
    
    # MinSpeedRatio at corners
    targets["stop_and_go"]["min_ratio_corner_narrow"] = np.mean([r["min_speed_ratio"] for r in by_condition.get((0.02, "corner"), [])]) if by_condition.get((0.02, "corner")) else 0.016
    targets["stop_and_go"]["min_ratio_corner_wide"] = np.mean([r["min_speed_ratio"] for r in by_condition.get((0.04, "corner"), [])]) if by_condition.get((0.04, "corner")) else 0.027
    
    # -------------------------------------------------------------------------
    # Speed statistics for computing parameter ranges
    # Use percentiles to avoid outliers (e.g., instantaneous zero speeds)
    # -------------------------------------------------------------------------
    if all_speeds:
        all_speeds_arr = np.array(all_speeds)
        # Filter out very small speeds (likely measurement artifacts)
        valid_speeds = all_speeds_arr[all_speeds_arr > 0.001]
        
        if len(valid_speeds) > 0:
            # Use 5th percentile for floor, 95th for ceil to avoid outliers
            global_min = float(np.percentile(valid_speeds, 5))
            global_max = float(np.percentile(valid_speeds, 95))
            global_mean = float(np.mean(valid_speeds))
            
            targets["speed_stats"]["global_min"] = global_min
            targets["speed_stats"]["global_max"] = global_max
            targets["speed_stats"]["global_mean"] = global_mean
            targets["speed_stats"]["speed_floor_ratio"] = global_min / global_mean if global_mean > 0 else 0.1
            targets["speed_stats"]["speed_ceil_ratio"] = global_max / global_mean if global_mean > 0 else 3.0
        else:
            # Fallback defaults
            targets["speed_stats"]["global_min"] = 0.01
            targets["speed_stats"]["global_max"] = 0.3
            targets["speed_stats"]["global_mean"] = 0.08
            targets["speed_stats"]["speed_floor_ratio"] = 0.1
            targets["speed_stats"]["speed_ceil_ratio"] = 3.0
    else:
        # Fallback defaults
        targets["speed_stats"]["global_min"] = 0.01
        targets["speed_stats"]["global_max"] = 0.3
        targets["speed_stats"]["global_mean"] = 0.08
        targets["speed_stats"]["speed_floor_ratio"] = 0.1
        targets["speed_stats"]["speed_ceil_ratio"] = 3.0
    
    return targets


# ---------------------------------------------------------------------------
# Config Management
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def save_config(config, config_path):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def update_config_params(base_config, params):
    """Update planner_weights with new parameters."""
    config = copy.deepcopy(base_config)
    config["planner_weights"].update(params)
    return config


# ---------------------------------------------------------------------------
# Run Experiments (subprocess to avoid state pollution)
# ---------------------------------------------------------------------------

def run_steering_law_experiment(config_path, rounds=3):
    """Run steering law experiment and parse results."""
    import re
    
    result = subprocess.run(
        [sys.executable, str(STEERING_LAW_SCRIPT), 
         "--config", str(config_path), 
         "--rounds", str(rounds)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONWARNINGS": "ignore"}
    )
    
    output = result.stdout
    metrics = {}
    
    # Parse output format:
    # │ W=0.02    │  MT=10.56s  v=0.058   MT= 6.75s  v=0.084  │
    # │ W=0.04    │  MT= 5.12s  v=0.109   MT= 3.60s  v=0.164  │
    # Model values are the second MT/v pair
    
    for line in output.split('\n'):
        if 'W=0.02' in line and '│' in line:
            # Narrow tunnel - extract second MT and v
            mt_match = re.findall(r'MT=\s*([\d.]+)s', line)
            v_match = re.findall(r'v=([\d.]+)', line)
            if len(mt_match) >= 2 and len(v_match) >= 2:
                metrics['mt_narrow'] = float(mt_match[1])  # model (second)
                metrics['speed_narrow'] = float(v_match[1])
        elif 'W=0.04' in line and '│' in line:
            # Wide tunnel
            mt_match = re.findall(r'MT=\s*([\d.]+)s', line)
            v_match = re.findall(r'v=([\d.]+)', line)
            if len(mt_match) >= 2 and len(v_match) >= 2:
                metrics['mt_wide'] = float(mt_match[1])
                metrics['speed_wide'] = float(v_match[1])
    
    return metrics, output


def run_stop_and_go_experiment(config_path, rounds=3):
    """Run stop-and-go experiment and parse results."""
    import re
    
    result = subprocess.run(
        [sys.executable, str(STOP_AND_GO_SCRIPT),
         "--config", str(config_path),
         "--rounds", str(rounds)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONWARNINGS": "ignore"}
    )
    
    output = result.stdout
    metrics = {}
    
    # Parse output format:
    # │ sigmoidal W=20mm        │   0.444    0.035   4.3 │  0.397    0.182   2.0    │
    # │ sigmoidal W=40mm        │   0.445    0.048   1.9 │  0.463    0.090   1.0    │
    # │ corner W=20mm           │   0.621    0.016   6.9 │  0.621    0.135   6.0    │
    # │ corner W=40mm           │   0.601    0.027   3.5 │  0.443    0.161   2.0    │
    # Columns: Condition | Human (SpeedCV MinRatio Dips) | Model (SpeedCV MinRatio Dips)
    
    for line in output.split('\n'):
        if '│' in line:
            parts = line.split('│')
            if len(parts) >= 4:
                condition = parts[1].strip().lower()
                model_part = parts[3].strip()
                
                # Extract model SpeedCV and MinRatio
                numbers = re.findall(r'[\d.]+', model_part)
                if len(numbers) >= 2:
                    try:
                        speed_cv = float(numbers[0])
                        min_ratio = float(numbers[1])
                        
                        if 'sigmoidal' in condition and '20' in condition:
                            metrics['speed_cv_sig_narrow'] = speed_cv
                            metrics['min_ratio_sig_narrow'] = min_ratio
                        elif 'sigmoidal' in condition and '40' in condition:
                            metrics['speed_cv_sig_wide'] = speed_cv
                            metrics['min_ratio_sig_wide'] = min_ratio
                        elif 'corner' in condition and '20' in condition:
                            metrics['speed_cv_cor_narrow'] = speed_cv
                            metrics['min_ratio_cor_narrow'] = min_ratio
                        elif 'corner' in condition and '40' in condition:
                            metrics['speed_cv_cor_wide'] = speed_cv
                            metrics['min_ratio_cor_wide'] = min_ratio
                    except (ValueError, IndexError):
                        pass
    
    return metrics, output


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def steering_law_loss(metrics):
    """
    Compute loss for steering law behavior.
    
    Targets:
    - Speed ratio (wide/narrow) should match human (~1.84)
    - MT ratio (wide/narrow) should match human (~0.5)
    
    Uses relative error to make loss scale-invariant.
    """
    loss = 0.0
    
    if 'speed_wide' in metrics and 'speed_narrow' in metrics:
        speed_ratio = metrics['speed_wide'] / max(metrics['speed_narrow'], 0.001)
        target_ratio = HUMAN_TARGETS["steering_law"]["speed_ratio_wide_narrow"]
        # Relative error squared
        rel_error = (speed_ratio - target_ratio) / target_ratio
        loss += rel_error ** 2
    else:
        loss += 10.0  # Penalty for missing data
    
    if 'mt_wide' in metrics and 'mt_narrow' in metrics:
        mt_ratio = metrics['mt_wide'] / max(metrics['mt_narrow'], 0.001)
        target_ratio = HUMAN_TARGETS["steering_law"]["mt_ratio_wide_narrow"]
        # Relative error squared
        rel_error = (mt_ratio - target_ratio) / target_ratio
        loss += rel_error ** 2
    else:
        loss += 10.0
    
    return loss


def stop_and_go_loss(metrics):
    """
    Compute loss for stop-and-go behavior.
    
    Targets:
    - SpeedCV difference (corner - sigmoidal) should match human
    - Both narrow and wide should show positive difference
    """
    loss = 0.0
    targets = HUMAN_TARGETS["stop_and_go"]
    
    # SpeedCV difference for narrow tunnels (target: 0.176)
    if 'speed_cv_cor_narrow' in metrics and 'speed_cv_sig_narrow' in metrics:
        cv_diff = metrics['speed_cv_cor_narrow'] - metrics['speed_cv_sig_narrow']
        target = targets["speed_cv_diff_narrow"]
        # Asymmetric loss: penalize undershooting more than overshooting
        if cv_diff < 0:
            loss += (cv_diff - target) ** 2 * 4  # Stronger penalty for negative
        else:
            loss += (cv_diff - target) ** 2
    else:
        loss += 5.0
    
    # SpeedCV difference for wide tunnels (target: 0.156)
    if 'speed_cv_cor_wide' in metrics and 'speed_cv_sig_wide' in metrics:
        cv_diff = metrics['speed_cv_cor_wide'] - metrics['speed_cv_sig_wide']
        target = targets["speed_cv_diff_wide"]
        if cv_diff < 0:
            loss += (cv_diff - target) ** 2 * 4  # Stronger penalty for negative
        else:
            loss += (cv_diff - target) ** 2
    else:
        loss += 5.0
    
    # MinSpeedRatio at corners - be less strict, just ensure it's reasonably low
    if 'min_ratio_cor_narrow' in metrics:
        target = targets["min_ratio_corner_narrow"]
        loss += (metrics['min_ratio_cor_narrow'] - target) ** 2 * 10
    
    if 'min_ratio_cor_wide' in metrics:
        target = targets["min_ratio_corner_wide"]
        loss += (metrics['min_ratio_cor_wide'] - target) ** 2 * 10
    
    return loss


# ---------------------------------------------------------------------------
# Optimization Objectives
# ---------------------------------------------------------------------------

def create_phase1_objective(base_config, temp_dir):
    """Create objective function for Phase 1 (Steering Law).
    
    Tunes: desired_speed, speed_alpha, speed_floor, speed_ceil
    Fixed: speed_reference (uses value from base_config)
    """
    # Get speed ranges from human data statistics
    speed_stats = HUMAN_TARGETS.get("speed_stats", {})
    floor_ratio = speed_stats.get("speed_floor_ratio", 0.1)
    ceil_ratio = speed_stats.get("speed_ceil_ratio", 3.0)
    
    # Ensure reasonable bounds
    floor_min = max(0.05, floor_ratio * 0.5)
    floor_max = min(0.5, floor_ratio * 2.0)
    ceil_min = max(1.0, ceil_ratio * 0.5)
    ceil_max = min(5.0, ceil_ratio * 2.0)
    
    def objective(trial):
        # Sample parameters (only the ones we're tuning)
        params = {
            "desired_speed": trial.suggest_float("desired_speed", 0.08, 0.18),
            "speed_alpha": trial.suggest_float("speed_alpha", 0.3, 0.9),
            "speed_floor": trial.suggest_float("speed_floor", floor_min, floor_max),
            "speed_ceil": trial.suggest_float("speed_ceil", ceil_min, ceil_max),
        }
        
        # Create temp config
        config = update_config_params(base_config, params)
        config_path = Path(temp_dir) / f"config_trial_{trial.number}.json"
        save_config(config, config_path)
        
        # Run experiment with 5 rounds for stability
        try:
            metrics, _ = run_steering_law_experiment(config_path, rounds=5)
            loss = steering_law_loss(metrics)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            loss = 100.0
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
        
        return loss
    
    return objective


def create_phase2_objective(base_config, temp_dir):
    """Create objective function for Phase 2 (Stop-and-Go).
    
    Tunes: rate_percentile, rate_alpha
    Fixed: rate_scale, rate_floor, rate_median_max (uses values from base_config)
    """
    
    def objective(trial):
        # Sample parameters (only the ones we're tuning)
        params = {
            "rate_percentile": trial.suggest_float("rate_percentile", 50.0, 95.0),
            "rate_alpha": trial.suggest_float("rate_alpha", 0.5, 2.0),
        }
        
        # Create temp config
        config = update_config_params(base_config, params)
        config_path = Path(temp_dir) / f"config_trial_{trial.number}.json"
        save_config(config, config_path)
        
        # Run experiment with 5 rounds for stability
        try:
            metrics, _ = run_stop_and_go_experiment(config_path, rounds=5)
            loss = stop_and_go_loss(metrics)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            loss = 100.0
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
        
        return loss
    
    return objective


def create_combined_objective(base_config, temp_dir):
    """Create objective function for combined optimization.
    
    Tunes all 6 parameters jointly:
    - desired_speed, speed_alpha, speed_floor, speed_ceil (steering law)
    - rate_percentile, rate_alpha (stop-and-go)
    """
    # Get speed ranges from human data statistics
    speed_stats = HUMAN_TARGETS.get("speed_stats", {})
    floor_ratio = speed_stats.get("speed_floor_ratio", 0.1)
    ceil_ratio = speed_stats.get("speed_ceil_ratio", 3.0)
    
    # Ensure reasonable bounds
    floor_min = max(0.05, floor_ratio * 0.5)
    floor_max = min(0.5, floor_ratio * 2.0)
    ceil_min = max(1.0, ceil_ratio * 0.5)
    ceil_max = min(5.0, ceil_ratio * 2.0)
    
    def objective(trial):
        # Sample all tunable parameters
        params = {
            # Steering law
            "desired_speed": trial.suggest_float("desired_speed", 0.08, 0.18),
            "speed_alpha": trial.suggest_float("speed_alpha", 0.3, 0.9),
            "speed_floor": trial.suggest_float("speed_floor", floor_min, floor_max),
            "speed_ceil": trial.suggest_float("speed_ceil", ceil_min, ceil_max),
            # Stop-and-go
            "rate_percentile": trial.suggest_float("rate_percentile", 50.0, 95.0),
            "rate_alpha": trial.suggest_float("rate_alpha", 0.5, 2.0),
        }
        
        # Create temp config
        config = update_config_params(base_config, params)
        config_path = Path(temp_dir) / f"config_trial_{trial.number}.json"
        save_config(config, config_path)
        
        try:
            # Run both experiments with 5 rounds each
            steering_metrics, _ = run_steering_law_experiment(config_path, rounds=5)
            stop_go_metrics, _ = run_stop_and_go_experiment(config_path, rounds=5)
            
            # Combined loss (weighted)
            loss = steering_law_loss(steering_metrics) + stop_and_go_loss(stop_go_metrics)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            loss = 200.0
        finally:
            if config_path.exists():
                config_path.unlink()
        
        return loss
    
    return objective


# ---------------------------------------------------------------------------
# Main Tuning Functions
# ---------------------------------------------------------------------------

def run_phase1_tuning(base_config, n_trials=30):
    """Phase 1: Tune steering law parameters."""
    print("\n" + "=" * 70)
    print("Phase 1: Tuning Steering Law Parameters")
    print("=" * 70)
    print("Parameters: desired_speed, speed_alpha, speed_floor, speed_ceil")
    print("Fixed: speed_reference")
    print(f"Trials: {n_trials}")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42)
        )
        
        objective = create_phase1_objective(base_config, temp_dir)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n--- Phase 1 Results ---")
    print(f"Best loss: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")
    
    return study.best_params


def run_phase2_tuning(base_config, n_trials=30):
    """Phase 2: Tune curvature rate parameters."""
    print("\n" + "=" * 70)
    print("Phase 2: Tuning Curvature Rate (Stop-and-Go) Parameters")
    print("=" * 70)
    print("Parameters: rate_percentile, rate_alpha")
    print("Fixed: rate_scale, rate_floor, rate_median_max")
    print(f"Trials: {n_trials}")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42)
        )
        
        objective = create_phase2_objective(base_config, temp_dir)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n--- Phase 2 Results ---")
    print(f"Best loss: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")
    
    return study.best_params


def run_combined_tuning(base_config, n_trials=50):
    """Combined: Tune all parameters together."""
    print("\n" + "=" * 70)
    print("Combined Tuning: All Tunable Parameters")
    print("=" * 70)
    print("Parameters: desired_speed, speed_alpha, speed_floor, speed_ceil,")
    print("            rate_percentile, rate_alpha")
    print("Fixed: speed_reference, rate_scale, rate_floor, rate_median_max")
    print(f"Trials: {n_trials}")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42)
        )
        
        objective = create_combined_objective(base_config, temp_dir)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n--- Combined Tuning Results ---")
    print(f"Best loss: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")
    
    return study.best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_hierarchical_tuning(base_config, phase1_trials=20, phase2_trials=20):
    """Run hierarchical tuning: Phase 1 first, then Phase 2 with Phase 1 results."""
    print("\n" + "=" * 70)
    print("Hierarchical Tuning: Phase 1 -> Phase 2")
    print("=" * 70)
    
    # Phase 1: Steering Law
    phase1_params = run_phase1_tuning(base_config, phase1_trials)
    
    # Update config with Phase 1 results for Phase 2
    config_with_phase1 = update_config_params(base_config, phase1_params)
    
    # Phase 2: Stop-and-Go (using Phase 1 results)
    phase2_params = run_phase2_tuning(config_with_phase1, phase2_trials)
    
    # Combine results
    all_params = {**phase1_params, **phase2_params}
    
    print("\n" + "=" * 70)
    print("Hierarchical Tuning Complete")
    print("=" * 70)
    print("Combined best parameters:")
    for k, v in all_params.items():
        print(f"  {k}: {v:.4f}")
    
    return all_params


def test_current_config(config_path):
    """Quick test of current config to see metrics."""
    print("\n" + "=" * 70)
    print("Testing Current Configuration")
    print("=" * 70)
    
    print("\nRunning Steering Law experiment...")
    steering_metrics, _ = run_steering_law_experiment(config_path, rounds=2)
    
    print("\nSteering Law Metrics:")
    if 'speed_wide' in steering_metrics and 'speed_narrow' in steering_metrics:
        ratio = steering_metrics['speed_wide'] / steering_metrics['speed_narrow']
        print(f"  Speed (narrow): {steering_metrics.get('speed_narrow', 'N/A'):.3f}")
        print(f"  Speed (wide):   {steering_metrics.get('speed_wide', 'N/A'):.3f}")
        print(f"  Speed ratio:    {ratio:.3f} (target: {HUMAN_TARGETS['steering_law']['speed_ratio_wide_narrow']})")
    
    if 'mt_wide' in steering_metrics and 'mt_narrow' in steering_metrics:
        mt_ratio = steering_metrics['mt_wide'] / steering_metrics['mt_narrow']
        print(f"  MT (narrow):    {steering_metrics.get('mt_narrow', 'N/A'):.2f}s")
        print(f"  MT (wide):      {steering_metrics.get('mt_wide', 'N/A'):.2f}s")
        print(f"  MT ratio:       {mt_ratio:.3f} (target: {HUMAN_TARGETS['steering_law']['mt_ratio_wide_narrow']})")
    
    steering_loss = steering_law_loss(steering_metrics)
    print(f"  Steering Law Loss: {steering_loss:.4f}")
    
    print("\nRunning Stop-and-Go experiment...")
    stopgo_metrics, _ = run_stop_and_go_experiment(config_path, rounds=2)
    
    print("\nStop-and-Go Metrics:")
    if 'speed_cv_sig_narrow' in stopgo_metrics and 'speed_cv_cor_narrow' in stopgo_metrics:
        diff_narrow = stopgo_metrics['speed_cv_cor_narrow'] - stopgo_metrics['speed_cv_sig_narrow']
        print(f"  SpeedCV (sigmoidal W=20): {stopgo_metrics['speed_cv_sig_narrow']:.3f}")
        print(f"  SpeedCV (corner W=20):    {stopgo_metrics['speed_cv_cor_narrow']:.3f}")
        print(f"  Difference:               {diff_narrow:.3f} (target: {HUMAN_TARGETS['stop_and_go']['speed_cv_diff_narrow']})")
    
    if 'speed_cv_sig_wide' in stopgo_metrics and 'speed_cv_cor_wide' in stopgo_metrics:
        diff_wide = stopgo_metrics['speed_cv_cor_wide'] - stopgo_metrics['speed_cv_sig_wide']
        print(f"  SpeedCV (sigmoidal W=40): {stopgo_metrics['speed_cv_sig_wide']:.3f}")
        print(f"  SpeedCV (corner W=40):    {stopgo_metrics['speed_cv_cor_wide']:.3f}")
        print(f"  Difference:               {diff_wide:.3f} (target: {HUMAN_TARGETS['stop_and_go']['speed_cv_diff_wide']})")
    
    stopgo_loss = stop_and_go_loss(stopgo_metrics)
    print(f"  Stop-and-Go Loss: {stopgo_loss:.4f}")
    
    total_loss = steering_loss + stopgo_loss
    print(f"\n  Total Loss: {total_loss:.4f}")
    
    return steering_metrics, stopgo_metrics


def main():
    global HUMAN_TARGETS
    
    parser = argparse.ArgumentParser(description="Automatic Parameter Tuning")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "all", "hierarchical", "test"],
                        help="Tuning mode: all=joint tuning (default, recommended), 1=steering law only, 2=stop-and-go only, hierarchical=phase1->phase2, test=evaluate current")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of optimization trials per phase")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Base configuration file")
    parser.add_argument("--save", action="store_true",
                        help="Save best parameters to config file")
    parser.add_argument("--trackpad-confidence", type=float, default=DEFAULT_TRACKPAD_CONFIDENCE,
                        help="Minimum trackpad confidence for filtering participants (0 to disable)")
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE and args.phase != "test":
        print("Error: optuna is required. Install with: pip install optuna")
        sys.exit(1)
    
    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    
    base_config = load_config(config_path)
    
    print("=" * 70)
    print("Automatic Parameter Tuning")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Phase: {args.phase}")
    if args.phase != "test":
        print(f"Trials: {args.trials}")
    print(f"Trackpad filter: confidence >= {args.trackpad_confidence}" if args.trackpad_confidence > 0 else "Trackpad filter: disabled")
    print()
    
    # Compute human targets from actual data
    print("Computing human targets from data...")
    trackpad_conf = args.trackpad_confidence if args.trackpad_confidence > 0 else None
    HUMAN_TARGETS = compute_human_targets(trackpad_confidence=trackpad_conf)
    
    print("\nHuman Target Metrics (computed from data):")
    print(f"  Steering Law:")
    print(f"    Speed ratio (wide/narrow): {HUMAN_TARGETS['steering_law']['speed_ratio_wide_narrow']:.3f}")
    print(f"    MT ratio (wide/narrow): {HUMAN_TARGETS['steering_law']['mt_ratio_wide_narrow']:.3f}")
    print(f"  Stop-and-Go:")
    print(f"    SpeedCV diff (narrow): {HUMAN_TARGETS['stop_and_go']['speed_cv_diff_narrow']:.3f}")
    print(f"    SpeedCV diff (wide): {HUMAN_TARGETS['stop_and_go']['speed_cv_diff_wide']:.3f}")
    print(f"    MinRatio corner narrow: {HUMAN_TARGETS['stop_and_go']['min_ratio_corner_narrow']:.3f}")
    
    # Print speed statistics for parameter range computation
    speed_stats = HUMAN_TARGETS.get("speed_stats", {})
    print(f"\n  Speed Statistics (for parameter ranges):")
    print(f"    Global min speed: {speed_stats.get('global_min', 0):.4f} m/s")
    print(f"    Global max speed: {speed_stats.get('global_max', 0):.4f} m/s")
    print(f"    Global mean speed: {speed_stats.get('global_mean', 0):.4f} m/s")
    print(f"    speed_floor range: [{max(0.05, speed_stats.get('speed_floor_ratio', 0.1) * 0.5):.3f}, {min(0.5, speed_stats.get('speed_floor_ratio', 0.1) * 2.0):.3f}]")
    print(f"    speed_ceil range: [{max(1.0, speed_stats.get('speed_ceil_ratio', 3.0) * 0.5):.3f}, {min(5.0, speed_stats.get('speed_ceil_ratio', 3.0) * 2.0):.3f}]")
    print(f"    MinRatio corner wide: {HUMAN_TARGETS['stop_and_go']['min_ratio_corner_wide']:.3f}")
    
    # Run tuning
    best_params = {}
    
    if args.phase == "test":
        test_current_config(config_path)
        print("\nDone.")
        return
    elif args.phase == "1":
        best_params = run_phase1_tuning(base_config, args.trials)
    elif args.phase == "2":
        best_params = run_phase2_tuning(base_config, args.trials)
    elif args.phase == "hierarchical":
        best_params = run_hierarchical_tuning(base_config, args.trials, args.trials)
    else:  # all
        best_params = run_combined_tuning(base_config, args.trials)
    
    # Optionally save to config
    if args.save and best_params:
        print("\n--- Saving Best Parameters ---")
        updated_config = update_config_params(base_config, best_params)
        save_config(updated_config, config_path)
        print(f"Saved to: {config_path}")
    else:
        print("\n--- Recommended Parameters (not saved) ---")
        print("Add --save flag to update config file")
        print()
        print("planner_weights update:")
        for k, v in best_params.items():
            print(f'    "{k}": {v:.4f},')
    
    print("\nDone.")


if __name__ == "__main__":
    main()
