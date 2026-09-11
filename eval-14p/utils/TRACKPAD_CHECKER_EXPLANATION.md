# Trackpad Checker: Features and Confidence Computation

## Overview

The trackpad checker distinguishes between **trackpad** and **mouse** input devices by analyzing movement characteristics from raw trajectory data. Trackpads and mice exhibit different movement patterns due to their physical interaction methods.

**Important Note**: This classifier is designed for **steering tasks** where users navigate along a path. The characteristics differ from point-and-click tasks:
- **Steering tasks**: Mice allow continuous movement (fewer zero periods), trackpads require finger repositioning (more zero periods)
- **Point-and-click tasks**: Behavior may differ (mice are lifted more frequently)

---

## Extracted Features

The system extracts **11 features** from trajectory and speed data:

### 1. **Zero-Velocity Ratio** (`zero_velocity_ratio`)
- **What it measures**: Proportion of time points where speed is near zero (< 0.001 m/s)
- **Why it matters** (for steering tasks): 
  - **Trackpads**: Users need to lift and reposition fingers multiple times for corrections, creating more zero-velocity periods
  - **Mice**: Allow continuous steering movements without lifting, resulting in fewer zero-velocity periods
- **Range**: 0.0 to 1.0 (0% to 100% of time)

### 2. **Speed Variance** (`speed_variance`)
- **What it measures**: Statistical variance of speed values
- **Why it matters**: 
  - **Trackpads**: More consistent speeds due to continuous finger movement
  - **Mice**: More variable speeds due to discrete hand movements
- **Range**: ≥ 0

### 3. **Speed Coefficient of Variation** (`speed_cv`)
- **What it measures**: Standard deviation divided by mean speed (normalized variance)
- **Why it matters**: Normalized measure of speed consistency, independent of absolute speed
- **Range**: ≥ 0

### 4. **Mean Jerk** (`mean_jerk`)
- **What it measures**: Average rate of change of acceleration (third derivative of position)
- **Why it matters**: 
  - **Trackpads**: Lower jerk due to smoother, more fluid finger movements
  - **Mice**: Higher jerk due to discrete hand movements and lifting/repositioning
- **Range**: ≥ 0 (units: m/s³)

### 5. **Jerk Standard Deviation** (`std_jerk`)
- **What it measures**: Variability in jerk values
- **Why it matters**: Indicates consistency of movement smoothness
- **Range**: ≥ 0

### 6. **Max Jerk** (`max_jerk`)
- **What it measures**: Maximum jerk value observed
- **Why it matters**: Captures sudden movements or jerky motions
- **Range**: ≥ 0

### 7. **Acceleration Variance** (`accel_variance`)
- **What it measures**: Statistical variance of acceleration magnitudes
- **Why it matters**: 
  - **Trackpads**: Lower variance (smoother acceleration changes)
  - **Mice**: Higher variance (more abrupt acceleration changes)
- **Range**: ≥ 0

### 8. **Acceleration Standard Deviation** (`accel_std`)
- **What it measures**: Standard deviation of acceleration magnitudes
- **Why it matters**: Another measure of acceleration smoothness
- **Range**: ≥ 0

### 9. **Micro-Movement Ratio** (`micro_movement_ratio`)
- **What it measures**: Proportion of movements smaller than 1cm (0.01m)
- **Why it matters**: 
  - **Trackpads**: More fine-grained, continuous adjustments
  - **Mice**: Fewer micro-movements due to discrete hand motions
- **Range**: 0.0 to 1.0

### 10. **Movement Continuity** (`movement_continuity`)
- **What it measures**: Ratio of longest consecutive non-zero movement streak to total movements
- **Why it matters** (for steering tasks): 
  - **Mice**: Longer continuous movement sequences (can steer continuously without repositioning)
  - **Trackpads**: More interrupted movement patterns (due to finger lifting/repositioning)
- **Range**: 0.0 to 1.0

### 11. **Speed Change Rate** (`speed_change_rate`)
- **What it measures**: Average absolute speed change divided by mean speed
- **Why it matters**: 
  - **Trackpads**: Lower rate (smoother speed transitions)
  - **Mice**: Higher rate (more abrupt speed changes)
- **Range**: ≥ 0

---

## Confidence Computation

The confidence score is computed using a **rule-based scoring system** with weighted evidence:

### Scoring System

1. **Initialize two counters**:
   - `trackpad_score`: Evidence favoring trackpad (starts at 0.0)
   - `mouse_score`: Evidence favoring mouse (starts at 0.0)

2. **Apply 6 classification rules** (each rule adds points to the appropriate counter):

#### Rule 1: Mean Jerk (Weight: 2.0 points)
- If `mean_jerk < 50`: Add **2.0** to `trackpad_score`
- If `mean_jerk > 200`: Add **2.0** to `mouse_score`
- **Rationale**: Jerk is the strongest indicator of movement smoothness

#### Rule 2: Zero-Velocity Ratio (Weight: 1.5 points)
- If `zero_velocity_ratio > 0.3` (more than 30%): Add **1.5** to `trackpad_score`
- If `zero_velocity_ratio < 0.1` (less than 10%): Add **1.5** to `mouse_score`
- **Rationale**: For steering tasks, trackpads require finger lifting/repositioning (more zero periods), while mice allow continuous movement (fewer zero periods)

#### Rule 3: Micro-Movement Ratio (Weight: 1.0 point)
- If `micro_movement_ratio > 0.3` (more than 30%): Add **1.0** to `trackpad_score`
- If `micro_movement_ratio < 0.1` (less than 10%): Add **1.0** to `mouse_score`
- **Rationale**: Trackpads enable more fine-grained movements

#### Rule 4: Movement Continuity (Weight: 1.0 point)
- If `movement_continuity > 0.5`: Add **1.0** to `mouse_score`
- If `movement_continuity < 0.2`: Add **1.0** to `trackpad_score`
- **Rationale**: For steering tasks, mice allow continuous steering without repositioning (higher continuity), while trackpads have more interruptions (lower continuity)

#### Rule 5: Speed Change Rate (Weight: 0.5 points)
- If `speed_change_rate < 0.5`: Add **0.5** to `trackpad_score`
- If `speed_change_rate > 1.5`: Add **0.5** to `mouse_score`
- **Rationale**: Trackpads have smoother speed transitions

#### Rule 6: Acceleration Variance (Weight: 0.5 points)
- If `accel_variance < 0.1`: Add **0.5** to `trackpad_score`
- If `accel_variance > 1.0`: Add **0.5** to `mouse_score`
- **Rationale**: Trackpads have more consistent acceleration

### Confidence Calculation

After applying all rules:

```python
total_score = trackpad_score + mouse_score

if total_score == 0:
    return 'unknown', 0.0  # No evidence

if trackpad_score > mouse_score:
    confidence = trackpad_score / total_score
    return 'trackpad', confidence
else:
    confidence = mouse_score / total_score
    return 'mouse', confidence
```

**Confidence Formula**: 
- Confidence = `winning_score / total_score`
- Range: 0.0 to 1.0
- **Interpretation**:
  - **0.5-0.6**: Weak evidence
  - **0.6-0.8**: Moderate evidence
  - **0.8-1.0**: Strong evidence

### Example

**Example 1: Trackpad User**
If a participant has:
- `mean_jerk = 40` → +2.0 trackpad
- `zero_velocity_ratio = 0.35` → +1.5 trackpad (high zero periods from finger repositioning)
- `micro_movement_ratio = 0.35` → +1.0 trackpad
- `movement_continuity = 0.15` → +1.0 trackpad (low continuity from interruptions)
- `speed_change_rate = 0.4` → +0.5 trackpad
- `accel_variance = 0.05` → +0.5 trackpad

**Result**:
- `trackpad_score = 6.5`
- `mouse_score = 0.0`
- `total_score = 6.5`
- `confidence = 6.5 / 6.5 = 1.0` (100% confident it's a trackpad)

**Example 2: Mouse User**
If a participant has:
- `mean_jerk = 250` → +2.0 mouse
- `zero_velocity_ratio = 0.05` → +1.5 mouse (low zero periods, continuous movement)
- `micro_movement_ratio = 0.05` → +1.0 mouse
- `movement_continuity = 0.7` → +1.0 mouse (high continuity, continuous steering)
- `speed_change_rate = 2.0` → +0.5 mouse
- `accel_variance = 1.5` → +0.5 mouse

**Result**:
- `trackpad_score = 0.0`
- `mouse_score = 6.5`
- `total_score = 6.5`
- `confidence = 6.5 / 6.5 = 1.0` (100% confident it's a mouse)

---

## Feature Aggregation Across Trials

For each participant, features are extracted from **all trials** and then **averaged**:

```python
# For each trial, extract features
all_features = [extract_features(trial) for trial in trials]

# Aggregate by taking mean (except num_points which is summed)
aggregated_features = {
    'mean_jerk': mean([f['mean_jerk'] for f in all_features]),
    'zero_velocity_ratio': mean([f['zero_velocity_ratio'] for f in all_features]),
    # ... etc
    'num_points': sum([f['num_points'] for f in all_features])
}
```

This provides a **robust estimate** by averaging across multiple trials, reducing the impact of outliers or unusual movements in individual trials.

---

## Threshold Values

The classification uses these thresholds (tuned based on typical trackpad vs mouse behavior):

| Feature | Trackpad Threshold | Mouse Threshold |
|---------|-------------------|-----------------|
| Mean Jerk | < 50 | > 200 |
| Zero-Velocity Ratio | > 0.3 (30%) | < 0.1 (10%) |
| Micro-Movement Ratio | > 0.3 (30%) | < 0.1 (10%) |
| Movement Continuity | < 0.2 | > 0.5 |
| Speed Change Rate | < 0.5 | > 1.5 |
| Acceleration Variance | < 0.1 | > 1.0 |

These thresholds can be adjusted based on validation data if needed.
