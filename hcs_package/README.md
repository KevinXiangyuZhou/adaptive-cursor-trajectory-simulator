# Human-like Cursor Simulator (hcs_package)

A Python package for generating human-like cursor trajectories for constrained and unconstrained pointing tasks. Uses Model Predictive Contouring Control (MPCC) with a geometry-adaptive speed planner and signal-dependent motor noise.

## Installation

```bash
cd hcs_package
pip install -e .
```

Optional dependency for the GAM speed model:
```bash
pip install pygam
```

## Architecture Overview

The package generates trajectories through a closed-loop pipeline:

1. **Reference Path Generator** — computes a race-tracing path from waypoints and constraints, with geometry-adaptive corner-cutting
2. **Speed Planner** — a Generalized Additive Model (GAM) predicts target speed from local clearance, curvature, and curvature rate
3. **MPCC Optimizer** — solves a receding-horizon optimization at each step, balancing smoothness, speed tracking, and constraint compliance
4. **Motor & Mouse Noise** — perturbs the planned output with signal-dependent noise and feeds the cursor state back

## Quick Start

### Basic Usage

```python
from hcs_package import CursorSimulator

# Initialize with a built-in persona (uses population-level GAM speed model)
sim = CursorSimulator("office_worker")

# Generate trajectory from a task file
trajectory = sim.generate_trajectory_with_waypoints(
    task_file="task.json",
    use_optimal_path=True,
)

# trajectory is a list of (x, y, delay_seconds) tuples
for x, y, delay in trajectory:
    # x, y are in task pixel coordinates
    # delay is the time to wait before the next point (in seconds)
    pass
```

### With Playwright

```python
from hcs_package import CursorSimulator

sim = CursorSimulator("office_worker")
trajectory = sim.generate_trajectory_with_waypoints(task_file="task.json")

for x, y, delay in trajectory:
    await page.mouse.move(x, y)
    await page.wait_for_timeout(int(delay * 1000))
```

### Using a Custom Config (e.g., per-user fitted model)

```python
sim = CursorSimulator("/path/to/fitted_config.json")
```

## Built-in Personas

Six personas are available, each with distinct motor profiles:

| Persona | `desired_speed` | `Th` | `nc` | Description |
|---------|----------------|------|------|-------------|
| `office_worker` | 0.20 | 0.30s | [0.20, 0.020] | Balanced speed and precision |
| `gamer` | 0.30 | 0.50s | [0.10, 0.010] | Fast, precise, long planning horizon |
| `novice` | 0.12 | 0.20s | [0.26, 0.026] | Slow, over-cautious |
| `fatigued` | 0.15 | 0.20s | [0.30, 0.030] | Reduced speed, elevated noise |
| `motor_impaired` | 0.10 | 0.15s | [0.40, 0.040] | Very slow, highest noise, maximum boundary avoidance |
| `young_children` | 0.12 | 0.15s | [0.36, 0.036] | Small forearm (0.22m), weak awareness |

All personas use a shared population-level GAM speed model (`population_gam.pkl`) that adapts speed to local tunnel geometry.

## Configuration Format

A configuration JSON file has this structure:

```json
{
  "Tp": 0.05,
  "Th": 0.30,
  "nc": [0.20, 0.020],
  "forearm": 0.35,
  "planner_weights": {
    "jerk": 5e-06,
    "progress": 3e-07,
    "constraint": 50,
    "contour": 10,
    "lag": 1.0,
    "desired_speed": 0.20
  },
  "speed_model": {
    "type": "gam",
    "path": "population_gam.pkl"
  }
}
```

### Parameter Reference

**Top-level parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Tp` | float | Planning interval in seconds (default: 0.05) |
| `Th` | float | Prediction horizon in seconds. Longer = more anticipatory planning |
| `nc` | [float, float] | Motor noise coefficients [directional, perpendicular]. Higher = noisier |
| `forearm` | float | Forearm length in metres. Affects mouse transfer function |
| `add_noise` | bool | Whether to apply motor/mouse noise (default: true) |
| `random_seed` | int | Random seed for noise reproducibility (default: 1000) |

**Planner weights** (in `planner_weights`):

| Weight | Description |
|--------|-------------|
| `jerk` | Smoothness penalty. Higher = smoother trajectories |
| `progress` | Speed tracking penalty. Higher = tracks target speed more tightly |
| `constraint` | Boundary violation penalty. Higher = stronger wall avoidance |
| `contour` | Lateral tracking penalty. Higher = follows reference path more tightly |
| `lag` | Longitudinal tracking penalty. Higher = less overshoot/undershoot along path |
| `desired_speed` | Nominal speed estimate (m/s). Used for lookahead estimation and optimizer scaling. **Not the actual target speed** — that comes from the GAM speed model |

**Speed model** (in `speed_model`):

| Field | Description |
|-------|-------------|
| `type` | Must be `"gam"` |
| `path` | Path to the fitted GAM pickle file. Relative paths are resolved against the config file's directory |

**Reference path** (optional `reference_path` section):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_cut` | 0.01 | Corner-cutting aggressiveness (0 = centerline, 1 = maximum cutting) |
| `w_suppress` | 0.0 | Suppression of cutting in dense/sharp turn regions |
| `w_width_exp` | 1.0 | Width sensitivity exponent for cutting |
| `cut_window_frac` | 0.05 | Lookahead window as fraction of path length |
| `global_clearance_ref` | 0.025 | Reference clearance for absolute task scaling |

## Task File Format

```json
{
  "waypoints": [[100, 200], [300, 250], [500, 300]],
  "screen_width": 1920,
  "screen_height": 1080,
  "target_radius": 0.01,
  "max_steps": 800,
  "constraints": {
    "coordinate_system": "normalized",
    "default_margin": 0.0,
    "regions": [
      {
        "constraint_type": "keep_in",
        "geometry": {
          "type": "path",
          "path": [[0.0, 0.13], [0.46, 0.13]],
          "width": 0.03
        },
        "enabled": true
      },
      {
        "constraint_type": "keep_out",
        "geometry": {
          "type": "rectangle",
          "x": 0.2, "y": 0.1,
          "width": 0.05, "height": 0.05
        }
      },
      {
        "constraint_type": "keep_in",
        "geometry": {
          "type": "polygon",
          "vertices": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.2], [0.1, 0.2]]
        }
      }
    ]
  }
}
```

**Constraint types:**
- **PathConstraint** (`"type": "path"`): Tunnel-like corridor with a centerline path and constant width. Enforced via path-relative lateral bounds.
- **PolygonConstraint** (`"type": "polygon"`): Arbitrary polygon region, tagged as `keep_in` or `keep_out`.
- **RectangleConstraint** (`"type": "rectangle"`): Axis-aligned rectangle, tagged as `keep_in` or `keep_out`.

All constraints are enforced as soft penalties in the MPCC optimizer's cost function.

## Key Differences from Baseline (chi-26-ea)

If migrating from the baseline package (`chi-26-ea_baseline_pacakage`):

| Change | Baseline | Current |
|--------|----------|---------|
| Boundary weight key | `"wall"` | `"constraint"` |
| Speed model | Fixed `desired_speed` only | GAM speed model required via `speed_model` config section |
| Reference path | QP-based with `alpha`, `beta`, `lambda_length`, `gamma_center` | Race-tracing with `w_cut`, `w_suppress`, etc. |
| `ParametricSpeedModel` | Available as fallback | Removed. GAM is required |
| Speed adaptation | None (constant target) | Geometry-adaptive via GAM (clearance, curvature, curvature rate) |

**Migration checklist:**
1. Replace `"wall"` with `"constraint"` in all planner_weights configs
2. Add `"speed_model": {"type": "gam", "path": "population_gam.pkl"}` to all configs
3. Copy `population_gam.pkl` from `hcs_package/src/hcs_package/user_configurations/` to your config directory
4. Remove any references to `ParametricSpeedModel`
5. Update reference path parameters if using custom configs (old QP params are ignored)

## Public API

```python
from hcs_package import CursorSimulator, GAMSpeedModel, model, reset_warm_start
```

- `CursorSimulator(user_config)` — main entry point
- `CursorSimulator.generate_trajectory_with_waypoints(task_file, ...)` — generate trajectory
- `reset_warm_start()` — reset MPCC warm-start cache (call between independent trajectories)
- `GAMSpeedModel` — speed model class (for fitting custom models)

## License

MIT License
