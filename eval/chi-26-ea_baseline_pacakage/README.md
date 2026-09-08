# Human-like Cursor Simulator

A Python package for generating human-like cursor movements that can be used as an extension to Playwright for realistic browser automation.

## Features

- **Human-like cursor trajectories**: Uses models based on motor control and human behavior research
- **Playwright compatible**: Easy integration with Playwright's mouse movement functions
- **Configurable parameters**: Adjust simulation parameters to match different user behaviors
- **Path following**: Support for both point-to-point and path-following trajectories

## Installation

```bash
cd hcs_package
pip install -e .
```

Or install from the parent directory:

```bash
pip install -e ./hcs_package
```

## Quick Start

### Basic Usage with Task File

The recommended way to use the simulator is with a `task.json` file that contains both waypoints and constraints:

**Example Usage:**
```python
from hcs_package import CursorSimulator

# Initialize simulator (uses default configuration)
simulator = CursorSimulator()

# Generate trajectory from task file
trajectory = simulator.generate_trajectory_with_waypoints(
    task_file="task.json",
    use_optimal_path=True
)
```
## API Reference

### `CursorSimulator`

Main class for generating cursor trajectories.

#### Initialization

```python
simulator = CursorSimulator()
```

The simulator uses default configuration parameters. No parameters are required.

#### Methods

##### `generate_trajectory_with_waypoints(task_file)`

Generate a trajectory following waypoints with optional constraints and optimal path generation.

**Parameters:**
- `task_file`: Optional path to `task.json` file containing `waypoints`

**Returns:**
- List of tuples `(x, y, delay)`

**Task File Format (task.json):**
```json
{
  "waypoints": [
    [100, 200],
    [300, 250],
    [500, 300]
  ],
  "screen_width": 1920,
  "screen_height": 1080,
  "constraints": {
    "coordinate_system": "normalized",
    "default_margin": 0.005,
    "regions": [
      {
        "constraint_type": "keep_in",
        "geometry": {
          "type": "circle",
          "center": [0.3, 0.2],
          "radius": 0.015
        },
        "margin": 0.002,
        "enabled": true
      }
    ]
  }
}
```

**Note:** The `waypoints` field is required in `task.json`. The `screen_width`, `screen_height`, and `constraints` fields are optional.

Supported geometry types: `circle`, `rectangle`, `polygon`, `line`/`path`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite the relevant papers on which this simulator is based.
