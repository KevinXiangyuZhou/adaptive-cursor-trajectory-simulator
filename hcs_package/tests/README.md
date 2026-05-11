# Tests

This directory contains tests for the human-like cursor simulator package.

## Running Tests

### Using pytest (recommended)

```bash
cd hcs_package
pip install -e ".[dev]"  # Install package with dev dependencies including pytest
pytest tests/ -v
```

### Running a specific test

```bash
pytest tests/test_cascading_menu.py -v
```

### Running tests directly with Python

```bash
cd hcs_package
python3 -m tests.test_cascading_menu
```

## Test Files

### `test_cascading_menu.py`

Tests for cascading menu task functionality:

- `test_cascading_menu_trajectory()`: Tests trajectory generation using the example `test_task.json` file with a complete cascading menu task configuration including waypoints, screen dimensions, and constraints
- `test_cascading_menu_without_constraints()`: Tests trajectory generation without constraints
- `test_with_example_task_file()`: Additional test using the example `test_task.json` file
- `test_cascading_menu_invalid_task_file()`: Tests error handling for invalid task files

### `test_task.json`

Example task configuration file used by the tests. This file demonstrates the proper structure for a cascading menu task:

- **waypoints**: List of [x, y] coordinates in screen pixels
  - Start position → main menu target → submenu target
- **screen_width**: Screen width in pixels (1920)
- **screen_height**: Screen height in pixels (1080)
- **constraints**: Constraint configuration with keep_in regions for:
  - Main menu rectangle (0.1, 0.02, 0.12x0.12 meters)
  - Submenu rectangle (0.22, 0.11, 0.1x0.12 meters)

This file is based on the structure from `task_config_cascading_menu.json` and serves as a reference example for creating task.json files.
