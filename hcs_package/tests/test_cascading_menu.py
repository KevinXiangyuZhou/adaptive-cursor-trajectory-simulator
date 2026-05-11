"""
Test for cascading menu task functionality.

This test verifies that the CursorSimulator can generate trajectories
for a cascading menu task based on task.json configuration.
"""

import json
import os
import tempfile
from pathlib import Path
import pytest
import numpy as np

from hcs_package import CursorSimulator




def test_cascading_menu_trajectory():
    """Test that cascading menu trajectory can be generated from task.json."""
    # Get the test_task.json file from tests directory
    test_dir = Path(__file__).parent
    task_file = test_dir / "test_task.json"
    
    # Verify test file exists
    assert task_file.exists(), f"Test task file not found: {task_file}"
    
    # Initialize simulator
    simulator = CursorSimulator()
    
    # Generate trajectory using the example test_task.json
    trajectory = simulator.generate_trajectory_with_waypoints(
        task_file=str(task_file),
        use_optimal_path=True
    )
    
    # Verify trajectory is generated
    assert trajectory is not None, "Trajectory should not be None"
    assert len(trajectory) > 0, "Trajectory should contain at least one point"
    assert len(trajectory) >= 2, "Trajectory should contain multiple waypoints"
    
    # Verify trajectory format: (x, y, delay)
    for point in trajectory:
        assert len(point) == 3, f"Each trajectory point should be (x, y, delay), got {point}"
        x, y, delay = point
        assert isinstance(x, (int, float)), f"x should be numeric, got {type(x)}"
        assert isinstance(y, (int, float)), f"y should be numeric, got {type(y)}"
        assert isinstance(delay, (int, float)), f"delay should be numeric, got {type(delay)}"
        assert delay > 0, f"delay should be positive, got {delay}"
    
    # Verify trajectory starts near first waypoint
    first_waypoint = [100, 50]  # Approximate start position in pixels
    first_traj_point = trajectory[0]
    start_distance = np.sqrt(
        (first_traj_point[0] - first_waypoint[0])**2 + 
        (first_traj_point[1] - first_waypoint[1])**2
    )
    # Allow some tolerance for coordinate conversion
    assert start_distance < 500, f"Trajectory should start near first waypoint, distance: {start_distance}"
    
    # Verify trajectory progresses (not all points are the same)
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]
    assert len(set(x_coords)) > 1 or len(set(y_coords)) > 1, "Trajectory should have movement"


def test_with_example_task_file():
    """Test using the example test_task.json file directly."""
    # Get the test_task.json file from tests directory
    test_dir = Path(__file__).parent
    task_file = test_dir / "test_task.json"
    
    # Verify test file exists
    assert task_file.exists(), f"Test task file not found: {task_file}"
    
    # Initialize simulator
    simulator = CursorSimulator()
    
    # Generate trajectory using the example test_task.json
    trajectory = simulator.generate_trajectory_with_waypoints(
        task_file=str(task_file)
    )
    
    # Verify trajectory is generated
    assert trajectory is not None, "Trajectory should not be None"
    assert len(trajectory) > 0, "Trajectory should contain at least one point"
    
    # Verify trajectory format
    for point in trajectory:
        assert len(point) == 3, f"Each trajectory point should be (x, y, delay), got {point}"
        x, y, delay = point
        assert isinstance(x, (int, float)), f"x should be numeric, got {type(x)}"
        assert isinstance(y, (int, float)), f"y should be numeric, got {type(y)}"
        assert delay > 0, f"delay should be positive, got {delay}"
    
    # Save trajectory output to tests/output directory
    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    output_file = output_dir / "test_trajectory_output.json"
    trajectory_data = {
        "trajectory": [[float(x), float(y), float(delay)] for x, y, delay in trajectory],
        "num_points": len(trajectory),
        "task_file": str(task_file)
    }
    with open(output_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    # Also save as CSV for easy viewing
    csv_file = output_dir / "test_trajectory_output.csv"
    import csv
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'delay'])
        writer.writerows(trajectory)
    
    print(f"\nTrajectory outputs saved to:")
    print(f"  - JSON: {output_file}")
    print(f"  - CSV: {csv_file}")
    print(f"  - Total points: {len(trajectory)}")




if __name__ == "__main__":
    # Run tests
    test_cascading_menu_trajectory()
    print("All tests passed!")
