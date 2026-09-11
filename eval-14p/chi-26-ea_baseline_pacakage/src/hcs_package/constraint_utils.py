"""
Utility functions for parsing constraints from JSON and converting them to corridor bounds.
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from pathlib import Path
from .constraints import (
    ConstraintConfig, ConstraintRegion, ConstraintType,
    PathConstraint, PolygonConstraint, CircleConstraint, RectangleConstraint
)
from .reference_path import ReferencePath


def parse_constraints_from_json(constraints_dict: Dict[str, Any]) -> Optional[ConstraintConfig]:
    """
    Parse constraints from a JSON dictionary.
    
    Args:
        constraints_dict: Dictionary containing constraint configuration
        
    Returns:
        ConstraintConfig object or None if no constraints
    """
    if not constraints_dict:
        return None
    
    regions = []
    
    # Handle simple boundary constraints
    if "left_boundary" in constraints_dict and "right_boundary" in constraints_dict:
        left_boundary = constraints_dict["left_boundary"]
        right_boundary = constraints_dict["right_boundary"]
        
        # Create path constraints for boundaries
        if left_boundary:
            regions.append(ConstraintRegion(
                constraint_type=ConstraintType.KEEP_IN,
                geometry=PathConstraint(
                    path=[tuple(p) for p in left_boundary],
                    width=None
                )
            ))
        
        if right_boundary:
            regions.append(ConstraintRegion(
                constraint_type=ConstraintType.KEEP_IN,
                geometry=PathConstraint(
                    path=[tuple(p) for p in right_boundary],
                    width=None
                )
            ))
    
    # Handle region-based constraints
    if "regions" in constraints_dict:
        for region_dict in constraints_dict["regions"]:
            if not region_dict.get("enabled", True):
                continue
            
            constraint_type_str = region_dict.get("constraint_type", "keep_out")
            constraint_type = ConstraintType.KEEP_OUT if constraint_type_str == "keep_out" else ConstraintType.KEEP_IN
            
            geometry_dict = region_dict.get("geometry", {})
            geometry_type = geometry_dict.get("type", "")
            
            geometry = None
            if geometry_type == "circle":
                geometry = CircleConstraint(
                    center=tuple(geometry_dict["center"]),
                    radius=geometry_dict["radius"]
                )
            elif geometry_type == "rectangle":
                geometry = RectangleConstraint(
                    x=geometry_dict["x"],
                    y=geometry_dict["y"],
                    width=geometry_dict["width"],
                    height=geometry_dict["height"]
                )
            elif geometry_type == "polygon":
                geometry = PolygonConstraint(
                    vertices=[tuple(v) for v in geometry_dict["vertices"]]
                )
            elif geometry_type == "line" or geometry_type == "path":
                path_points = geometry_dict.get("path", [])
                if not path_points and "start" in geometry_dict and "end" in geometry_dict:
                    path_points = [geometry_dict["start"], geometry_dict["end"]]
                geometry = PathConstraint(
                    path=[tuple(p) for p in path_points],
                    width=geometry_dict.get("width")
                )
            
            if geometry is not None:
                region = ConstraintRegion(
                    constraint_type=constraint_type,
                    geometry=geometry
                )
                # Store margin in the region if provided
                if "margin" in region_dict:
                    region.margin = region_dict["margin"]
                regions.append(region)
    
    # Handle symmetric corridor
    if "corridor_width" in constraints_dict and not regions:
        # Create a simple symmetric corridor constraint
        corridor_width = constraints_dict["corridor_width"]
        default_margin = constraints_dict.get("default_margin", 0.0)
        # This will be handled by the corridor_bounds conversion
    
    if not regions:
        return None
    
    config = ConstraintConfig(regions=regions)
    config.coordinate_system = constraints_dict.get("coordinate_system", "normalized")
    config.default_margin = constraints_dict.get("default_margin", 0.0)
    
    return config


def convert_constraints_to_corridor_bounds(
    constraints: Optional[ConstraintConfig],
    reference_path: ReferencePath,
    default_margin: float = 0.005,
    screen_width: Optional[float] = None,
    screen_height: Optional[float] = None
) -> Optional[Tuple[Callable, Callable]]:
    """
    Convert constraints to corridor_bounds functions for the MPC planner.
    
    This is a simplified version that creates symmetric bounds based on the
    closest constraint at each point along the path.
    
    Args:
        constraints: ConstraintConfig object
        reference_path: ReferencePath object
        default_margin: Default safety margin
        screen_width: Screen width in pixels (for coordinate conversion)
        screen_height: Screen height in pixels (for coordinate conversion)
    
    Returns:
        Tuple of (left_bound_func, right_bound_func) or None if no constraints
    """
    if constraints is None:
        return None
    
    # Sample reference path
    num_samples = max(50, int(reference_path.total_length / 0.01))
    s_samples = np.linspace(0, reference_path.total_length, num_samples)
    
    left_bounds = []
    right_bounds = []
    
    margin = getattr(constraints, 'default_margin', default_margin)
    
    for s in s_samples:
        path_point = np.array(reference_path(s), dtype=float)
        normal = reference_path.normal(s)  # Right-pointing normal
        
        # Initialize bounds (large = unconstrained)
        left_bound = 1e6
        right_bound = 1e6
        
        # Process each constraint region
        for region in constraints.regions:
            region_margin = getattr(region, 'margin', None)
            if region_margin is None:
                region_margin = margin
            geometry = region.geometry
            
            # Simple distance computation for different geometry types
            if isinstance(geometry, CircleConstraint):
                center = np.array(geometry.center)
                vec_to_center = center - path_point
                dist_to_center = np.linalg.norm(vec_to_center)
                
                if region.constraint_type == ConstraintType.KEEP_OUT:
                    # Obstacle: stay away
                    if dist_to_center < geometry.radius:
                        # Inside obstacle - compute distance to edge
                        dist_to_edge = geometry.radius - dist_to_center
                        # Project onto normal
                        proj = np.dot(vec_to_center, normal)
                        if proj < 0:
                            left_bound = min(left_bound, max(0.0, dist_to_edge - region_margin))
                        else:
                            right_bound = min(right_bound, max(0.0, dist_to_edge - region_margin))
                else:  # KEEP_IN
                    # Boundary: stay inside
                    if dist_to_center < geometry.radius:
                        dist_to_edge = geometry.radius - dist_to_center
                        # Constrain both sides
                        safe_dist = dist_to_edge - region_margin
                        left_bound = min(left_bound, max(0.0, safe_dist))
                        right_bound = min(right_bound, max(0.0, safe_dist))
            
            elif isinstance(geometry, RectangleConstraint):
                # Check if point is inside rectangle
                x_min = geometry.x
                x_max = geometry.x + geometry.width
                y_min = geometry.y
                y_max = geometry.y + geometry.height
                
                inside = (x_min <= path_point[0] <= x_max and 
                         y_min <= path_point[1] <= y_max)
                
                if inside:
                    # Compute distance to nearest edge
                    dist_left = path_point[0] - x_min
                    dist_right = x_max - path_point[0]
                    dist_bottom = path_point[1] - y_min
                    dist_top = y_max - path_point[1]
                    
                    min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
                    
                    if region.constraint_type == ConstraintType.KEEP_IN:
                        safe_dist = min_dist - region_margin
                        left_bound = min(left_bound, max(0.0, safe_dist))
                        right_bound = min(right_bound, max(0.0, safe_dist))
                    else:  # KEEP_OUT
                        # Avoid obstacle - use minimum distance
                        safe_dist = min_dist - region_margin
                        # Project onto normal to determine left/right
                        # Simplified: use x-direction
                        if path_point[0] < (x_min + x_max) / 2:
                            left_bound = min(left_bound, max(0.0, safe_dist))
                        else:
                            right_bound = min(right_bound, max(0.0, safe_dist))
            
            elif isinstance(geometry, PathConstraint) and geometry.width is not None:
                # Path with width defines a corridor
                half_width = geometry.width / 2.0
                safe_dist = half_width - region_margin
                left_bound = min(left_bound, max(0.0, safe_dist))
                right_bound = min(right_bound, max(0.0, safe_dist))
        
        left_bounds.append(left_bound)
        right_bounds.append(right_bound)
    
    # Create interpolated functions
    left_bounds = np.array(left_bounds)
    right_bounds = np.array(right_bounds)
    
    # Clamp to reasonable values
    max_bound = 0.1  # 10cm max
    left_bounds = np.clip(left_bounds, 0.0, max_bound)
    right_bounds = np.clip(right_bounds, 0.0, max_bound)
    
    def left_bound_func(s):
        idx = np.clip(int(s / reference_path.total_length * (num_samples - 1)), 0, num_samples - 1)
        return float(left_bounds[idx])
    
    def right_bound_func(s):
        idx = np.clip(int(s / reference_path.total_length * (num_samples - 1)), 0, num_samples - 1)
        return float(right_bounds[idx])
    
    return (left_bound_func, right_bound_func)
