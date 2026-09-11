"""Unified constraint representation for steering tasks.

This module provides a flexible constraint system that can represent various
types of spatial constraints (obstacles, boundaries, corridors) and convert
them to corridor_bounds functions used by the steering simulator.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Union
from enum import Enum


class ConstraintType(str, Enum):
    """Types of constraints"""
    KEEP_OUT = "keep_out"      # Avoid this region (obstacles)
    KEEP_IN = "keep_in"         # Stay within this region (boundaries)

@dataclass
class PathConstraint:
    """A line segment constraint"""
    path: List[Tuple[float, float]]
    width: Optional[float] = None  # Thickness if it's a thick line


@dataclass
class PolygonConstraint:
    """A polygon constraint"""
    vertices: List[Tuple[float, float]]
    

@dataclass
class CircleConstraint:
    """A circular constraint"""
    center: Tuple[float, float]
    radius: float


@dataclass
class RectangleConstraint:
    """A rectangular constraint"""
    x: float
    y: float
    width: float
    height: float


@dataclass
class ConstraintRegion:
    """A single constraint region with type and geometry"""
    constraint_type: ConstraintType
    geometry: Union[
        PathConstraint,
        PolygonConstraint,
        CircleConstraint,
        RectangleConstraint
    ]
    margin: Optional[float] = None  # Safety margin for this region

@dataclass
class ConstraintConfig:
    """Complete constraint configuration for a task"""
    regions: List[ConstraintRegion] = None

    def __post_init__(self):
        """Initialize regions list if None"""
        if self.regions is None:
            self.regions = []
