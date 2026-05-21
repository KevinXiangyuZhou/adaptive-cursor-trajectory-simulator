"""Unified constraint representation for steering tasks (obstacles, boundaries, corridors)."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Union
from enum import Enum


class ConstraintType(str, Enum):
    KEEP_OUT = "keep_out"
    KEEP_IN = "keep_in"


@dataclass
class PathConstraint:
    path: List[Tuple[float, float]]
    width: Optional[Union[float, List[float]]] = None


@dataclass
class PolygonConstraint:
    vertices: List[Tuple[float, float]]


@dataclass
class RectangleConstraint:
    x: float
    y: float
    width: float
    height: float


@dataclass
class ConstraintRegion:
    constraint_type: ConstraintType
    geometry: Union[
        PathConstraint,
        PolygonConstraint,
        RectangleConstraint
    ]
    margin: Optional[float] = None


@dataclass
class ConstraintConfig:
    regions: List[ConstraintRegion] = None

    def __post_init__(self):
        if self.regions is None:
            self.regions = []
