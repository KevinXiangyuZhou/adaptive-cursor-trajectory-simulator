"""Typed parameter structures for steering models."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class BumpParams:
    """Parameters for the intermittent BUMP planning model."""
    pred_horizon: int
    Tp: int
    nc: Sequence[float]


@dataclass
class EnvParams:
    """Environment and device parameters."""
    interval: float


@dataclass
class TunnelInfo:
    """Centerline path and width of the tunnel environment."""
    tunnel_path: List[Tuple[float, float]]
    tunnel_width: float
    top_wall: Optional[List[Tuple[float, float]]] = None
    bottom_wall: Optional[List[Tuple[float, float]]] = None


@dataclass
class HOCLParams:
    """Open-loop correction pulse parameters (HOCL)."""
    mL: float
    mR: float
    Krb: float
    Trb: float


@dataclass
class SteeringModelInput:
    """Structured input for steering models.

    - state_cog: (cursor_x, cursor_y, cursor_vx, cursor_vy) - cursor position and velocity
    - bump: Bump planning parameters
    - env: Environment parameters
    - tunnel: Tunnel geometry
    """
    state_cog: Tuple[float, float, float, float]
    bump: BumpParams
    env: EnvParams
    tunnel: TunnelInfo
    planner_weights: Optional[dict] = None
    planner_margin: Optional[float] = None
    reference_path: Optional[object] = None
    current_acc: Optional[Tuple[float, float]] = None
    corridor_bounds: Optional[Tuple] = None  # (left_bound, right_bound), path-relative
    cartesian_constraints: Optional[List[Any]] = None  # List[ConstraintRegion], world-space
    clearance_profile: Optional[Tuple] = None  # (s_array, clearance_array)
    curvature_rate_profile: Optional[Tuple] = None  # (s_array, rate_array)
    curvature_profile: Optional[Tuple] = None  # (s_array, kappa_array)
    # --- anchor-drive planning (the gaze module's per-solve decision) ---
    # Arc length of the gaze anchor: the plan's via-point target. Required.
    anchor_s: Optional[float] = None
    # Horizon node count by which the plan must reach the anchor (the plan
    # deadline); nodes beyond it are the coast/pace tail.
    deadline_steps: Optional[int] = None
    # Progress pace (m/s) of the pace-holding tail beyond the deadline node;
    # 0 disables the tail.
    anchor_pace: float = 0.0
    # Steps executed since the previous solve — the warm-start shift for the
    # MPCC. 1 for per-step replanning; >1 under intermittent replanning.
    warm_shift: int = 1