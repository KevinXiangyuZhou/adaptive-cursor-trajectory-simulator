"""Motor module: solve one MPCC plan toward the gaze module's fixation.

This is the paper's motor module. Given a Fixation (anchor, deadline, pace)
it builds the SteeringModelInput, runs the anchor-drive MPCC solve, and
exposes the resulting open-loop plan — planned per-step displacements and
velocities, the anchor's arrival geometry, and the deviation-trigger scale.
It decides the SHAPE of the movement; where and by when come from the gaze
module.

State is per trajectory: construct one MotorModule per generate_* call.
"""

from typing import List, Optional, Tuple

import numpy as np

from .gaze_module import Fixation
from .model import model, FREE_SPACE_CLEARANCE_M
from .params import SteeringModelInput, BumpParams, EnvParams, TunnelInfo


class MotorModule:

    def __init__(
        self,
        reference_path,
        *,
        interval: float,
        tp: int,
        nc: List[float],
        planner_weights: dict,
        planner_margin: float,
        tunnel_path,
        tunnel_width: Optional[float],
        corridor_bounds,
        cartesian_regions,
        clearance_profile: Optional[Tuple],
        curvature_profile: Optional[Tuple],
        target_radius: float,
        anchor_tail_pace: bool,
    ):
        self.reference_path = reference_path
        self.interval = interval
        self.tp = tp
        self.nc = nc
        self.planner_weights = planner_weights
        self.planner_margin = planner_margin
        self.tunnel_path = tunnel_path
        self.tunnel_width = tunnel_width
        self.corridor_bounds = corridor_bounds
        self.cartesian_regions = cartesian_regions
        self.clearance_profile = clearance_profile
        self.curvature_profile = curvature_profile
        self.target_radius = float(target_radius)
        self.anchor_tail_pace = bool(anchor_tail_pace)
        # The current open-loop plan (None before the first solve).
        self.plan = None

    def solve(self, cursor_pos, cursor_vel, cursor_acc, fixation: Fixation,
              warm_shift: int):
        """Solve the MPCC toward the fixation and store/return the plan."""
        model_input = SteeringModelInput(
            state_cog=(
                float(cursor_pos[0]),
                float(cursor_pos[1]),
                float(cursor_vel[0]),
                float(cursor_vel[1])
            ),
            bump=BumpParams(
                pred_horizon=fixation.n_solve,
                Tp=self.tp,
                nc=self.nc
            ),
            env=EnvParams(interval=self.interval),
            tunnel=TunnelInfo(
                tunnel_path=self.tunnel_path,
                tunnel_width=self.tunnel_width or 0.1,
                top_wall=None,
                bottom_wall=None
            ),
            planner_weights=self.planner_weights,
            planner_margin=self.planner_margin,
            reference_path=self.reference_path,
            current_acc=(float(cursor_acc[0]), float(cursor_acc[1])),
            corridor_bounds=self.corridor_bounds,
            cartesian_constraints=self.cartesian_regions if self.cartesian_regions else None,
            clearance_profile=self.clearance_profile,
            curvature_profile=self.curvature_profile,
            # The gaze anchor is the plan's via point at the deadline node
            # (n_base steps); the padded tail coasts (or holds the fixation's
            # pace with anchor_tail_pace).
            anchor_s=fixation.solve_anchor_s,
            deadline_steps=fixation.n_base,
            anchor_pace=(float(fixation.pace) if self.anchor_tail_pace else 0.0),
            warm_shift=int(warm_shift),
        )

        cursor_info, plan_debug = model(model_input)
        c_pos_dx, c_pos_dy, c_vel_x, c_vel_y = cursor_info

        # Deviation-trigger scale: the gaze module's local usable width where
        # the task is constrained; for free-space plans the plan's own span
        # (a plan that has drifted by deviation_frac of the distance it set
        # out to cover is invalid), floored at the target diameter so
        # terminal micro-plans keep a sane threshold.
        dev_scale = fixation.w_solve if (fixation.w_solve and fixation.w_solve > 0) else None
        if dev_scale is not None and dev_scale >= FREE_SPACE_CLEARANCE_M:
            span = float(np.hypot(np.sum(c_pos_dx), np.sum(c_pos_dy)))
            dev_scale = max(span, 2.0 * self.target_radius)
        plan = {
            'anchor_xy': None, 'arrival_tol': 0.0,
            'c_pos_dx': c_pos_dx, 'c_pos_dy': c_pos_dy,
            'c_vel_x': c_vel_x, 'c_vel_y': c_vel_y,
            'n_steps': len(c_pos_dx),
            'pos_x': cursor_pos[0] + np.cumsum(c_pos_dx),
            'pos_y': cursor_pos[1] + np.cumsum(c_pos_dy),
            'dev_scale': dev_scale,
        }
        a_xy = self.reference_path(float(fixation.anchor_s))
        plan['anchor_xy'] = (float(a_xy[0]), float(a_xy[1]))
        plan['anchor_s'] = float(fixation.anchor_s)
        # room at the anchor: half the local usable width, or the target radius in free space
        tol = 0.5 * dev_scale if (dev_scale and dev_scale < FREE_SPACE_CLEARANCE_M) else self.target_radius
        plan['arrival_tol'] = max(float(tol), 1e-3)
        self.plan = plan
        return plan
