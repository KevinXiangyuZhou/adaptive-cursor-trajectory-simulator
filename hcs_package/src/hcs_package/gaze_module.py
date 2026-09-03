"""Gaze module: WHERE the eyes anchor and WHEN the motor plan is renewed.

This is the paper's gaze module. Per fixation it produces the anchor (the
difficulty-budget lookahead with its floors), the plan deadline t_plan (and
its node counts), and the fixation's pace; per step it runs the replan
triggers (arrival + latency, deviation interrupt, exhaustion backstop) that
end the current fixation. Everything speed-shaped that the motor module is
given — the via-point and its deadline — originates here.

Finalized cycle design (2026-09-03): in constrained space the plan deadline
is the GAM-predicted traversal time of the lead, t_plan = integral ds/v(s)
with v from speed_model.GAMSpeedModel (clearance, |kappa|, anticipatory
kappa over the next 50 mm; raw, no clamp). This replaced the hand-designed
deadline rule stack (width-scaled base time plan_width_time_exp and the
turning-time terms plan_turn_time_s / plan_turn_width_exp — dropped; last
carried at the commit before this change). Free space keeps the bare base
deadline T0 (pointing: the GAM is a corridor model). The lead/v_max and
bounded-acceleration floors and the numerical >=3-node horizon floor
remain. The MPCC stays anchor-driven — speed still emerges as
lookahead / deadline; the GAM only decides how much time the plan gets.

State is per trajectory: construct one GazeModule per generate_* call.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .intermittent import DifficultyBudgetHorizon, ReplanEvent, ReplanScheduler
from .model import FREE_SPACE_CLEARANCE_M


@dataclass
class Fixation:
    """One planning event: the gaze module's per-solve decision."""
    trigger: str            # what ended the previous fixation
    theta0: float           # cursor arc-length progress at solve time
    anchor_s: float         # fixation anchor (arrival geometry; capped at path end)
    solve_anchor_s: float   # via-point target for the solve (differs from
                            # anchor_s only when the 3-node numerical horizon
                            # floor stretches the plan: schedule position at
                            # the horizon end, pace unchanged)
    n_base: int             # deadline node count (plan reaches the anchor here)
    n_solve: int            # solve horizon (deadline + open-loop padding, clamped)
    pace: float             # planned progress rate lead / t_plan (m/s)
    t_plan: float           # planned time-to-anchor (s)
    t0: float               # wall-clock time of the fixation onset
    w_solve: Optional[float]  # local usable width at solve time (None: no profile;
                              # 2R stands in where the task is unconstrained)


class GazeModule:

    def __init__(
        self,
        reference_path,
        budget_horizon: DifficultyBudgetHorizon,
        scheduler: ReplanScheduler,
        clearance_profile: Optional[Tuple],
        curvature_profile: Optional[Tuple],
        *,
        interval: float,
        tau_steps: int,
        target_radius: float,
        replan_mode: str,
        arrival_mode: str,
        deviation_frac: float,
        anchor_lead_floor: bool,
        plan_deadline_s: float,
        plan_vmax: float,
        traversal_speed_model,
        acc_max: float,
        horizon_min_steps: int,
        horizon_max_steps: int,
    ):
        self.reference_path = reference_path
        self.budget_horizon = budget_horizon
        self.scheduler = scheduler
        self.clearance_profile = clearance_profile
        self.curvature_profile = curvature_profile
        self.interval = interval
        self.tau_steps = tau_steps
        self.target_radius = float(target_radius)
        self.replan_mode = replan_mode
        self.arrival_mode = arrival_mode
        self.deviation_frac = float(deviation_frac)
        self.anchor_lead_floor = bool(anchor_lead_floor)
        self.plan_deadline_s = float(plan_deadline_s)
        self.plan_vmax = float(plan_vmax)
        self.acc_max = float(acc_max)
        self.horizon_min_steps = int(horizon_min_steps)
        self.horizon_max_steps = int(horizon_max_steps)
        # Warm guess for the arc-length projection, shared across all
        # projections of this trajectory.
        self.theta_track = 0.0
        # Traversal-time table (finalized deadline): cumulative
        # T(s) = integral ds / v_gam(s) on the profile grid, v_gam from the
        # raw GAM (clearance = W/2 clipped to its trained range, |kappa|,
        # anticipatory kappa = max |kappa| over the next KAPPA_AHEAD_M, and
        # runway = arc distance to the next steering demand — the nearest
        # point ahead with |kappa| > K_DEMAND_RAD_M, or the path end when
        # none remains). None when no speed model or no profiles
        # (pointing-only tasks) — then the bare T0 deadline applies
        # everywhere.
        self._cum_T = None
        if (traversal_speed_model is not None and clearance_profile is not None
                and curvature_profile is not None):
            from .speed_model import KAPPA_AHEAD_M, K_DEMAND_RAD_M
            s_prof, w_prof = clearance_profile
            _, k_prof = curvature_profile
            s_prof = np.asarray(s_prof, float)
            kappa = np.abs(np.asarray(k_prof, float))
            ds = float(np.mean(np.diff(s_prof))) if len(s_prof) > 1 else 1.0
            win = max(1, int(round(KAPPA_AHEAD_M / max(ds, 1e-9))) + 1)
            # forward-window max via reversed sliding maximum
            from numpy.lib.stride_tricks import sliding_window_view
            pad = np.concatenate([kappa, np.full(win - 1, kappa[-1])])
            k_ahead = sliding_window_view(pad, win).max(axis=1)
            # runway: distance to the next demand point via searchsorted
            # (grid-spacing independent; the target end is a braking point)
            demand_s = s_prof[kappa > K_DEMAND_RAD_M]
            nxt = np.full(len(s_prof), s_prof[-1])
            if len(demand_s):
                j = np.searchsorted(demand_s, s_prof, side='left')
                has = j < len(demand_s)
                nxt[has] = demand_s[np.minimum(j, len(demand_s) - 1)][has]
            runway = np.maximum(nxt - s_prof, 0.0)
            v = traversal_speed_model.predict_speed_raw(
                np.asarray(w_prof, float) / 2.0, kappa, k_ahead, runway)
            v = np.clip(v, 0.02, None)  # numerical speed floor
            dgrid = np.diff(s_prof)
            self._traversal_s = s_prof
            self._cum_T = np.concatenate(
                [[0.0], np.cumsum(0.5 * (1.0 / v[1:] + 1.0 / v[:-1]) * dgrid)])

    def _traversal_time(self, s0: float, s1: float) -> Optional[float]:
        """GAM traversal time of [s0, s1]; None when no table is available."""
        if self._cum_T is None:
            return None
        t0 = float(np.interp(s0, self._traversal_s, self._cum_T))
        t1 = float(np.interp(s1, self._traversal_s, self._cum_T))
        return max(0.0, t1 - t0)

    # ------------------------------------------------------------- triggers

    def check_trigger(self, cursor_pos, plan) -> Optional[str]:
        """Run the per-step replan triggers against the executing plan.

        Returns the trigger name (a new fixation must be planned) or None
        (keep executing open-loop). ``plan`` is the motor module's current
        plan dict (None before the first solve).
        """
        scheduler = self.scheduler
        theta_now = None
        if scheduler.wants_theta:
            theta_now = float(self.reference_path.find_closest_theta(
                cursor_pos, initial_guess=self.theta_track))
            self.theta_track = theta_now
        # Early-replan interrupt: realised drift from the open-loop plan,
        # scaled by the usable width at the last solve. Zero until at
        # least one step of the plan has executed (the plan starts at the
        # realised cursor state, and noiseless execution tracks exactly).
        deviation_ratio = None
        if (scheduler.mode == 'intermittent'
                and self.deviation_frac > 0.0
                and plan is not None and scheduler.plan_idx >= 2
                and plan.get('dev_scale')):
            k_exec = min(scheduler.plan_idx - 2, plan['n_steps'] - 1)
            dev = float(np.hypot(
                cursor_pos[0] - plan['pos_x'][k_exec],
                cursor_pos[1] - plan['pos_y'][k_exec]))
            deviation_ratio = dev / plan['dev_scale']
        theta_for_trigger = theta_now if theta_now is not None else -np.inf
        if (self.arrival_mode == 'distance' and plan is not None
                and plan.get('anchor_xy') is not None and scheduler.wants_theta):
            d_anchor = float(np.hypot(cursor_pos[0] - plan['anchor_xy'][0],
                                      cursor_pos[1] - plan['anchor_xy'][1]))
            # Arrival = reached OR passed the fixation point: within the local
            # room of the anchor point, or progress beyond its arc position.
            # (Distance alone is missed at speed — 13 mm/step at 0.26 m/s skips
            # a 5 mm window — leaving the fixation alive and the schedule
            # sliding ahead; progress alone stalls in a corner's wedge.)
            passed = (theta_now is not None and plan.get('anchor_s') is not None
                      and theta_now >= plan['anchor_s'] - 1e-9)
            theta_for_trigger = np.inf if (d_anchor <= plan['arrival_tol'] or passed) else -np.inf
        return scheduler.needs_replan(theta_for_trigger, deviation_ratio=deviation_ratio)

    # ------------------------------------------------------------ fixations

    def _local_room(self, theta: float) -> float:
        """Arrival tolerance at ``theta``: half the local usable width in a
        corridor, the target radius where the task is unconstrained."""
        s_cl, c_cl = self.clearance_profile
        w_here = float(np.interp(theta, s_cl, c_cl))
        return 0.5 * w_here if w_here < FREE_SPACE_CLEARANCE_M else self.target_radius

    def plan_fixation(self, cursor_pos, cursor_vel, current_time: float,
                      trigger: str) -> Fixation:
        """Produce the next fixation: anchor, deadline, pace, solve horizon."""
        ref = self.reference_path
        total = float(ref.total_length)
        theta0 = float(ref.find_closest_theta(
            cursor_pos, initial_guess=self.theta_track))
        self.theta_track = theta0

        solve_anchor_s = None
        s_from = min(theta0, total)
        anchor_s = self.budget_horizon.anchor(
            s_from, v_now=float(np.hypot(cursor_vel[0], cursor_vel[1])))
        if self.anchor_lead_floor and self.clearance_profile is not None:
            # Lead floor BEFORE the deadline: if the floor extends the anchor,
            # the deadline must stretch with it (review finding: floor applied
            # after n_base made the via-point demand the extended lead in the
            # old time).
            room0 = self._local_room(theta0)
            anchor_s = min(max(float(anchor_s), theta0 + room0), total)

        # Fixed plan duration: speed = lookahead / deadline.
        # Round half-up with a float guard (0.175/0.05 = 3.4999.. -> 4).
        lead_now = max(0.0, float(anchor_s) - theta0)
        # Finalized deadline: in constrained space, the GAM-predicted
        # traversal time of the lead — t_plan = integral ds / v_gam(s) over
        # [theta0, anchor]. The fitted GAM subsumes the old rule stack (the
        # width-scaled T0 and the turning-time terms): v_gam slows with
        # narrowness (~W^1) and with curvature ahead, so narrow corridors
        # and bends stretch the deadline without any hand-set exponents.
        # Free space keeps the bare T0 (pointing; the GAM is a corridor
        # model), and the lead/v_max floor guards far free-space targets.
        t_gam = None
        w_loc0 = None
        if self.clearance_profile is not None:
            s_cw0, c_cw0 = self.clearance_profile
            w_loc0 = float(np.interp(theta0, s_cw0, c_cw0))
        if (w_loc0 is not None and 0.0 < w_loc0 < FREE_SPACE_CLEARANCE_M):
            t_gam = self._traversal_time(theta0, float(anchor_s))
        if t_gam is not None:
            t_plan = max(t_gam, lead_now / max(self.plan_vmax, 1e-6))
        else:
            t_plan = max(self.plan_deadline_s, lead_now / max(self.plan_vmax, 1e-6))
        if self.acc_max > 0.0:
            # ... nor faster than the bounded hand can cover the lead from
            # its current speed: 1/2 a t^2 + v t = h.
            v_now = float(np.hypot(cursor_vel[0], cursor_vel[1]))
            t_acc = (-v_now + np.sqrt(v_now * v_now + 2.0 * self.acc_max * lead_now)) / self.acc_max
            t_plan = max(t_plan, float(t_acc))
        n_base = max(1, int(np.floor(t_plan / self.interval + 0.5 + 1e-9)))
        # Numerical floor: >= 3 nodes (shorter horizons destabilise the
        # solve).
        n_min = 3
        if n_base < n_min:
            # Horizon floor: demand the schedule position at the horizon
            # end (pace unchanged), not the fixation point itself.
            n_base = n_min
            pace0 = lead_now / max(t_plan, 1e-6)
            solve_anchor_s = min(theta0 + pace0 * n_base * self.interval, total)
        pace = lead_now / max(t_plan, 1e-6)
        if os.environ.get('HCS_DEBUG_PLAN'):
            print(f"[plan] t={current_time:.2f} s0={theta0*1000:.0f}mm lead={lead_now*1000:.1f}mm "
                  f"t_plan={t_plan:.3f}s pace={pace:.3f} n_base={n_base} solve_anchor={'%.0f' % (solve_anchor_s*1000) if solve_anchor_s is not None else '-'}", flush=True)
        anchor_s = min(float(anchor_s), total)
        if self.replan_mode == 'intermittent':
            # The plan must survive the post-arrival latency (the old plan
            # keeps executing past the anchor), plus the same margin again as
            # slack for arrival running later than the reference-speed
            # estimate (start-up, corners, noise). This is plan availability,
            # not behaviour: the replan time is still set by arrival + tau;
            # exhaustion only backstops.
            n_solve = n_base + 2 * self.tau_steps
        else:
            n_solve = n_base
        n_solve = int(np.clip(n_solve, self.horizon_min_steps, self.horizon_max_steps))

        # Deviation-trigger scale at solve time: the local usable width where
        # the task is constrained; in free space the task's own accuracy
        # scale, the target diameter (the motor module widens it to the plan
        # span for free-space plans).
        w_solve = None
        if self.clearance_profile is not None:
            s_cl, c_cl = self.clearance_profile
            w_here = float(np.interp(theta0, s_cl, c_cl))
            w_solve = (w_here if w_here < FREE_SPACE_CLEARANCE_M
                       else 2.0 * self.target_radius)

        return Fixation(
            trigger=trigger, theta0=theta0, anchor_s=float(anchor_s),
            solve_anchor_s=(float(solve_anchor_s) if solve_anchor_s is not None
                            else float(anchor_s)),
            n_base=n_base, n_solve=n_solve, pace=pace, t_plan=t_plan,
            t0=current_time, w_solve=w_solve,
        )

    def commit(self, step: int, current_time: float, fixation: Fixation, plan):
        """Register the solved plan with the scheduler (fixation onset)."""
        ev = ReplanEvent(step=step, t=current_time, theta=fixation.theta0,
                         anchor=fixation.anchor_s, n_steps=fixation.n_solve,
                         trigger=fixation.trigger)
        ev.arrival_tol = plan.get('arrival_tol')
        self.scheduler.on_replan(ev, anchor=fixation.anchor_s,
                                 plan_len=plan['n_steps'])

    # ----------------------------------------------------------- step hooks

    @property
    def warm_shift(self) -> int:
        """Steps executed since the previous solve (MPCC warm-start shift)."""
        return max(1, self.scheduler.plan_idx - 1)

    @property
    def plan_step_index(self) -> int:
        """1-based index of the next plan step to execute."""
        return self.scheduler.plan_idx

    def on_step_executed(self):
        self.scheduler.on_step_executed()

    @property
    def events(self):
        return self.scheduler.events
