"""Difficulty-budget planning horizon and intermittent replan scheduling.

Both modules are fit to the gaze-cursor data in eval/eval-gaze-cursor and are
independently switchable so each can be ablated against the baseline
(fixed horizon, replan every step):

1. Difficulty-budget horizon (``horizon_mode: "budget"``). At each planning
   event the gaze anchor sits a lookahead distance h ahead of the cursor such
   that the accumulated path difficulty equals a per-person budget
   (lookahead_difficulty.py; pooled fit rho=0.52 vs 0.41 for a width power
   law):

       integral_{s0}^{s0+h} (W_ref/W(s))^gamma / W_ref ds = D0

   W is the local usable corridor width (compute_clearance_profile returns
   wl+wr for corridor constraints; in free space it saturates, the density
   -> 0, and h runs to the path end — capped). The integral is
   dimensionless, so D0 transfers between task units and meters unchanged.

   The density is WIDTH-ONLY: gaze lead shrinks with narrowing width and
   with nothing else. An additive curvature term lam*|kappa| was removed
   2026-08-24 — corner-dwell analysis showed humans dwell ~1.4-1.5x longer
   when the anchor sits at a corner, and the model reproduces this
   emergently (constant lead + apex slowdown -> longer catch-up) exactly
   when lam=0, while a fitted lam>0 (participant B) shrank the corner lead
   enough to cancel the effect and doubled the anchor overshoot; the CV
   gain of lam>0 was ~1% (lookahead_floor_summary.json).

   The lookahead additionally has a visuomotor-delay floor h >= v * T_min
   (refit_floor.py): human leads at the narrowest widths do NOT shrink
   proportionally to 1/W as the pure budget predicts (human median lead
   ~0.015 at W=0.01 vs 0.006 for the bare budget) — the gaze must stay at
   least one reaction time ahead of the cursor at its current speed.

2. Intermittent replanning (``replan_mode: "intermittent"``). The gaze data
   (intermittency_analysis.py) show plan-execute-replan cycles at 2-2.7 Hz,
   not per-step replanning: gaze saccades to the anchor, the cursor closes
   the gap open-loop, and the next planning event fires when the cursor
   reaches the anchor plus a post-arrival latency (median 0.17-0.21 s,
   CV ~0.9 across participants), during which the old plan keeps executing
   (hence the observed overshoot of the anchor to ~-0.5x lead). Triggers,
   in priority order:
     * deviation: the realised cursor has drifted from the planned position
       by more than a fraction of the local usable width (the open-loop
       plan is invalid — replan early). Matches the ~20% of human cycles
       that end BEFORE anchor crossing (frac_crossed 0.76-0.85).
     * arrival at anchor + tau (tau optionally lognormal, median
       replan_latency_s, CV replan_latency_cv).
     * plan exhaustion (backstop).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Numerical floors: corridor width (m) below which the density is clamped,
# and reference speed (m/s) for the time-to-traverse integral.
_WIDTH_FLOOR = 1e-3
_SPEED_FLOOR = 0.02


class DifficultyBudgetHorizon:
    """Arc-length lookahead from a difficulty budget, with time conversion.

    Precomputes cumulative difficulty and cumulative traversal time on the
    trajectory's profile grid, so each planning event is an O(log n) query.
    """

    def __init__(
        self,
        s_profile: np.ndarray,
        width_profile: np.ndarray,
        v_ref_profile: np.ndarray,
        D0: float,
        T_min: float = 0.0,
        gamma: float = 1.0,
        W_ref: float = 0.026,
    ):
        self.s = np.asarray(s_profile, dtype=float)
        self.D0 = float(D0)
        self.T_min = float(T_min)
        self.gamma = float(gamma)
        self.W_ref = float(W_ref)

        width = np.clip(np.asarray(width_profile, dtype=float), _WIDTH_FLOOR, None)
        # Width-only density (W_ref/W)^gamma / W_ref: gamma=1 reduces exactly
        # to 1/W (W_ref cancels). gamma<1 makes the lead sublinear in width —
        # on a uniform corridor h = D0 * W^gamma * W_ref^(1-gamma) — matching
        # the human lead ~ w^0.66 scaling that a linear budget cannot. W_ref
        # (geometric mean of the studied widths) keeps the density in
        # 1/length so D0 stays dimensionless; it is not a behavioural knob —
        # rescaling it is absorbed by D0. See the module docstring for why
        # there is deliberately NO curvature term here.
        density = ((self.W_ref / width) ** self.gamma) / self.W_ref

        v_ref = np.clip(np.asarray(v_ref_profile, dtype=float), _SPEED_FLOOR, None)

        # Cumulative integrals on the grid (trapezoid), for O(1) lookups.
        ds = np.diff(self.s)
        self.cum_D = np.concatenate(
            [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * ds)])
        self.cum_T = np.concatenate(
            [[0.0], np.cumsum(0.5 * (1.0 / v_ref[1:] + 1.0 / v_ref[:-1]) * ds)])

    def anchor(self, s0: float, v_now: float = 0.0) -> float:
        """Arc length of the planning anchor: budget spent from s0, floored
        at the visuomotor-delay lookahead v_now * T_min, capped at the path
        end."""
        s0 = float(np.clip(s0, self.s[0], self.s[-1]))
        d0 = float(np.interp(s0, self.s, self.cum_D))
        d_target = d0 + self.D0
        if d_target >= self.cum_D[-1]:
            s_budget = float(self.s[-1])
        else:
            # cum_D is nondecreasing: invert by interpolation
            s_budget = float(np.interp(d_target, self.cum_D, self.s))
        s_floor = s0 + max(0.0, float(v_now)) * self.T_min
        return float(min(max(s_budget, s_floor), self.s[-1]))

    def traverse_time(self, s0: float, s1: float) -> float:
        """Time (s) to traverse [s0, s1] at the reference speed profile."""
        t0 = float(np.interp(s0, self.s, self.cum_T))
        t1 = float(np.interp(s1, self.s, self.cum_T))
        return max(0.0, t1 - t0)


@dataclass
class ReplanEvent:
    """Diagnostics for one planning event (the model's 'fixation onset')."""
    step: int
    t: float
    theta: float          # cursor arc-length progress at solve time
    anchor: float         # planned anchor arc length (theta + lookahead)
    n_steps: int          # solve horizon length (steps)
    trigger: str          # "init" | "arrival+latency" | "deviation" |
                          # "exhausted" | "every_step"


@dataclass
class ReplanScheduler:
    """Plan-execute-replan trigger state machine.

    Modes:
      * every_step: replan at every simulation step (baseline).
      * intermittent: execute the current plan open-loop; replan when the
        cursor's arc-length progress reaches the plan's anchor plus a
        post-arrival latency tau (the old plan keeps executing during tau),
        when the realised cursor deviates from the planned position by more
        than deviation_frac of the local usable width (early interrupt), or
        immediately when the plan is exhausted (backstop).

    With latency_cv > 0 the latency of each cycle is drawn from a lognormal
    with median latency_steps and coefficient of variation latency_cv
    (gaze post-arrival dwell: median 0.17-0.21 s, CV ~0.9), via the global
    numpy RNG so runs stay reproducible under np.random.seed.
    """
    mode: str = "every_step"
    latency_steps: int = 4          # tau / dt, rounded (median when cv > 0)
    latency_cv: float = 0.0         # lognormal CV of the per-cycle latency
    # Upper cap on a single latency draw (steps; 0 = uncapped). The human
    # post-arrival dwell statistics were measured under the intermittency
    # analysis's 1.5 s event-duration filter, so the model's lognormal must
    # not produce pauses that data could never contain: uncapped tail draws
    # (P(tau > 1 s) ~ 0.7-2.3 %/cycle at the fitted CVs) froze the gaze
    # anchor for the rest of a trial while the cursor ran on open-loop.
    latency_max_steps: int = 0
    deviation_frac: float = 0.0     # early-replan threshold; 0 disables
    # Explicit minimum open-loop interval (steps): after a replan no
    # feedback-driven trigger (deviation, arrival) may fire until this many
    # plan steps have executed — the psychological refractory period
    # (Alvarez Martin et al. 2021 fit 0.03-0.05 s). The exhaustion backstop
    # is exempt: it is plan availability, not a feedback event.
    min_open_loop_steps: int = 0
    # Dedicated RNG for latency draws so scheduler stochasticity is
    # decoupled from the motor-noise stream (toggling add_noise must not
    # change the latency sequence). None falls back to the global numpy RNG.
    rng: Optional[np.random.Generator] = None
    plan_len: int = 0               # steps in the current plan
    anchor: float = np.inf          # arc-length trigger point
    plan_idx: int = 1               # next velocity index to execute (1-based)
    latency_left: Optional[int] = None
    events: List[ReplanEvent] = field(default_factory=list)

    def _sample_latency_steps(self) -> int:
        """Per-cycle latency draw (steps). Deterministic when latency_cv=0."""
        if self.latency_cv <= 0.0 or self.latency_steps <= 0:
            return self.latency_steps
        sigma = float(np.sqrt(np.log1p(self.latency_cv ** 2)))
        z = (self.rng.standard_normal() if self.rng is not None
             else np.random.standard_normal())
        mult = float(np.exp(sigma * z))
        tau = max(0, int(round(self.latency_steps * mult)))
        if self.latency_max_steps > 0:
            tau = min(tau, self.latency_max_steps)
        return tau

    def needs_replan(self, theta_now: float,
                     deviation_ratio: Optional[float] = None) -> Optional[str]:
        """Trigger check at the top of a step; returns the trigger name or
        None. Call before executing the step. deviation_ratio is the realised
        cursor's distance from the planned position divided by the local
        usable width (None when unavailable or the trigger is disabled)."""
        if self.mode == "every_step":
            return "every_step" if self.events else "init"
        if not self.events:
            return "init"
        if self.plan_idx > self.plan_len:
            return "exhausted"
        # Refractory gate: latency_left is always None here right after a
        # replan (on_replan resets it), so this cannot freeze a countdown.
        if self.plan_idx - 1 < self.min_open_loop_steps:
            return None
        if self.latency_left is not None:
            if self.latency_left <= 0:
                return "arrival+latency"
            self.latency_left -= 1
            return None
        # Pre-arrival only: once the post-arrival countdown runs, the next
        # replan is imminent anyway (the human analog: gaze already moved on).
        if (self.deviation_frac > 0.0 and deviation_ratio is not None
                and deviation_ratio > self.deviation_frac):
            return "deviation"
        if theta_now >= self.anchor:
            # Arrival at the anchor starts the latency countdown; the replan
            # fires tau steps later while the plan keeps executing.
            tau = self._sample_latency_steps()
            if tau <= 0:
                return "arrival+latency"
            self.latency_left = tau - 1
        return None

    def on_replan(self, event: ReplanEvent, anchor: float, plan_len: int):
        self.events.append(event)
        self.anchor = float(anchor)
        self.plan_len = int(plan_len)
        self.plan_idx = 1
        self.latency_left = None

    def on_step_executed(self):
        self.plan_idx += 1

    @property
    def wants_theta(self) -> bool:
        """Whether the trigger check needs the cursor's current arc-length
        progress (saves a spline projection when it does not)."""
        return (
            self.mode == "intermittent"
            and bool(self.events)
            and self.latency_left is None
            and self.plan_idx <= self.plan_len
            and self.plan_idx - 1 >= self.min_open_loop_steps
        )
