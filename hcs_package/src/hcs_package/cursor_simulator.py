"""Playwright-compatible cursor simulator for generating human-like trajectories."""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from .model import model, FREE_SPACE_CLEARANCE_M

# Finite stand-in for "no constraint" when a consumer needs a finite number
# (the GAM's log-clearance feature; interpolation). Purely numerical: both
# the GAM (saturated at its ceiling) and the free-space mask (threshold
# FREE_SPACE_CLEARANCE_M) are insensitive to anything this large.
W_TASK_FINITE_CAP = 1.0
from .mpcc_model import reset_warm_start
from .params import SteeringModelInput, BumpParams, EnvParams, TunnelInfo
from .reference_path import ReferencePath, generate_optimal_reference_path, densify_polyline
from .noise import single_step_motor_and_device_noise
from .constraints import ConstraintConfig, ConstraintRegion, ConstraintType, PathConstraint, RectangleConstraint, PolygonConstraint
from .constraint_utils import parse_constraints_from_json, convert_constraints_to_corridor_bounds
from .adapt import compute_clearance_profile, compute_curvature_rate_profile, compute_curvature_spike_profile
from .intermittent import DifficultyBudgetHorizon, ReplanEvent, ReplanScheduler

#imports for baseline
from .baseline_model import generate_baseline_mpc
from .mpcc_model import _build_A_vel_from_jerk, _build_A_pos_from_jerk


class CursorSimulator:
    """
    A simulator for generating human-like cursor movements.
    
    This class provides a simple interface for generating cursor trajectories
    that can be used with Playwright's mouse movement functions.
    
    Example:
        >>> simulator = CursorSimulator()                          # default: office_worker
        >>> simulator = CursorSimulator("gamer")                   # built-in persona name
        >>> simulator = CursorSimulator("/path/to/custom.json")    # custom config file
        >>> trajectory = simulator.generate_trajectory_with_waypoints(task_file="task.json")
        >>> for x, y, delay in trajectory:
        ...     page.mouse.move(x, y)
        ...     await page.wait_for_timeout(int(delay * 1000))

    Available built-in personas:
        office_worker, novice, gamer, young_children, fatigued, motor_impaired
    """

    _CONFIG_DIR = Path(__file__).parent / "user_configurations"
    
    def __init__(self, user_config: Optional[Union[str, Path]] = None):
        """
        Initialize the cursor simulator.
        
        Args:
            user_config: User persona configuration. Can be:
                - None: uses "office_worker" (default)
                - A built-in persona name: "office_worker", "novice", "gamer",
                  "young_children", "fatigued", "motor_impaired"
                - A path to a custom JSON configuration file
        """
        config = {
            "Interval": 0.05,
            "Tp": 0.05,
            "Th": 0.3,
            "nc": [0.2, 0.02],
            "forearm": 0.357,
            "mouseGain": 1,
            "planner_weights": {
                "jerk": 1.2e-06,
                "progress": 0.1e-06,
                "constraint": 50,
                "contour": 20,
                "lag": 0.05,
                "desired_speed": 0.2, #default 0.2
                "goal_precision": 0.0  # OFF by default (2026-08-21, matching the gaze-cursor
                                       # branch decision): seed-averaged ablation on the gaze
                                       # cohort shows the well explains only ~10% of the Fitts
                                       # slope at 1e-4 and nothing in steering. The guarded term
                                       # is kept as a fittable switch — on the Prolific data it
                                       # was the only term reproducing the endpoint-depth-by-R
                                       # trend (deeper settling in larger targets).
            },
            "planner_margin": 0.0,
            # Carry the realised cursor acceleration into the next MPC solve.
            # The planner is a jerk-driven 3rd-order model; if the plant is
            # re-seeded with a=0 every step, the acceleration the plan builds
            # up over the horizon is never realised (per-step dv is only
            # 0.5*j0*dt^2), so braking is much weaker than planned -> late,
            # overshooting arrivals in free space.
            "carry_acceleration": True,
            # Consecutive time inside the target that ends a trajectory (s).
            "dwell_s": 0.25,
            "add_noise": True,
            "ddm_enabled": False,
            "random_seed": 1000,
            # --- planning-horizon / replanning modules (gaze-cursor fits) ---
            # horizon_mode "fixed": pred horizon = Th/Interval steps (baseline).
            # horizon_mode "budget": per-solve horizon from the difficulty
            # width-only budget  ∫(W_ref/W)^gamma/W_ref ds = D0
            # (eval/eval-gaze-cursor refit_floor.py; unit-invariant; the
            # lam|kappa| curvature term was removed 2026-08-24 — legacy
            # configs with a budget.lam key still load, the key is ignored),
            # floored at
            # the visuomotor-delay lookahead v * T_min (refit_floor.py) —
            # which also floors the solve horizon in time so it cannot
            # collapse near the path end. Defaults are the pooled refit
            # (cross-validated, magnitude-calibrated) sim_params from
            # results/lookahead_floor_summary.json.
            # gamma < 1 makes the lookahead sublinear in width (human lead ~
            # w^0.66); W_ref is the dimensional reference (geometric mean of
            # the studied widths), absorbed by D0 — not a behavior knob.
            "horizon_mode": "fixed",
            "budget": {"D0": 1.66, "T_min": 0.1,
                       "gamma": 1.0, "W_ref": 0.026},
            # replan_mode "every_step": re-solve each step (baseline).
            # replan_mode "intermittent": execute the plan open-loop and
            # re-solve on arrival at the planned anchor + a post-arrival
            # latency (intermittency_analysis.py: median dwell 0.17-0.21 s,
            # CV ~0.9). replan_latency_cv > 0 draws each cycle's latency
            # from a lognormal with that median and CV; 0 = deterministic.
            # replan_deviation_frac > 0 adds an early-replan interrupt when
            # the realised cursor drifts from the planned position by more
            # than that fraction of the local usable width. Default 0.15 is
            # calibrated so ~20% of cycles end before anchor crossing,
            # matching the human non-crossed fraction (frac_crossed ~0.8);
            # inert under replan_mode "every_step".
            "replan_mode": "every_step",
            "replan_latency_s": 0.19,
            "replan_latency_cv": 0.89,
            # Cap on a single latency draw (s). Matches the 1.5 s event-
            # duration filter under which the human dwell median/CV were
            # measured — uncapped lognormal tails froze the gaze anchor for
            # the remainder of a trial (see intermittent.ReplanScheduler).
            "replan_latency_max_s": 1.5,
            "replan_deviation_frac": 0.15,
            # Explicit minimum open-loop interval (s): after a replan no
            # feedback-driven trigger (deviation, arrival) fires until this
            # much of the plan has executed — the psychological refractory
            # period (IC literature fits 0.03-0.05 s). Exhaustion is exempt.
            "min_open_loop_s": 0.05,
            # Solve-horizon clamp (steps). The fixed-mode default (Th=0.3 ->
            # 6 steps) lies inside the clamp, so the baseline is unaffected.
            # With budget T_min > 0 the binding floor is ceil(T_min/dt), not
            # horizon_min_steps.
            "horizon_min_steps": 2,
            "horizon_max_steps": 40
        }

        if user_config is None:
            user_config = "office_worker"

        config_path = Path(user_config)

        if not config_path.exists():
            builtin_path = self._CONFIG_DIR / f"{user_config}.json"
            if builtin_path.exists():
                config_path = builtin_path
            else:
                available = [f.stem for f in self._CONFIG_DIR.glob("*.json")]
                raise FileNotFoundError(
                    f"User config not found: '{user_config}'. "
                    f"Available built-in personas: {available}"
                )

        with open(config_path, 'r') as f:
            user_cfg = json.load(f)

        # Deep-merge planner_weights so partial overrides work
        if 'planner_weights' in user_cfg:
            config['planner_weights'].update(user_cfg.pop('planner_weights'))
        config.update(user_cfg)

        self.interval = config['Interval']
        self.forearm = config['forearm']
        
        TH_SCALE = 1.0  # change this value to scale prediction horizon (e.g. 1.5, 2.0)
        th = config['Th'] * TH_SCALE
        self.pred_horizon = max(1, int(round(th / self.interval)))

        tp = config['Tp']
        self.tp = max(1, int(round(tp / self.interval)))

        self.nc = list(config['nc'])
        self.planner_weights = config['planner_weights']
        self.planner_margin = config['planner_margin']
        self.carry_acceleration = bool(config.get('carry_acceleration', True))
        self.dwell_s = float(config.get('dwell_s', 0.25))
        self.add_noise = config['add_noise']

        self.horizon_mode = str(config.get('horizon_mode', 'fixed'))
        if self.horizon_mode not in ('fixed', 'budget'):
            raise ValueError(f"horizon_mode must be 'fixed' or 'budget', got {self.horizon_mode!r}")
        budget_cfg = config.get('budget') or {}
        # Fallbacks match the default config above (single source of truth
        # for "unset": the pooled cross-validated refit).
        self.budget_D0 = float(budget_cfg.get('D0', 1.66))
        self.budget_T_min = float(budget_cfg.get('T_min', 0.1))
        self.budget_gamma = float(budget_cfg.get('gamma', 1.0))
        self.budget_W_ref = float(budget_cfg.get('W_ref', 0.026))
        self.budget_curvature_weighted = bool(budget_cfg.get('curvature_weighted', False))
        self.replan_mode = str(config.get('replan_mode', 'every_step'))
        if self.replan_mode not in ('every_step', 'intermittent'):
            raise ValueError(f"replan_mode must be 'every_step' or 'intermittent', got {self.replan_mode!r}")
        self.replan_latency_s = float(config.get('replan_latency_s', 0.19))
        self.replan_latency_cv = float(config.get('replan_latency_cv', 0.89))
        self.replan_deviation_frac = float(config.get('replan_deviation_frac', 0.15))
        self.min_open_loop_s = float(config.get('min_open_loop_s', 0.05))
        self.replan_latency_max_s = float(config.get('replan_latency_max_s', 1.5))
        self.horizon_min_steps = int(config.get('horizon_min_steps', 2))
        self.horizon_max_steps = int(config.get('horizon_max_steps', 40))
        # Anchor-drive planning (speed_model type "none"): the plan's
        # intended time-to-anchor. Every solve covers plan_deadline_s of
        # movement and asks the cursor to be AT the gaze anchor at that
        # deadline (a via point, not a rest point), so cruise speed is the
        # budget lookahead divided by this deadline — v = h(W)/T_plan —
        # and no speed is prescribed anywhere. Gaze data: the time from
        # fixation onset to the cursor crossing the anchor is ~0.19 s and
        # width-invariant (lead and speed both scale with width, their
        # ratio does not); it lengthens in curved tunnels, which the
        # planner reproduces by trading punctuality against effort.
        self.plan_deadline_s = float(config.get('plan_deadline_s', 0.19))
        # Coast-safety (anchor-drive): the deadline state's ballistic
        # continuation over the replan latency must stay inside the walls.
        self.coast_safety = bool(config.get('coast_safety', True))
        # Maximum hand speed (m/s): the plan deadline cannot demand arrival
        # sooner than lookahead / v_max. Inert in tunnels (lookahead <= 5 cm);
        # in pointing it stretches the deadline for far targets so the drive
        # asks for a physiological peak speed instead of an absurd one.
        self.plan_vmax = float(config.get('plan_vmax', 0.8))
        # Turning-time deadline (s per radian of turning inside the lead); 0 = off.
        self.plan_turn_time_s = float(config.get('plan_turn_time_s', 0.0))
        self.plan_turn_width_exp = float(config.get('plan_turn_width_exp', 0.0))
        # Single plan per fixation with a pace-holding tail (item-4 ablation of the
        # motor-replan loop): use with motor_period_s = 0.
        self.anchor_tail_pace = bool(config.get('anchor_tail_pace', False))
        # Trial abort on leaving the tunnel by more than this margin (m); the
        # steering experiment restarts such trials. None = never abort.
        self.abort_on_breach_m = config.get('abort_on_breach_m', None)
        # Arrival rule for the replan trigger: "progress" = cursor's arc-length
        # projection reaches the anchor (stalls in the wedge of a sharp corner,
        # where every inside point projects to the apex); "distance" = cursor
        # within the local room (half-width; target radius in free space) of
        # the anchor point.
        self.arrival_mode = str(config.get('arrival_mode', 'progress'))
        # Anchor lead floor: the anchor must be at least one arrival tolerance
        # (the local room) ahead of the cursor at planning time — one cannot
        # plan toward a point one has already arrived at. Without it the
        # budget parks the anchor on a corner apex, the cursor reaches it and
        # the via-point drive vanishes (stalls of ~1 s per corner).
        self.anchor_lead_floor = bool(config.get('anchor_lead_floor', False))
        # BUMP-style motor period (s): between fixation events, re-solve the
        # MPCC every motor_period_s toward the CURRENT fixation's progress
        # schedule (anchor slid along it); a plan's tail is never executed
        # (Bye & Neilson BUMP; Do et al. CHI'21). 0 = off.
        self.motor_period_s = float(config.get('motor_period_s', 0.0))
        # Gaze memory (anchor-drive): the difficulty budget is charged from the
        # last fixated point rather than from the cursor (the eyes do not look
        # back), and a fixated corner is consumed — the next budget starts
        # where the corner's curvature ends. Humans fixate each corner once,
        # for longer; without memory a lumpy budget re-fixates the same apex.
        self.anchor_memory = bool(config.get('anchor_memory', False))
        self.corner_consume = bool(config.get('corner_consume', False))
        self.corner_kappa = float(config.get('corner_kappa', 60.0))
        if self.arrival_mode not in ('progress', 'distance'):
            raise ValueError("arrival_mode must be 'progress' or 'distance'")
        # Diagnostics of the most recent generate_* call (replan events etc.).
        self.last_diagnostics = None

        seed = config['random_seed']
        if seed is not None:
            np.random.seed(seed)
        # Dedicated generator for replan-latency draws — decoupled from the
        # global stream so toggling add_noise does not change the latency
        # sequence (and vice versa).
        self._replan_rng = np.random.default_rng(seed)

        rp_cfg = config.get('reference_path', {})
        pw = self.planner_weights
        self.reference_path_params = {
            'w_cut': rp_cfg.get('w_cut', pw.get('w_cut', 0.01)),
            'w_suppress': rp_cfg.get('w_suppress', pw.get('w_suppress', 0.0)),
            'w_width_exp': rp_cfg.get('w_width_exp', pw.get('w_width_exp', 1.0)),
            'cut_window_frac': rp_cfg.get('cut_window_frac', pw.get('cut_window_frac', 0.05)),
            'global_clearance_ref': rp_cfg.get('global_clearance_ref', pw.get('global_clearance_ref', 0.025)),
        }

        self.speed_model = self._load_speed_model(config, config_path)
        # speed_model "none" -> anchor-drive planning (see plan_deadline_s).
        self.anchor_drive = self.speed_model is None
        if self.anchor_drive and self.horizon_mode != 'budget':
            raise ValueError("speed_model type 'none' (anchor-drive planning) requires "
                             "horizon_mode='budget': the gaze anchor is the drive")

    @staticmethod
    def _load_speed_model(config: dict, config_path: Path):
        """Load speed model from config's ``speed_model`` section.

        Format: ``"speed_model": {"type": "gam", "path": "model.pkl"}``.
        Returns None if absent. Relative paths resolve against the config
        file's directory.
        """
        sm_cfg = config.get('speed_model')
        if sm_cfg is None or not isinstance(sm_cfg, dict):
            return None

        model_type = sm_cfg.get('type', 'gam')

        if model_type == 'none':
            # No prescribed speed: anchor-drive planning.
            return None
        if model_type == 'gam':
            from .speed_model import GAMSpeedModel
            model_path = sm_cfg.get('path')
            if model_path is None:
                raise ValueError("speed_model type='gam' requires a 'path' key")
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = config_path.parent / model_path
            if not model_path.exists():
                raise FileNotFoundError(f"GAM speed model not found: {model_path}")
            return GAMSpeedModel.load(str(model_path))

        raise ValueError(f"Unknown speed_model type: {model_type!r}")

    def generate_trajectory_with_waypoints(
        self,
        task_file: Optional[Union[str, Path]] = None,
        waypoints: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[Union[Dict[str, Any], str, Path]] = None,
        screen_width: float = 1920.0,
        screen_height: float = 1080.0,
        max_steps: int = 2000,
        target_radius: float = 0.01,
        use_optimal_path: bool = True,
        return_timestamps: bool = False,
        return_reference_path: bool = False
    ) -> Union[List[Tuple[float, float, float]], Tuple[List[Tuple[float, float, float]], Any]]:
        """
        Generate a trajectory following waypoints with optional constraints.
        
        Args:
            task_file: Path to task.json file containing "waypoints", "constraints", and optionally
                      "screen_width" and "screen_height". If provided, these values from the file will be used.
                      Format: {"waypoints": [[x1, y1], [x2, y2], ...], "constraints": {...}, 
                               "screen_width": 1920, "screen_height": 1080}
            waypoints: Optional list of (x, y) waypoints in screen pixels. 
                      Ignored if task_file is provided.
            constraints: Optional constraints in JSON format. Ignored if task_file is provided.
                        Can be:
                        - Dictionary with constraint configuration
                        - Path to JSON file
                        - JSON string
            screen_width: Screen width in pixels (default: 1920). Overridden by task_file if present.
            screen_height: Screen height in pixels (default: 1080). Overridden by task_file if present.
            max_steps: Maximum simulation steps (default: 2000)
            target_radius: Target radius in normalized coordinates (default: 0.01)
            use_optimal_path: If True, generate optimal reference path from waypoints (default: True)
            return_timestamps: If True, return timestamps instead of delays (default: False)
            return_reference_path: If True, also return the reference path object (default: False)
        
        Returns:
            If return_reference_path is False: List of tuples (x, y, delay) or (x, y, timestamp)
            If return_reference_path is True: Tuple of (trajectory, reference_path)
        """
        if task_file is not None:
            task_path = Path(task_file)
            if not task_path.exists():
                raise FileNotFoundError(f"Task file not found: {task_file}")

            with open(task_path, 'r') as f:
                task_data = json.load(f)

            if "waypoints" not in task_data:
                raise ValueError("task.json must contain 'waypoints' key")

            waypoints = [tuple(wp) for wp in task_data["waypoints"]]

            if "constraints" in task_data:
                constraints = task_data["constraints"]

            # task_file overrides screen_width/screen_height parameters
            if "screen_width" in task_data:
                screen_width = float(task_data["screen_width"])
            if "screen_height" in task_data:
                screen_height = float(task_data["screen_height"])

        if waypoints is None or len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required (either from task_file or waypoints parameter)")

        # Screen pixels → normalized meters
        screen_width_m = 0.46
        screen_height_m = screen_height / screen_width * screen_width_m

        waypoints_norm = [
            (x / screen_width * screen_width_m, y / screen_height * screen_height_m)
            for x, y in waypoints
        ]

        constraint_config = None
        if constraints is not None:
            if isinstance(constraints, (str, Path)):
                path = Path(constraints)
                if path.exists():
                    with open(path, 'r') as f:
                        constraints_dict = json.load(f)
                        if "constraints" in constraints_dict:
                            constraints_dict = constraints_dict["constraints"]
                else:
                    constraints_dict = json.loads(constraints)
            else:
                constraints_dict = constraints

            constraint_config = parse_constraints_from_json(constraints_dict)

        # Build cartesian_regions first — needed for constraint-aware QP bounds
        cartesian_regions = []
        if constraint_config is not None:
            for region in constraint_config.regions:
                if isinstance(region.geometry, (RectangleConstraint, PolygonConstraint)):
                    cartesian_regions.append(region)

        tunnel_width = None
        if use_optimal_path and len(waypoints_norm) >= 2:
            # Prefer PathConstraint width; fall back to waypoint-spacing heuristic
            if constraint_config is not None:
                for region in constraint_config.regions:
                    if isinstance(region.geometry, PathConstraint):
                        w = region.geometry.width
                        tunnel_width = float(min(w) if isinstance(w, list) else w)
                        break
            if tunnel_width is None:
                distances = [
                    np.linalg.norm(np.array(waypoints_norm[i+1]) - np.array(waypoints_norm[i]))
                    for i in range(len(waypoints_norm) - 1)
                ]
                avg_distance = np.mean(distances) if distances else 0.1
                tunnel_width = min(0.1, max(0.02, avg_distance * 0.3))

            rp = self.reference_path_params
            w_cut                = rp['w_cut']
            w_width_exp          = rp['w_width_exp']
            w_center             = rp.get('w_center', 1.0)
            global_clearance_ref = rp['global_clearance_ref']

            # Cache centerline spline — reused later for curvature rate profile
            centerline_corridor_bounds = None
            centerline_spline = ReferencePath(densify_polyline(waypoints_norm), s=0.0, k=3)
            if constraint_config is not None:
                centerline_corridor_bounds = convert_constraints_to_corridor_bounds(
                    constraint_config,
                    centerline_spline,
                    default_margin=self.planner_margin,
                )

            reference_path = generate_optimal_reference_path(
                tunnel_path=waypoints_norm,
                tunnel_width=tunnel_width,
                margin=self.planner_margin,
                num_knots=None,
                w_cut=w_cut,
                w_width_exp=w_width_exp,
                w_center=w_center,
                global_clearance_ref=global_clearance_ref,
                cartesian_constraints=cartesian_regions if cartesian_regions else None,
                corridor_bounds=centerline_corridor_bounds,
                centerline_cache=centerline_spline,
            )
        else:
            centerline_spline = None
            reference_path = ReferencePath(waypoints_norm, s=0.0, k=1)

        corridor_bounds = None
        if constraint_config is not None:
            corridor_bounds = convert_constraints_to_corridor_bounds(
                constraint_config,
                reference_path,
                default_margin=self.planner_margin,
            )

        clearance_profile = None
        task_width_profile = None
        curvature_rate_profile = None
        curvature_profile = None
        if reference_path.total_length > 0:
            n_profile = 500
            s_profile = np.linspace(0, reference_path.total_length, n_profile)
            # TRUE task width W_task(s): unclamped distance between active
            # constraint bounds, np.inf where no constraint covers s. This is
            # the perceptual difficulty signal (gaze budget, free-space mask,
            # speed model, deviation scale). The planner's corridor_bounds
            # above stay clamped (max_bound) purely for QP conditioning and
            # must not leak into behaviour — a 10 m corridor is free space,
            # not a 0.2 m tunnel.
            task_bounds = None
            if constraint_config is not None:
                task_bounds = convert_constraints_to_corridor_bounds(
                    constraint_config, reference_path,
                    default_margin=self.planner_margin, max_bound=None)
            w_task_profile = compute_clearance_profile(
                reference_path, s_profile,
                corridor_bounds=task_bounds,
                cartesian_constraints=cartesian_regions if cartesian_regions else None,
                unconstrained="inf",
            )
            task_width_profile = (s_profile, w_task_profile)
            # Finite copy for log-feature / interpolation consumers.
            c_profile = np.minimum(w_task_profile, W_TASK_FINITE_CAP)
            clearance_profile = (s_profile, c_profile)

            # Curvature-rate signal computed on the CENTERLINE (not optimized
            # path) so corner difficulty survives corner-cutting smoothing.
            centerline_path = centerline_spline if centerline_spline is not None else ReferencePath(waypoints_norm, s=0.0, k=3)
            n_cl = 500
            s_cl = np.linspace(0, centerline_path.total_length, n_cl)
            rate_cl = compute_curvature_rate_profile(centerline_path, s_cl)

            # Map centerline signal to optimized path via normalized progress
            progress_cl = s_cl / centerline_path.total_length
            progress_opt = s_profile / reference_path.total_length
            rate_profile = np.interp(progress_opt, progress_cl, rate_cl)
            curvature_rate_profile = (s_profile, rate_profile)

            kappa_profile = np.array([abs(reference_path.curvature(float(s))) for s in s_profile])
            curvature_profile = (s_profile, kappa_profile)

        cursor_pos = np.array([waypoints_norm[0][0], waypoints_norm[0][1]], dtype=float)
        cursor_vel = np.array([0.0, 0.0], dtype=float)
        cursor_acc = np.array([0.0, 0.0], dtype=float)
        hand_pos = np.array([0.0, 0.0], dtype=float)

        reset_warm_start()
        trajectory = []
        current_time = 0.0
        final_target = np.array(waypoints_norm[-1])

        # --- planning-horizon / replan-scheduling setup ---
        desired_speed = float(self.planner_weights.get('desired_speed', 0.12))
        budget_horizon = None
        if self.horizon_mode == 'budget':
            if clearance_profile is None:
                raise ValueError("horizon_mode='budget' requires a reference path with profiles")
            s_prof, c_prof = clearance_profile
            _, w_task_prof = task_width_profile
            _, k_prof = curvature_profile
            _, r_prof = curvature_rate_profile
            # Reference speed along the path — same (finite) signals model()
            # feeds the speed model — used to convert the arc-length
            # lookahead to steps.
            if self.speed_model is not None:
                v_ref_prof = self.speed_model.compute_speed_profile(
                    s_prof, c_prof, k_prof, r_prof)
            else:
                # Anchor-drive: the horizon is a fixed DURATION (the plan
                # deadline), so no reference speed is needed to convert the
                # lookahead to steps. Placeholder keeps the budget class API.
                v_ref_prof = np.full_like(s_prof, desired_speed)
            # The budget consumes the TRUE width: density is exactly zero
            # where W_task is inf, so in free space the anchor runs to the
            # goal (path end) — gaze goes straight to the target, no fitted
            # or hidden constant involved.
            budget_horizon = DifficultyBudgetHorizon(
                s_prof, w_task_prof, v_ref_prof,
                D0=self.budget_D0, T_min=self.budget_T_min,
                gamma=self.budget_gamma, W_ref=self.budget_W_ref,
                kappa_profile=k_prof, curvature_weighted=self.budget_curvature_weighted,
            )
        tau_steps = max(0, int(round(self.replan_latency_s / self.interval)))
        # Time-based solve-horizon floor: the plan must always cover at least
        # one visuomotor delay of movement, so the horizon cannot collapse to
        # horizon_min_steps when the anchor caps at the path end (the
        # short-horizon + precision-braking stall behind budget-mode
        # timeouts).
        floor_steps = int(np.ceil(self.budget_T_min / self.interval)) \
            if self.budget_T_min > 0 else 0
        scheduler = ReplanScheduler(
            mode=self.replan_mode, latency_steps=tau_steps,
            latency_cv=self.replan_latency_cv,
            latency_max_steps=max(0, int(round(self.replan_latency_max_s / self.interval))),
            deviation_frac=self.replan_deviation_frac,
            min_open_loop_steps=max(0, int(round(self.min_open_loop_s / self.interval))),
            rng=self._replan_rng,
        )
        plan = None            # planned velocity/displacement sequences of the current solve
        last_anchor_s = None   # gaze memory: the previous plan's anchor (arc length)
        theta_track = 0.0      # warm guess for arc-length projection
        motor_steps = max(1, int(round(self.motor_period_s / self.interval))) if self.motor_period_s > 0 else 0
        fix = None             # current fixation: dict(t0, s0, pace)

        # Termination: DWELL_S of consecutive samples inside the target
        # (stands in for the human click latency; the pointing data show
        # ~0.3 s median from final target entry to click).
        dwell_required = int(round(self.dwell_s / self.interval))
        dwell_steps = 0
        aborted_breach = False
        for step in range(max_steps):
            dist_to_target = np.linalg.norm(cursor_pos - final_target)
            if dist_to_target < target_radius:
                dwell_steps += 1
                if dwell_steps >= dwell_required:
                    break
            else:
                dwell_steps = 0

            theta_now = None
            if scheduler.wants_theta:
                theta_now = float(reference_path.find_closest_theta(
                    cursor_pos, initial_guess=theta_track))
                theta_track = theta_now
            # Early-replan interrupt: realised drift from the open-loop plan,
            # scaled by the usable width at the last solve. Zero until at
            # least one step of the plan has executed (the plan starts at the
            # realised cursor state, and noiseless execution tracks exactly).
            deviation_ratio = None
            if (scheduler.mode == 'intermittent'
                    and self.replan_deviation_frac > 0.0
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
            trigger = scheduler.needs_replan(theta_for_trigger, deviation_ratio=deviation_ratio)
            if (trigger is None and motor_steps > 0 and fix is not None
                    and fix.get('steps', 0) >= fix.get('max_steps') if fix and fix.get('max_steps') else False):
                trigger = 'exhausted'      # fixation-level backstop (motor replans reset plan_idx)
            motor_replan = (trigger is None and motor_steps > 0 and self.anchor_drive
                            and fix is not None and scheduler.plan_idx > motor_steps)

            if trigger is not None or motor_replan:
                theta0 = float(reference_path.find_closest_theta(
                    cursor_pos, initial_guess=theta_track))
                theta_track = theta0
                anchor_solve_s = None
                if motor_replan:
                    # Slide the anchor along the fixation's schedule: demand the
                    # schedule position one deadline ahead of NOW (no new fixation).
                    t_ahead = (current_time - fix['t0']) + self.plan_deadline_s
                    anchor_solve_s = None
                    if self.plan_turn_time_s > 0.0:
                        # Turning-time mode: the motor plan spans the numerical
                        # horizon floor; demand the schedule at its end.
                        n_base_m = max(3, motor_steps + 2)
                        t_ahead = (current_time - fix['t0']) + n_base_m * self.interval
                    # Slid anchor keeps the same lead floor as fixations: at
                    # least one local room ahead of the cursor — otherwise a
                    # cursor that overtakes the schedule loses all drive and
                    # waits for the schedule to catch up (mid-trial stall).
                    room_m = 0.0
                    if clearance_profile is not None:
                        s_cl_m, c_cl_m = clearance_profile
                        w_here_m = float(np.interp(theta0, s_cl_m, c_cl_m))
                        room_m = 0.5 * w_here_m if w_here_m < FREE_SPACE_CLEARANCE_M else float(target_radius)
                    anchor_s = min(max(fix['s0'] + fix['pace'] * t_ahead, theta0 + room_m),
                                   float(reference_path.total_length))
                    n_base = max(1, int(np.floor(self.plan_deadline_s / self.interval + 0.5 + 1e-9)))
                    if self.plan_turn_time_s > 0.0:
                        n_base = max(3, motor_steps + 2)
                if motor_replan:
                    pass
                elif budget_horizon is not None:
                    s_from = theta0
                    if self.anchor_drive and self.anchor_memory and last_anchor_s is not None:
                        s_from = max(s_from, last_anchor_s)
                        if self.corner_consume and curvature_profile is not None:
                            s_k, k_k = curvature_profile
                            if float(np.interp(last_anchor_s, s_k, k_k)) > self.corner_kappa:
                                beyond = s_k[(s_k > last_anchor_s) & (k_k < self.corner_kappa)]
                                if len(beyond):
                                    s_from = max(s_from, float(beyond[0]))
                    s_from = min(s_from, float(reference_path.total_length))
                    anchor_s = budget_horizon.anchor(
                        s_from, v_now=float(np.hypot(cursor_vel[0], cursor_vel[1])))
                    if self.anchor_drive and self.anchor_memory:
                        anchor_s = max(float(anchor_s), s_from)
                    if self.anchor_drive and self.anchor_lead_floor and clearance_profile is not None:
                        # Lead floor BEFORE the deadline: if the floor extends
                        # the anchor, the deadline must stretch with it
                        # (review finding: floor applied after n_base made the
                        # via-point demand the extended lead in the old time).
                        s_cl0, c_cl0 = clearance_profile
                        w_here0 = float(np.interp(theta0, s_cl0, c_cl0))
                        room0 = 0.5 * w_here0 if w_here0 < FREE_SPACE_CLEARANCE_M else float(target_radius)
                        anchor_s = min(max(float(anchor_s), theta0 + room0), float(reference_path.total_length))
                    if self.anchor_drive:
                        # Fixed plan duration: speed = lookahead / deadline.
                        # Round half-up with a float guard (0.175/0.05 = 3.4999.. -> 4).
                        lead_now = max(0.0, float(anchor_s) - theta0)
                        t_plan = max(self.plan_deadline_s, lead_now / max(self.plan_vmax, 1e-6))
                        # Turning-time mode: t_plan = max(T0, lead/v_max) + tau*theta*(W_ref/W)^beta,
                        # T0 = plan_deadline_s (gaze: straight-segment time-to-anchor, ~0.1-0.2 s).
                        # The numerical horizon floor (3 nodes) is separate, see below.
                        if self.plan_turn_time_s > 0.0 and curvature_profile is not None:
                            # Turning-time deadline (config-gated): the planned
                            # time-to-anchor grows with the turning angle inside
                            # the lead, tau_turn seconds per radian. Gaze data:
                            # the lead is width-only and crosses corners, while
                            # the time to reach the anchor lengthens in bends —
                            # one slower movement, not a shorter look-ahead.
                            s_kp, k_kp = curvature_profile
                            s_lo, s_hi = theta0, float(anchor_s)
                            if s_hi > s_lo + 1e-9:
                                s_grid = np.linspace(s_lo, s_hi, max(4, int((s_hi - s_lo) / 0.001) + 2))
                                theta_lead = float(np.trapz(np.interp(s_grid, s_kp, k_kp), s_grid))
                                w_fac = 1.0
                                if self.plan_turn_width_exp > 0.0 and clearance_profile is not None:
                                    # Tolerance-scaled turning time: seconds per
                                    # radian grow as the room shrinks, (W_ref/W)^beta
                                    # (gaze data: B's corner crossing 0.46 s at 20 mm
                                    # vs 0.26 s at 40-50 mm; beta = 1 fits best).
                                    s_cw, c_cw = clearance_profile
                                    w_loc = float(np.interp(theta0, s_cw, c_cw))
                                    if 0.0 < w_loc < FREE_SPACE_CLEARANCE_M:
                                        w_fac = (self.budget_W_ref / w_loc) ** self.plan_turn_width_exp
                                t_plan += self.plan_turn_time_s * theta_lead * w_fac
                        a_max = float(self.planner_weights.get('acc_max', 0.0) or 0.0)
                        if a_max > 0.0:
                            # ... nor faster than the bounded hand can cover the
                            # lead from its current speed: 1/2 a t^2 + v t = h.
                            v_now = float(np.hypot(cursor_vel[0], cursor_vel[1]))
                            t_acc = (-v_now + np.sqrt(v_now * v_now + 2.0 * a_max * lead_now)) / a_max
                            t_plan = max(t_plan, float(t_acc))
                        n_base = max(1, int(np.floor(t_plan / self.interval + 0.5 + 1e-9)))
                        anchor_solve_s = None
                        if self.plan_turn_time_s > 0.0:
                            # Numerical floor: >= 3 nodes (shorter horizons destabilise
                            # the solve) and > the motor period, else the scheduler's
                            # plan exhaustion fires at the next motor tick before arrival.
                            n_min = max(3, motor_steps + 2)
                            if n_base < n_min:
                                # Horizon floor: demand the schedule position at the
                                # horizon end (pace unchanged), not the fixation
                                # point itself — same rule as the motor replans.
                                n_base = n_min
                                pace0 = lead_now / max(t_plan, 1e-6)
                                anchor_solve_s = min(theta0 + pace0 * n_base * self.interval,
                                                     float(reference_path.total_length))
                        fix = {'t0': current_time, 's0': theta0,
                               'pace': lead_now / max(t_plan, 1e-6),
                               'steps': 0, 'max_steps': None}
                        if os.environ.get('HCS_DEBUG_PLAN'):
                            print(f"[plan] t={current_time:.2f} s0={theta0*1000:.0f}mm lead={lead_now*1000:.1f}mm "
                                  f"t_plan={t_plan:.3f}s pace={fix['pace']:.3f} n_base={n_base} solve_anchor={'%.0f' % (anchor_solve_s*1000) if anchor_solve_s is not None else '-'}", flush=True)
                    else:
                        n_base = int(np.ceil(
                            budget_horizon.traverse_time(theta0, anchor_s) / self.interval))
                        # Anchor and floor both cap at the path end, so the
                        # traverse time (and with it n_base) shrinks toward zero
                        # there; the plan itself must still cover >= T_min.
                        n_base = max(n_base, floor_steps)
                else:
                    n_base = self.pred_horizon
                    anchor_s = None  # fixed mode: set from the solved plan below
                if anchor_s is not None and not motor_replan:
                    anchor_s = min(float(anchor_s), float(reference_path.total_length))
                if self.replan_mode == 'intermittent':
                    # The plan must survive the post-arrival latency (the old
                    # plan keeps executing past the anchor), plus the same
                    # margin again as slack for arrival running later than the
                    # reference-speed estimate (start-up, corners, noise).
                    # This is plan availability, not behaviour: the replan time
                    # is still set by arrival + tau; exhaustion only backstops.
                    # BUMP mode: plans refresh every motor period and are never
                    # executed past it — padding beyond the deadline is dead
                    # weight (3x solve cost for nodes that never run).
                    n_solve = n_base if motor_steps > 0 else n_base + 2 * tau_steps
                else:
                    n_solve = n_base
                if self.horizon_mode == 'budget' or self.replan_mode == 'intermittent':
                    n_solve = int(np.clip(
                        n_solve, self.horizon_min_steps, self.horizon_max_steps))

                tunnel_path = waypoints_norm
                model_input = SteeringModelInput(
                    state_cog=(
                        float(cursor_pos[0]),
                        float(cursor_pos[1]),
                        float(cursor_vel[0]),
                        float(cursor_vel[1])
                    ),
                    bump=BumpParams(
                        pred_horizon=n_solve,
                        Tp=self.tp,
                        nc=self.nc
                    ),
                    env=EnvParams(interval=self.interval),
                    tunnel=TunnelInfo(
                        tunnel_path=tunnel_path,
                        tunnel_width=tunnel_width or 0.1,
                        top_wall=None,
                        bottom_wall=None
                    ),
                    planner_weights=self.planner_weights,
                    planner_margin=self.planner_margin,
                    reference_path=reference_path,
                    current_acc=(0.0, 0.0),
                    corridor_bounds=corridor_bounds,
                    cartesian_constraints=cartesian_regions if cartesian_regions else None,
                    clearance_profile=clearance_profile,
                    curvature_rate_profile=curvature_rate_profile,
                    curvature_profile=curvature_profile,
                    speed_model=self.speed_model,
                    target_radius=target_radius,  # added for pointing model
                )
                model_input.current_acc = (float(cursor_acc[0]), float(cursor_acc[1]))
                # Warm-start shift = steps executed since the previous solve.
                model_input.warm_shift = max(1, scheduler.plan_idx - 1)
                # Anchor-drive: the gaze anchor is the plan's via point at
                # the deadline node (n_base steps); the padded tail coasts.
                model_input.anchor_s = (anchor_solve_s if (self.anchor_drive and anchor_solve_s is not None)
                                        else (anchor_s if self.anchor_drive else None))
                model_input.deadline_steps = n_base
                model_input.anchor_pace = (float(fix['pace']) if (self.anchor_tail_pace and fix is not None) else 0.0)
                model_input.safety_steps = tau_steps if (self.anchor_drive and self.coast_safety) else 0

                cursor_info, plan_debug = model(model_input)
                c_pos_dx, c_pos_dy, c_vel_x, c_vel_y = cursor_info
                # Absolute planned positions after each step, and the
                # deviation trigger's scale: the local usable width in a
                # corridor. In unconstrained space the clearance saturates
                # far above any tunnel width and a width-scaled trigger could
                # never fire — there the scale is the plan's own span (a plan
                # that has drifted by deviation_frac of the distance it set
                # out to cover is invalid), floored at the target diameter so
                # terminal micro-plans keep a sane threshold.
                # Deviation-trigger scale: the local usable width where the
                # task is constrained; in free space (no walls) the task's
                # own accuracy scale, the target diameter.
                w_solve = None
                if clearance_profile is not None:
                    s_cl, c_cl = clearance_profile
                    w_here = float(np.interp(theta0, s_cl, c_cl))
                    w_solve = (w_here if w_here < FREE_SPACE_CLEARANCE_M
                               else 2.0 * target_radius)
                dev_scale = w_solve if (w_solve and w_solve > 0) else None
                if dev_scale is not None and dev_scale >= FREE_SPACE_CLEARANCE_M:
                    span = float(np.hypot(np.sum(c_pos_dx), np.sum(c_pos_dy)))
                    dev_scale = max(span, 2.0 * target_radius)
                _carry_axy = plan.get('anchor_xy') if (motor_replan and plan) else None
                _carry_tol = plan.get('arrival_tol', 0.0) if (motor_replan and plan) else 0.0
                plan = {
                    'anchor_xy': _carry_axy, 'arrival_tol': _carry_tol,
                    'c_pos_dx': c_pos_dx, 'c_pos_dy': c_pos_dy,
                    'c_vel_x': c_vel_x, 'c_vel_y': c_vel_y,
                    'n_steps': len(c_pos_dx),
                    'pos_x': cursor_pos[0] + np.cumsum(c_pos_dx),
                    'pos_y': cursor_pos[1] + np.cumsum(c_pos_dy),
                    'dev_scale': dev_scale,
                }
                if anchor_s is None:
                    # Fixed-horizon anchor: the plan's own progress at the
                    # attentional horizon Th (n_base steps) — where the solved
                    # plan expects the cursor to be, so arrival is achievable
                    # by construction and noise supplies the variability.
                    k_anchor = min(n_base, plan['n_steps'])
                    planned_end = (
                        cursor_pos[0] + float(np.sum(c_pos_dx[:k_anchor])),
                        cursor_pos[1] + float(np.sum(c_pos_dy[:k_anchor])),
                    )
                    anchor_s = float(reference_path.find_closest_theta(
                        np.array(planned_end), initial_guess=theta0))
                    anchor_s = min(anchor_s, float(reference_path.total_length))
                if motor_replan and fix is not None and fix.get('anchor_xy') is not None:
                    # Motor replans slide the PLANNING anchor, but arrival is
                    # judged against the FIXATION point the eyes rest on —
                    # keep the fixation's arrival geometry (else the slid
                    # anchor is always ahead and arrival can never fire).
                    plan['anchor_xy'] = fix['anchor_xy']
                    plan['anchor_s'] = fix['anchor_s']
                    plan['arrival_tol'] = fix['arrival_tol']
                elif anchor_s is not None:
                    a_xy = reference_path(float(anchor_s))
                    plan['anchor_xy'] = (float(a_xy[0]), float(a_xy[1]))
                    plan['anchor_s'] = float(anchor_s)
                    # room at the anchor: half the local usable width, or the target radius in free space
                    tol = 0.5 * dev_scale if (dev_scale and dev_scale < FREE_SPACE_CLEARANCE_M) else float(target_radius)
                    plan['arrival_tol'] = max(float(tol), 1e-3)
                    if fix is not None:
                        fix['anchor_xy'] = plan['anchor_xy']; fix['anchor_s'] = plan['anchor_s']; fix['arrival_tol'] = plan['arrival_tol']
                _ev = ReplanEvent(step=step, t=current_time, theta=theta0,
                                  anchor=anchor_s, n_steps=n_solve, trigger=trigger)
                _ev.arrival_tol = plan.get('arrival_tol')
                last_anchor_s = float(anchor_s) if anchor_s is not None else None
                if motor_replan:
                    scheduler.plan_len = plan['n_steps']; scheduler.plan_idx = 1
                else:
                    scheduler.on_replan(_ev, anchor=anchor_s, plan_len=plan['n_steps'])
                    if fix is not None:
                        # Fixation lifetime backstop: deadline + 2*latency,
                        # independent of the (unpadded) motor-solve length.
                        fix['max_steps'] = n_base + 2 * tau_steps

            # Execute the next step of the current plan. j indexes the planned
            # velocity at the END of the step (c_vel_* start with the solve-time
            # velocity at index 0); the matching displacement is c_pos_d*[j-1].
            j = min(scheduler.plan_idx, plan['n_steps'])
            c_vel_x = plan['c_vel_x']
            c_vel_y = plan['c_vel_y']

            if self.add_noise:
                # The step integrates trapezoidally from the cursor's realised
                # start velocity to the (noisy) planned end velocity — in the
                # zero-noise limit this advances by the planner's displacement,
                # matching the add_noise=False branch. Between replans the plan
                # keeps executing open-loop: motor noise perturbs each realised
                # step but is unobservable to the planner until the next solve.
                c_pos_dx_step, c_pos_dy_step, c_vel_x_step, c_vel_y_step, \
                hand_pos[0], hand_pos[1], _, _ = single_step_motor_and_device_noise(
                    c_vel_x[j], c_vel_y[j],
                    hand_pos[0], hand_pos[1],
                    self.nc,
                    self.interval,
                    self.forearm,
                    c_vel_x_prev=float(cursor_vel[0]), c_vel_y_prev=float(cursor_vel[1]),
                )
            else:
                c_pos_dx_step = plan['c_pos_dx'][j - 1]
                c_pos_dy_step = plan['c_pos_dy'][j - 1]
                c_vel_x_step = c_vel_x[j]
                c_vel_y_step = c_vel_y[j]

            cursor_pos[0] += c_pos_dx_step
            cursor_pos[1] += c_pos_dy_step
            if self.carry_acceleration:
                # Planned (pre-noise) acceleration over the executed step —
                # the motor/device noise is unobservable to the planner.
                cursor_acc[0] = (c_vel_x[j] - c_vel_x[j - 1]) / self.interval
                cursor_acc[1] = (c_vel_y[j] - c_vel_y[j - 1]) / self.interval
            cursor_vel[0] = c_vel_x_step
            cursor_vel[1] = c_vel_y_step
            scheduler.on_step_executed()
            if fix is not None:
                fix['steps'] = fix.get('steps', 0) + 1

            screen_x = cursor_pos[0] / screen_width_m * screen_width
            screen_y = cursor_pos[1] / screen_height_m * screen_height

            if return_timestamps:
                trajectory.append((screen_x, screen_y, current_time))
            else:
                trajectory.append((screen_x, screen_y, self.interval))

            current_time += self.interval

            if self.abort_on_breach_m is not None and clearance_profile is not None:
                # Trial failure: the cursor left the tunnel by more than the
                # margin (the experiment restarts such trials). Stops the
                # runaway sims that otherwise burn the 30 s cap.
                th_b = float(reference_path.find_closest_theta(cursor_pos, initial_guess=theta_track))
                s_cb, c_cb = clearance_profile
                w_b = float(np.interp(th_b, s_cb, c_cb))
                if 0.0 < w_b < FREE_SPACE_CLEARANCE_M:
                    off_b = float(np.linalg.norm(cursor_pos - np.asarray(reference_path(th_b), dtype=float).reshape(-1)[:2]))
                    if off_b > 0.5 * w_b + float(self.abort_on_breach_m):
                        aborted_breach = True
                        break

        self.last_diagnostics = {
            'horizon_mode': self.horizon_mode,
            'replan_mode': self.replan_mode,
            'anchor_drive': self.anchor_drive,
            'plan_deadline_s': self.plan_deadline_s,
            'coast_safety': self.coast_safety,
            'plan_vmax': self.plan_vmax, 'plan_turn_time_s': self.plan_turn_time_s, 'plan_turn_width_exp': self.plan_turn_width_exp, 'arrival_mode': self.arrival_mode, 'anchor_lead_floor': self.anchor_lead_floor,
            'anchor_memory': self.anchor_memory, 'corner_consume': self.corner_consume,
            'budget_T_min': self.budget_T_min,
            'replan_latency_cv': self.replan_latency_cv,
            'replan_deviation_frac': self.replan_deviation_frac,
            'min_open_loop_s': self.min_open_loop_s,
            'interval': self.interval,
            'n_steps_executed': len(trajectory),
            'aborted_breach': aborted_breach,
            'n_solves': len(scheduler.events),
            'total_length': float(reference_path.total_length),
            'replan_events': [
                {'step': e.step, 't': e.t, 'theta': e.theta, 'anchor': e.anchor,
                 'n_steps': e.n_steps, 'trigger': e.trigger,
                 'arrival_tol': getattr(e, 'arrival_tol', None)}
                for e in scheduler.events
            ],
        }

        if return_reference_path:
            return trajectory, reference_path
        return trajectory

    def generate_trajectory_with_start_and_end(
        self,
        task_file: Optional[Union[str, Path]] = None,
        waypoints: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[Union[Dict[str, Any], str, Path]] = None,
        screen_width: float = 1920.0,
        screen_height: float = 1080.0,
        max_steps: int = 2000,
        target_radius: float = 0.01,
        use_optimal_path: bool = True,  # Ignored internally, but kept for signature parity
        return_timestamps: bool = False,
        return_reference_path: bool = False
    ) -> Union[List[Tuple[float, float, float]], Tuple[List[Tuple[float, float, float]], Any]]:
        """
        Generate a point-to-point pointing trajectory (Fitts' Law baseline).

        ⚠️ RULED OUT for pointing tasks (2026-08): shows jittery start/end
        behavior and a severe distance-driven completion-time blowup vs.
        human data. Replaced by MPCC-in-a-bypass-tunnel — use
        generate_trajectory_with_waypoints with an
        experiment.environment._create_pointing_bypass_env-built task
        instead (see eval/eval-new-data/run_eval.py::build_fitts_bypass_config).
        Its pygame visual debugger was removed; do not use this method as
        the default pointing-task model in new code.

        Signature and unit conventions mirror generate_trajectory_with_waypoints
        exactly (same task_file/waypoints/screen_width/screen_height loading,
        same target_radius units — normalized meters, NOT pixels). Only the
        FIRST and LAST waypoints are used as the pointing start/end; any
        intermediate waypoints are ignored, since this baseline flies straight
        to the target rather than following a multi-waypoint tunnel.

        This baseline is intentionally constraint-free (see baseline_model.py):
        `constraints` is accepted only for signature parity and is not applied,
        and `use_optimal_path` is ignored (there is no tunnel corridor to
        optimize against for a two-point straight-line path).

        Args:
            task_file: Path to task.json file containing "waypoints" and
                optionally "screen_width"/"screen_height". If provided, these
                values override the corresponding parameters below.
            waypoints: Optional list of (x, y) waypoints in screen pixels.
                Ignored if task_file is provided. Only waypoints[0] and
                waypoints[-1] are used (start and end of the point-to-point move).
            constraints: Accepted for signature parity with
                generate_trajectory_with_waypoints; NOT applied — this baseline
                has no boundary/corridor constraints.
            screen_width: Screen width in pixels (default: 1920). Overridden by
                task_file if present.
            screen_height: Screen height in pixels (default: 1080). Overridden
                by task_file if present.
            max_steps: Maximum simulation steps (default: 2000).
            target_radius: Target radius, in normalized meters (default: 0.01)
                — same units as generate_trajectory_with_waypoints, NOT pixels.
            use_optimal_path: Accepted for signature parity; ignored (no tunnel
                corridor exists to optimize for a straight point-to-point path).
            return_timestamps: If True, return timestamps instead of delays
                (default: False).
            return_reference_path: If True, also return the reference path
                object (default: False).
    
        Returns:
            If return_reference_path is False: List of tuples (x, y, delay) or
                (x, y, timestamp) in screen pixels.
            If return_reference_path is True: Tuple of (trajectory, reference_path).
        """
        # --- Task/waypoint loading: identical to generate_trajectory_with_waypoints ---
        if task_file is not None:
            task_path = Path(task_file)
            if not task_path.exists():
                raise FileNotFoundError(f"Task file not found: {task_file}")
    
            with open(task_path, 'r') as f:
                task_data = json.load(f)
    
            if "waypoints" not in task_data:
                raise ValueError("task.json must contain 'waypoints' key")
    
            waypoints = [tuple(wp) for wp in task_data["waypoints"]]
    
            if "constraints" in task_data:
                constraints = task_data["constraints"]
    
            # task_file overrides screen_width/screen_height parameters
            if "screen_width" in task_data:
                screen_width = float(task_data["screen_width"])
            if "screen_height" in task_data:
                screen_height = float(task_data["screen_height"])
    
        if waypoints is None or len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required (either from task_file or waypoints parameter)")
    
        # This baseline is point-to-point: only the endpoints matter. Any
        # intermediate waypoints are accepted (for signature/task_file parity)
        # but ignored.
        start_point = waypoints[0]
        end_point = waypoints[-1]
    
        # `constraints` and `use_optimal_path` are intentionally unused from here
        # on — this baseline is constraint-free by design (see baseline_model.py).
    
        # Screen pixels → normalized meters (identical conversion to
        # generate_trajectory_with_waypoints).
        screen_width_m = 0.46
        screen_height_m = screen_height / screen_width * screen_width_m
    
        start_norm = (
            start_point[0] / screen_width * screen_width_m,
            start_point[1] / screen_height * screen_height_m,
        )
        end_norm = (
            end_point[0] / screen_width * screen_width_m,
            end_point[1] / screen_height * screen_height_m,
        )
    
        # Straight-line reference path (k=1: linear spline through the two points).
        ref_path = ReferencePath([start_norm, end_norm], s=0.0, k=1)
    
        # Massive static "tunnel" so any internal boundary checks pass through
        # without rejecting or scaling anything.
        tunnel_path = [start_norm, end_norm]
        tunnel_width = 10.0
    
        path_length = float(np.hypot(end_norm[0] - start_norm[0], end_norm[1] - start_norm[1]))
        desired_speed = self.planner_weights.get('desired_speed', 0.2)
        planning_T = max(0.5, path_length / desired_speed)
    
        t_nodes = np.arange(1, self.pred_horizon + 1) * self.interval
        tau = t_nodes / planning_T
        # Minimum-jerk bell-shaped velocity profile: v(t) = (L/T)(30τ²-60τ³+30τ⁴)
        speed_profile = np.where(
            tau < 1.0,
            (path_length / planning_T) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4),
            0.0,
        )
    
        reset_warm_start()
    
        cursor_pos = np.array([start_norm[0], start_norm[1]], dtype=float)
        cursor_vel = np.array([0.0, 0.0], dtype=float)
        cursor_acc = np.array([0.0, 0.0], dtype=float)
        hand_pos = np.array([0.0, 0.0], dtype=float)
        s = 0.0
    
        planner_weights = dict(self.planner_weights)
        planner_weights.setdefault('acceleration', 1e-4)
    
        limits = {'acc_max': 100.0}
    
        trajectory = []
        current_time = 0.0
        final_target = np.array(end_norm)
    
        dwell_required = int(round(0.25 / self.interval))
        dwell_steps = 0
    
        for step in range(max_steps):
            dist_to_target = np.linalg.norm(cursor_pos - final_target)
            # target_radius is in normalized meters here (NOT pixels) — matches
            # generate_trajectory_with_waypoints's units exactly.
            if dist_to_target < target_radius:
                dwell_steps += 1
                if dwell_steps >= dwell_required:
                    break
            else:
                dwell_steps = 0
    
            state_0 = [
                float(cursor_pos[0]), float(cursor_pos[1]),
                float(cursor_vel[0]), float(cursor_vel[1]),
                float(cursor_acc[0]), float(cursor_acc[1]),
                s,
            ]
    
            model_input = SteeringModelInput(
                state_cog=(
                    float(cursor_pos[0]), float(cursor_pos[1]),
                    float(cursor_vel[0]), float(cursor_vel[1]),
                ),
                bump=BumpParams(
                    pred_horizon=self.pred_horizon,
                    Tp=self.tp,
                    nc=self.nc
                ),
                env=EnvParams(interval=self.interval),
                tunnel=TunnelInfo(
                    tunnel_path=tunnel_path,
                    tunnel_width=tunnel_width,
                    top_wall=None,
                    bottom_wall=None
                ),
                planner_weights=planner_weights,
                planner_margin=self.planner_margin,
                reference_path=ref_path,
                current_acc=(float(cursor_acc[0]), float(cursor_acc[1])),
                corridor_bounds=None,
                cartesian_constraints=None,
                clearance_profile=None,
                curvature_rate_profile=None,
                curvature_profile=None,
                speed_model=self.speed_model,
                target_radius=target_radius,
            )
    
            controls, opt_info = generate_baseline_mpc(
                ref_path=ref_path,
                state_0=state_0,
                num_steps=self.pred_horizon,
                dt=self.interval,
                weights=model_input.planner_weights,
                limits=limits,
                speed_profile=speed_profile,
                desired_speed=desired_speed,
            )
    
            jx = controls[:, 0]
            jy = controls[:, 1]
            vs = controls[:, 2]
    
            A_vel = _build_A_vel_from_jerk(self.pred_horizon, self.interval)
            A_pos = _build_A_pos_from_jerk(self.pred_horizon, self.interval)
    
            t_vec = np.arange(1, self.pred_horizon + 1) * self.interval
    
            vx_free = cursor_vel[0] + cursor_acc[0] * t_vec
            vy_free = cursor_vel[1] + cursor_acc[1] * t_vec
            px_free = cursor_pos[0] + cursor_vel[0] * t_vec + 0.5 * cursor_acc[0] * (t_vec**2)
            py_free = cursor_pos[1] + cursor_vel[1] * t_vec + 0.5 * cursor_acc[1] * (t_vec**2)
    
            pos_x = px_free + A_pos @ jx
            pos_y = py_free + A_pos @ jy
            vel_x = vx_free + A_vel @ jx
            vel_y = vy_free + A_vel @ jy
    
            c_vel_x = np.insert(vel_x, 0, cursor_vel[0])
            c_vel_y = np.insert(vel_y, 0, cursor_vel[1])
    
            all_pos_x = np.insert(pos_x, 0, cursor_pos[0])
            all_pos_y = np.insert(pos_y, 0, cursor_pos[1])
            c_pos_dx = np.diff(all_pos_x)
            c_pos_dy = np.diff(all_pos_y)
    
            # c_vel_x has length pred_horizon+1: [v0, v1, ..., vN]; index 1 = first planned step
            planned_vel_idx = min(1, len(c_vel_x) - 1)
    
            if self.add_noise:
                c_pos_dx_step, c_pos_dy_step, c_vel_x_step, c_vel_y_step, \
                hand_pos[0], hand_pos[1], _, _ = single_step_motor_and_device_noise(
                    c_vel_x[planned_vel_idx], c_vel_y[planned_vel_idx],
                    hand_pos[0], hand_pos[1],
                    self.nc,
                    self.interval,
                    self.forearm,
                    c_vel_x_prev=c_vel_x[0], c_vel_y_prev=c_vel_y[0],
                )
            else:
                c_pos_dx_step = c_pos_dx[0]
                c_pos_dy_step = c_pos_dy[0]
                c_vel_x_step = c_vel_x[planned_vel_idx]
                c_vel_y_step = c_vel_y[planned_vel_idx]
    
            cursor_pos[0] += c_pos_dx_step
            cursor_pos[1] += c_pos_dy_step
            cursor_vel[0] = c_vel_x_step
            cursor_vel[1] = c_vel_y_step
            # cursor_acc is intentionally left at (0.0, 0.0): each replanning step
            # starts from zero acceleration, matching the convention already used
            # by generate_trajectory_with_waypoints (which hardcodes
            # current_acc=(0.0, 0.0) every step rather than carrying it forward).
            s += float(vs[0]) * self.interval
    
            screen_x = cursor_pos[0] / screen_width_m * screen_width
            screen_y = cursor_pos[1] / screen_height_m * screen_height
    
            if return_timestamps:
                trajectory.append((screen_x, screen_y, current_time))
            else:
                trajectory.append((screen_x, screen_y, self.interval))
    
            current_time += self.interval
    
        if return_reference_path:
            return trajectory, ref_path
        return trajectory