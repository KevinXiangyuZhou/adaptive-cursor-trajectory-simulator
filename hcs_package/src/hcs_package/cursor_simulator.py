"""Playwright-compatible cursor simulator for generating human-like trajectories."""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from .model import model
from .mpcc_model import reset_warm_start
from .params import SteeringModelInput, BumpParams, EnvParams, TunnelInfo
from .reference_path import ReferencePath, generate_optimal_reference_path
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
                "goal_precision": 0.0001  # fallback; overridden per-user via user_configurations.
                                          # 2026-08: with the free-space LQR objective, the well at
                                          # ~1e-4 gives the human endpoint-depth trend and target
                                          # re-entry rate; ~3e-4+ over-slows R=5 mm targets.
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
            # budget  ∫(1/W + lam|kappa|)ds = D0  (eval/eval-gaze-cursor
            # lookahead_difficulty.py; D0/lam are unit-invariant), floored at
            # the visuomotor-delay lookahead v * T_min (refit_floor.py) —
            # which also floors the solve horizon in time so it cannot
            # collapse near the path end. Defaults are the pooled refit
            # (cross-validated, magnitude-calibrated) sim_params from
            # results/lookahead_floor_summary.json.
            "horizon_mode": "fixed",
            "budget": {"D0": 1.66, "lam": 0.5, "T_min": 0.1},
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
            "replan_deviation_frac": 0.15,
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
        self.budget_lam = float(budget_cfg.get('lam', 0.5))
        self.budget_T_min = float(budget_cfg.get('T_min', 0.1))
        self.replan_mode = str(config.get('replan_mode', 'every_step'))
        if self.replan_mode not in ('every_step', 'intermittent'):
            raise ValueError(f"replan_mode must be 'every_step' or 'intermittent', got {self.replan_mode!r}")
        self.replan_latency_s = float(config.get('replan_latency_s', 0.19))
        self.replan_latency_cv = float(config.get('replan_latency_cv', 0.89))
        self.replan_deviation_frac = float(config.get('replan_deviation_frac', 0.15))
        self.horizon_min_steps = int(config.get('horizon_min_steps', 2))
        self.horizon_max_steps = int(config.get('horizon_max_steps', 40))
        # Diagnostics of the most recent generate_* call (replan events etc.).
        self.last_diagnostics = None

        seed = config['random_seed']
        if seed is not None:
            np.random.seed(seed)

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
            w_suppress           = rp['w_suppress']
            w_width_exp          = rp['w_width_exp']
            cut_window_frac      = rp['cut_window_frac']
            global_clearance_ref = rp['global_clearance_ref']

            # Cache centerline spline — reused later for curvature rate profile
            centerline_corridor_bounds = None
            centerline_spline = ReferencePath(waypoints_norm, s=0.0, k=3)
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
                w_suppress=w_suppress,
                w_width_exp=w_width_exp,
                cut_window_frac=cut_window_frac,
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
        curvature_rate_profile = None
        curvature_profile = None
        if reference_path.total_length > 0:
            n_profile = 500
            s_profile = np.linspace(0, reference_path.total_length, n_profile)
            c_profile = compute_clearance_profile(
                reference_path, s_profile,
                corridor_bounds=corridor_bounds,
                cartesian_constraints=cartesian_regions if cartesian_regions else None,
            )
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
            _, k_prof = curvature_profile
            _, r_prof = curvature_rate_profile
            # Reference speed along the path — same signals model() feeds the
            # speed model — used to convert the arc-length lookahead to steps.
            v_ref_prof = self.speed_model.compute_speed_profile(
                s_prof, c_prof, k_prof, r_prof)
            budget_horizon = DifficultyBudgetHorizon(
                s_prof, c_prof, k_prof, v_ref_prof,
                D0=self.budget_D0, lam=self.budget_lam,
                T_min=self.budget_T_min,
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
            deviation_frac=self.replan_deviation_frac,
        )
        plan = None            # planned velocity/displacement sequences of the current solve
        theta_track = 0.0      # warm guess for arc-length projection

        # Termination: DWELL_S of consecutive samples inside the target
        # (stands in for the human click latency; the pointing data show
        # ~0.3 s median from final target entry to click).
        dwell_required = int(round(self.dwell_s / self.interval))
        dwell_steps = 0
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
                    and plan.get('w_solve')):
                k_exec = min(scheduler.plan_idx - 2, plan['n_steps'] - 1)
                dev = float(np.hypot(
                    cursor_pos[0] - plan['pos_x'][k_exec],
                    cursor_pos[1] - plan['pos_y'][k_exec]))
                deviation_ratio = dev / plan['w_solve']
            trigger = scheduler.needs_replan(
                theta_now if theta_now is not None else -np.inf,
                deviation_ratio=deviation_ratio)

            if trigger is not None:
                theta0 = float(reference_path.find_closest_theta(
                    cursor_pos, initial_guess=theta_track))
                theta_track = theta0
                if budget_horizon is not None:
                    anchor_s = budget_horizon.anchor(
                        theta0, v_now=float(np.hypot(cursor_vel[0], cursor_vel[1])))
                    n_base = int(np.ceil(
                        budget_horizon.traverse_time(theta0, anchor_s) / self.interval))
                    # Anchor and floor both cap at the path end, so the
                    # traverse time (and with it n_base) shrinks toward zero
                    # there; the plan itself must still cover >= T_min.
                    n_base = max(n_base, floor_steps)
                else:
                    n_base = self.pred_horizon
                    anchor_s = None  # fixed mode: set from the solved plan below
                if anchor_s is not None:
                    anchor_s = min(float(anchor_s), float(reference_path.total_length))
                if self.replan_mode == 'intermittent':
                    # The plan must survive the post-arrival latency (the old
                    # plan keeps executing past the anchor), plus the same
                    # margin again as slack for arrival running later than the
                    # reference-speed estimate (start-up, corners, noise).
                    # This is plan availability, not behaviour: the replan time
                    # is still set by arrival + tau; exhaustion only backstops.
                    n_solve = n_base + 2 * tau_steps
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

                cursor_info, plan_debug = model(model_input)
                c_pos_dx, c_pos_dy, c_vel_x, c_vel_y = cursor_info
                # Absolute planned positions after each step, and the usable
                # width at the solve position — the deviation trigger's scale.
                w_solve = None
                if clearance_profile is not None:
                    s_cl, c_cl = clearance_profile
                    w_solve = float(np.interp(theta0, s_cl, c_cl))
                plan = {
                    'c_pos_dx': c_pos_dx, 'c_pos_dy': c_pos_dy,
                    'c_vel_x': c_vel_x, 'c_vel_y': c_vel_y,
                    'n_steps': len(c_pos_dx),
                    'pos_x': cursor_pos[0] + np.cumsum(c_pos_dx),
                    'pos_y': cursor_pos[1] + np.cumsum(c_pos_dy),
                    'w_solve': w_solve if w_solve and w_solve > 0 else None,
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
                scheduler.on_replan(
                    ReplanEvent(step=step, t=current_time, theta=theta0,
                                anchor=anchor_s, n_steps=n_solve, trigger=trigger),
                    anchor=anchor_s, plan_len=plan['n_steps'],
                )

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

            screen_x = cursor_pos[0] / screen_width_m * screen_width
            screen_y = cursor_pos[1] / screen_height_m * screen_height

            if return_timestamps:
                trajectory.append((screen_x, screen_y, current_time))
            else:
                trajectory.append((screen_x, screen_y, self.interval))

            current_time += self.interval

        self.last_diagnostics = {
            'horizon_mode': self.horizon_mode,
            'replan_mode': self.replan_mode,
            'budget_T_min': self.budget_T_min,
            'replan_latency_cv': self.replan_latency_cv,
            'replan_deviation_frac': self.replan_deviation_frac,
            'interval': self.interval,
            'n_steps_executed': len(trajectory),
            'n_solves': len(scheduler.events),
            'total_length': float(reference_path.total_length),
            'replan_events': [
                {'step': e.step, 't': e.t, 'theta': e.theta, 'anchor': e.anchor,
                 'n_steps': e.n_steps, 'trigger': e.trigger}
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