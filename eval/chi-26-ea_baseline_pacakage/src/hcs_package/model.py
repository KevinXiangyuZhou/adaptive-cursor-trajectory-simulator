"""
Steering model

Reference:
1) "A Simulation Model of Intermittently Controlled Point-and-Click Behavior"
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from .params import SteeringModelInput
from .reference_path import ReferencePath


def model(model_input: SteeringModelInput):
    """Steering model using structured parameters with reference path MPC."""

    # Unpack structured inputs
    state_cog = model_input.state_cog
    pred_horizon = model_input.bump.pred_horizon
    interval = model_input.env.interval
    tunnel_info = (
        model_input.tunnel.tunnel_path,
        model_input.tunnel.tunnel_width,
        model_input.tunnel.top_wall,
        model_input.tunnel.bottom_wall,
    )

    # States
    cursor_pos_x, cursor_pos_y, cursor_vel_x, cursor_vel_y = state_cog
    tunnel_path, tunnel_width, top_wall, bottom_wall = tunnel_info

    # Parameters for the models (nc is not used in model function)
    
    # Use passed reference path or create from tunnel centerline
    if model_input.reference_path is not None:
        ref_path = model_input.reference_path
    else:
        ref_path = ReferencePath(tunnel_path, s=0.0, k=3)
    
    # Find current progress θ along the path
    current_pos = np.array([cursor_pos_x, cursor_pos_y])
    theta_0 = ref_path.find_closest_theta(current_pos)
    
    
    current_acc = getattr(model_input, 'current_acc', (0.0, 0.0))
    if current_acc is None: current_acc = (0.0, 0.0)
    ax0, ay0 = current_acc
    
    # Setup full state vector for MPCC: [px, py, vx, vy, ax, ay, s]
    state_0 = [cursor_pos_x, cursor_pos_y, cursor_vel_x, cursor_vel_y, ax0, ay0, theta_0]
    
    # Limits and Safety
    # Assume some reasonable defaults if not provided
    limits = {'acc_max': 100.0}
    
    # Get corridor_bounds from input if provided, otherwise compute from tunnel_width
    corridor_bounds = getattr(model_input, 'corridor_bounds', None)
    
    if corridor_bounds is None:
        # Fallback: Convert tunnel_width to corridor_bounds (asymmetric bounds)
        # For symmetric tunnel: both bounds = half_width
        if tunnel_width is not None:
            half_width = float(tunnel_width) / 2.0  # Convert width to radius
            # We don't wait until going outside the tunnel to penalize
            # Apply 0.8 factor to both bounds for early penalty
            bound_value = half_width * 0.8
            corridor_bounds = (bound_value, bound_value)  # Symmetric bounds
        else:
            corridor_bounds = None
    
    # Get desired_speed from planner_weights if available, otherwise use default
    desired_speed = 0.12
    if model_input.planner_weights and 'desired_speed' in model_input.planner_weights:
        desired_speed = float(model_input.planner_weights['desired_speed'])
    
    controls, opt_info = generate_mpcc(
        ref_path=ref_path,
        state_0=state_0,
        num_steps=pred_horizon,
        dt=interval,
        weights=model_input.planner_weights,
        limits=limits,
        desired_speed=desired_speed,
        corridor_bounds=corridor_bounds
    )
    
    # Process output
    # Reconstruct acceleration and velocity profiles from Jerk controls
    # controls: [jx, jy, vs]
    # We need to return info compatible with existing pipeline
    # c_pos_dx, c_pos_dy: delta positions per step
    # c_vel_x, c_vel_y: velocities
    
    jx = controls[:, 0]
    jy = controls[:, 1]
    
    # Integrate to get trajectory
    # We need vectors of length pred_horizon
    
    # Vectorized integration using dynamics matrices
    # Reuse _build... functions
    # A_acc = _build_A_acc(pred_horizon, interval)
    A_vel = _build_A_vel_from_jerk(pred_horizon, interval)
    A_pos = _build_A_pos_from_jerk(pred_horizon, interval)
    
    t_vec = np.arange(1, pred_horizon + 1) * interval
    t2_vec = 0.5 * t_vec ** 2
    
    # Free response
    # For zero jerk (j=0), position is: p(t) = p0 + v0*t + 0.5*a0*t^2
    vx_free = cursor_vel_x + ax0 * t_vec
    vy_free = cursor_vel_y + ay0 * t_vec
    
    px_free = cursor_pos_x + cursor_vel_x * t_vec + 0.5 * ax0 * (t_vec**2)
    py_free = cursor_pos_y + cursor_vel_y * t_vec + 0.5 * ay0 * (t_vec**2)
    
    pos_x = px_free + A_pos @ jx
    pos_y = py_free + A_pos @ jy
    vel_x = vx_free + A_vel @ jx
    vel_y = vy_free + A_vel @ jy
    
    # Add initial velocity to start of velocity array (t=0)
    c_vel_x = np.insert(vel_x, 0, cursor_vel_x)
    c_vel_y = np.insert(vel_y, 0, cursor_vel_y)
    
    # Delta positions
    # pos_x contains p1...pN. p0 is cursor_pos_x.
    all_pos_x = np.insert(pos_x, 0, cursor_pos_x)
    all_pos_y = np.insert(pos_y, 0, cursor_pos_y)
    c_pos_dx = np.diff(all_pos_x)
    c_pos_dy = np.diff(all_pos_y)
    
    # Ideal planned segment (absolute positions)
    ideal_seg_x = pos_x
    ideal_seg_y = pos_y
    
    # Summary of the outputs
    cursor_info = c_pos_dx, c_pos_dy, c_vel_x, c_vel_y
    
    # Debug info with reference path target
    ref_target = ref_path(theta_0)
    plan_debug = {
        "ideal_segment": (ideal_seg_x.tolist(), ideal_seg_y.tolist()),
        "target_waypoint": (float(ref_target[0]), float(ref_target[1])),
        "theta": float(theta_0),
        "opt_info": opt_info,  # Include optimization information
    }
    # print(f"Plan debug: {plan_debug}")
    # print("--------------------------------")
    return cursor_info, plan_debug


def _build_A_acc(num_steps, dt):
    """Build matrix mapping jerk to acceleration via integration."""
    # a_k = a_0 + sum(j * dt)
    # This matrix maps J vector to A vector update (excluding initial a0)
    # Lower triangular with dt
    A_acc = np.tril(np.ones((num_steps, num_steps))) * dt
    return A_acc


def _build_A_vel_from_jerk(num_steps, dt):
    """Build matrix mapping jerk to velocity."""
    # Discrete update: v_{k+1} = v_k + a_k dt + 0.5 j_k dt^2
    # Matrix derived from recursive application
    A_v = np.zeros((num_steps, num_steps))
    for k in range(num_steps): # State index k (t = (k+1)dt)
        for i in range(k + 1): # Input index i
            # Coeff for j_i contributing to v_{k+1}
            # For i < k: (k - i + 0.5) * dt^2
            # For i = k: 0.5 * dt^2
            A_v[k, i] = (k - i + 0.5) * (dt ** 2)
    return A_v


def _build_A_pos_from_jerk(num_steps, dt):
    """Build matrix mapping jerk to position."""
    # Discrete update: p_{k+1} = p_k + v_k dt + 0.5 a_k dt^2 + (1/6) j_k dt^3
    A_p = np.zeros((num_steps, num_steps))
    # Fill by impulse response simulation
    for i in range(num_steps):
        # Impulse j at step i (affects states starting from i+1)
        dp, dv, da = 0.0, 0.0, 0.0
        for k in range(num_steps):
            j_val = 1.0 if k == i else 0.0
            
            # Update for step k -> k+1
            term_p = dp + dv * dt + 0.5 * da * (dt**2) + (1.0/6.0) * j_val * (dt**3)
            term_v = dv + da * dt + 0.5 * j_val * (dt**2)
            term_a = da + j_val * dt
            
            dp, dv, da = term_p, term_v, term_a
            
            # Store impact on state k (which is time (k+1)*dt)
            A_p[k, i] = dp
            
    return A_p


def generate_mpcc(
    ref_path,
    state_0,
    num_steps,
    dt,
    weights,
    limits,
    desired_speed=1.0,
    corridor_bounds=None,
):
    """
    Generate MPCC (Model Predictive Contouring Control) plan.
    
    Args:
        ref_path: ReferencePath object
        state_0: Initial state [px, py, vx, vy, ax, ay, s]
        num_steps: Prediction horizon (N)
        dt: Time step
        weights: Dictionary of weights (jerk, progress, tracking, wall)
        limits: Dictionary of limits (acc_max)
        desired_speed: Target virtual speed for lag cost
        corridor_bounds: Tuple (bound_left, bound_right) for asymmetric corridor constraints.
            Each bound can be:
            - A float: static width
            - A callable function: f(s) -> float (for dynamic width, e.g., Lasso funnel)
            - None: no wall avoidance (if corridor_bounds is None)
            Examples:
            - Symmetric: (40.0, 40.0)
            - Partially constrained: (30.0, 10000.0) - left wall only
            - Dynamic: (lambda s: 50 - s*0.1, lambda s: 50 - s*0.1) - narrowing funnel
        
    Returns:
        controls: (N, 3) array of [jx, jy, vs]
        opt_info: optimization result info
    """
    # Unpack weights
    if weights is None:
        weights = {}
        
    # Get weights from config, with reasonable defaults
    # Jerk cost needs to be very small because jerk units (m/s^3) are large relative to step displacement
    w_jerk = weights.get('jerk', 1.5e-6)
    w_tracking = weights.get('tracking', 1.0)  # Weight for tracking error (contour + lag)
    w_progress = weights.get('progress', 1.0e-5)
    w_corridor = weights.get('wall', 1e3)  # Weight for wall avoidance (deadband penalty)
    w_contour = weights.get('contour', 1.0)  # Weight for contour error
    w_lag = weights.get('lag', 0.1)  # Weight for lag error
    
    # Get desired_speed from weights if provided, otherwise use function parameter
    if weights and 'desired_speed' in weights:
        desired_speed = float(weights['desired_speed'])
    
    # Get curvature scaling factor for dynamic speed adaptation
    # Higher values = more aggressive speed reduction in curves
    curvature_scale = weights.get('curvature_scale', 10.0)  # Default: moderate scaling
    
    # Scale factors for optimization variables to improve conditioning
    # jx/jy ~ 100-1000, vs ~ desired_speed
    SCALE_JERK = 1000.0
    # TODO: Automatically adapt SCALE_VS to keep optimization variable near 1.0
    # We clamp it to a minimum of 0.1 to avoid division by near-zero if desired_speed is tiny
    SCALE_VS = max(0.1, desired_speed)
    
    # Unpack limits
    acc_max = limits.get('acc_max', 100.0)
    
    # Initial state
    px0, py0, vx0, vy0, ax0, ay0, s0 = state_0
    
    # Pre-compute dynamics matrices
    # Maps J vector (N,) to P, V, A vectors (N,)
    A_acc_mat = _build_A_acc(num_steps, dt)
    A_vel_mat = _build_A_vel_from_jerk(num_steps, dt)
    A_pos_mat = _build_A_pos_from_jerk(num_steps, dt)
    
    # Time vector for initial condition propagation
    t_vec = np.arange(1, num_steps + 1) * dt
    t2_vec = 0.5 * t_vec ** 2
    
    # Initial state propagation (free response)
    # p_free = p0 + v0*t + 0.5*a0*t^2
    px_free = px0 + vx0 * t_vec + ax0 * t2_vec
    py_free = py0 + vy0 * t_vec + ay0 * t2_vec
    
    # v_free = v0 + a0*t
    vx_free = vx0 + ax0 * t_vec
    vy_free = vy0 + ay0 * t_vec
    
    # a_free = a0 (constant if j=0)
    ax_free = np.full(num_steps, ax0)
    ay_free = np.full(num_steps, ay0)
    
    # Optimization variables:
    # x = [jx_0..N-1, jy_0..N-1, vs_0..N-1]
    # Total 3 * num_steps variables
    n_vars = 3 * num_steps
    
    # Indices
    idx_jx = slice(0, num_steps)
    idx_jy = slice(num_steps, 2 * num_steps)
    idx_vs = slice(2 * num_steps, 3 * num_steps)
    
    # Progress integration matrix (vs -> s)
    # s_k = s_0 + sum_{i=0}^{k-1} vs_i * dt
    # Lower triangular matrix of dt
    S_mat = np.tril(np.ones((num_steps, num_steps))) * dt
    
    def unpack_x(x):
        jx = x[idx_jx] * SCALE_JERK
        jy = x[idx_jy] * SCALE_JERK
        vs = x[idx_vs] * SCALE_VS
        return jx, jy, vs
        
    def objective(x):
        jx, jy, vs = unpack_x(x)
        
        # 1. Smoothness (Jerk)
        j_cost = np.sum(jx**2 + jy**2) * w_jerk
        
        # 2. Progress reward
        # Compute s trajectory
        s_traj = s0 + S_mat @ vs
        
        # Reconstruct velocities to compute physical speed
        vx = vx_free + A_vel_mat @ jx
        vy = vy_free + A_vel_mat @ jy
        
        # Compute actual physical speed (magnitude of velocity vector)
        physical_speed = np.sqrt(vx**2 + vy**2)
        
        # Compute curvature-dependent desired speed
        # Evaluate curvature at each point along the trajectory
        curvature = np.zeros(num_steps)
        for k in range(num_steps):
            curvature[k] = abs(ref_path.curvature(float(s_traj[k])))
        
        # Scale desired speed based on curvature
        # Formula: dynamic_desired_speed = desired_speed / (1 + curvature_scale * |curvature|)
        # This gives: desired_speed when curvature=0, approaches 0 when curvature is high
        dynamic_desired_speed = desired_speed / (1.0 + curvature_scale * curvature)
        
        # Progress cost is the difference between dynamic desired speed and actual physical speed
        prog_cost = np.sum((physical_speed - dynamic_desired_speed)**2) * w_progress
        
        # 3. Tracking Error (Contour + Lag)
        # Reconstruct positions
        px = px_free + A_pos_mat @ jx
        py = py_free + A_pos_mat @ jy
        
        # Query reference path
        # Note: ReferencePath uses splprep/splev which are not inherently vectorized for single calls,
        # but ref_path(s_array) might work if implemented with splev(s_array).
        # existing ReferencePath.__call__ does: splev(u, tck) where u is mapped from s.
        # This SHOULD be vectorized if u is an array.
        # Let's try passing the array.
        ref_pts = ref_path(s_traj).T # returns (2, N) if vectorized, so .T -> (N, 2)
        if ref_pts.shape != (num_steps, 2):
            # Fallback to loop if shape is wrong (e.g. if it didn't vectorize)
            ref_pts = np.zeros((num_steps, 2))
            for k in range(num_steps):
                ref_pts[k] = ref_path(float(s_traj[k]))
       
        rx, ry = ref_pts[:, 0], ref_pts[:, 1]
        
        # Compute tracking error in path's local frame
        # Error vector: e_k = R(φ(θ_k)) * (p_k - p^r(θ_k))
        # where R(φ) = [ sin φ  -cos φ ]
        #              [-cos φ  -sin φ ]
        # e^c (contour) = first component (lateral deviation)
        # e^l (lag) = second component (longitudinal deviation)
        tracking_cost = 0.0
        
        for k in range(num_steps):
            # Position error in global frame
            pos_k = np.array([px[k], py[k]], dtype=float)
            ref_k = np.array([rx[k], ry[k]], dtype=float)
            pos_error = pos_k - ref_k
            
            # Get path heading at θ_k = s_traj[k]
            tangent = ref_path.tangent(s_traj[k])
            # tangent = [cos φ, sin φ]
            cos_phi = tangent[0]
            sin_phi = tangent[1]
            
            # Transformation matrix R(φ)
            # [ sin φ  -cos φ ]
            # [-cos φ  -sin φ ]
            R = np.array([
                [sin_phi, -cos_phi],
                [-cos_phi, -sin_phi]
            ], dtype=float)
            
            # Transform error to path's local frame
            e_k = R @ pos_error  # (2,)
            
            e_contour = e_k[0] # Lateral deviation (Distance from path)
            e_lag     = e_k[1] # Longitudinal deviation (Distance behind/ahead of virtual target)
            
            # Apply Weighted Costs (Equation 9 in MPCC paper)
            # We penalize contour error heavily (to stay in tunnel) 
            # but lag error lightly (to allow slowing down for corners/walls)
            
            # Replaces: tracking_cost += w_tracking * np.dot(e_k, e_k)
            tracking_cost += (w_contour * e_contour**2) + (w_lag * e_lag**2)
            
            
            # 4. Asymmetric Wall Avoidance: One-Sided Quadratic Penalty (Deadband)
            # Formula: J_wall = w_corridor * (max(0, e^c_k - W_left)^2 + max(0, -e^c_k - W_right)^2)
            # where e^c_k = e_k[0] is the lateral contour error
            # e^c_k > 0 means deviation to the left, e^c_k < 0 means deviation to the right
            if corridor_bounds is not None:
                # 1. Resolve Bounds
                # corridor_bounds is a tuple: (left_input, right_input)
                b_left_in, b_right_in = corridor_bounds
                
                # Evaluate dynamic (function) or static (float) bounds based on progress s_traj[k]
                w_left = b_left_in(s_traj[k]) if callable(b_left_in) else float(b_left_in)
                w_right = b_right_in(s_traj[k]) if callable(b_right_in) else float(b_right_in)
                
                # 2. Calculate One-Sided Violations (Deadband)
                # e_k[0] is lateral contour error (+ is Left, - is Right)
                violation_left = max(0.0, e_k[0] - w_left)
                violation_right = max(0.0, -e_k[0] - w_right)
                
                # 3. Add Quadratic Cost
                # Squared term ensures C1 continuity for the solver
                tracking_cost += w_corridor * (violation_left**2 + violation_right**2)

        return j_cost + prog_cost + tracking_cost

    # Constraints
    constraints = []
    
    # 1. Physical limits (Acceleration)
    # |ax| <= a_max, |ay| <= a_max
    # Linear constraints: A_acc @ jx <= a_max - ax_free, etc.
    
    def constraint_acc_x_max(x):
        jx = x[idx_jx]
        ax = ax_free + A_acc_mat @ jx
        return acc_max - ax # >= 0
        
    def constraint_acc_x_min(x):
        jx = x[idx_jx]
        ax = ax_free + A_acc_mat @ jx
        return ax + acc_max # >= 0
        
    def constraint_acc_y_max(x):
        jy = x[idx_jy]
        ay = ay_free + A_acc_mat @ jy
        return acc_max - ay
        
    def constraint_acc_y_min(x):
        jy = x[idx_jy]
        ay = ay_free + A_acc_mat @ jy
        return ay + acc_max

    constraints.extend([
        {'type': 'ineq', 'fun': constraint_acc_x_max},
        {'type': 'ineq', 'fun': constraint_acc_x_min},
        {'type': 'ineq', 'fun': constraint_acc_y_max},
        {'type': 'ineq', 'fun': constraint_acc_y_min}
    ])
    
    # Bounds for v_s (>= 0)
    bounds = []
    # Jx, Jy: No explicit bounds (controlled by Acc limits)
    bounds.extend([(None, None)] * num_steps)
    bounds.extend([(None, None)] * num_steps)
    # Vs: >= 0
    bounds.extend([(0.0, None)] * num_steps)
    
    # Initial guess
    x0_guess = np.zeros(n_vars)
    x0_guess[idx_vs] = desired_speed / SCALE_VS
    
    # Solve
    result = minimize(
        objective, 
        x0_guess, 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 10000, 'eps': 1e-8} 
    )
    
    # Extract results
    jx_opt, jy_opt, vs_opt = unpack_x(result.x)
    
    controls = np.column_stack((jx_opt, jy_opt, vs_opt))
    
    opt_info = {
        'success': result.success,
        'cost': result.fun,
        'message': result.message,
        'nit': result.nit
    }
    
    return controls, opt_info
