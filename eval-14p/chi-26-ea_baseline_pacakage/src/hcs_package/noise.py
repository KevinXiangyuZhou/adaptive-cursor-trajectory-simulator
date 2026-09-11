
import numpy as np
from .point_and_click_modules import mouse_module as mouse
from .point_and_click_modules import upper_limb_module as limb

def motor_and_device_noise(c_vel_x, c_vel_y, h_pos_x, h_pos_y, pred_horizon, nc, interval, forearm):
    """
    Apply motor and device noise.
    """
    # Set the mouse noise
    assert len(c_vel_x) >= pred_horizon
    assert len(c_vel_y) >= pred_horizon
    # Use exactly pred_horizon samples (your c_vel_* are pulse_steps+1)
    vx = np.asarray(c_vel_x[:pred_horizon + 1], dtype=float)
    vy = np.asarray(c_vel_y[:pred_horizon + 1], dtype=float)    
    speed = np.sqrt(vx**2 + vy**2)
    mouse_gain = np.asarray(mouse.gain_func_can(speed), dtype=float)

    tiny = np.finfo(float).tiny
    # Replace non-finite and clamp to tiny to avoid division by zero
    mouse_gain[~np.isfinite(mouse_gain)] = tiny
    mouse_gain = np.maximum(mouse_gain, tiny)
    inv_gain = 1.0 / mouse_gain  # now safe
    
    # Device-frame velocities (safe)
    vx_dev = vx * inv_gain
    vy_dev = vy * inv_gain
    
    # Get the ideal hand pos
    _, _, h_pos_x_ideal, h_pos_y_ideal = limb.mouse_noise(vx_dev, vy_dev, h_pos_x, h_pos_y, forearm, mouse_gain, interval)
    h_pos_delta_x = h_pos_x_ideal - h_pos_x
    h_pos_delta_y = h_pos_y_ideal - h_pos_y
    # print(f"Hand position ideal: ({h_pos_x_ideal}, {h_pos_y_ideal}), Delta: ({h_pos_delta_x}, {h_pos_delta_y})")

    # Set the motor noise
    _, _, vx_noisy_dev, vy_noisy_dev = limb.motor_noise(vx_dev, vy_dev, pred_horizon, 0, 0, nc, interval)

    # Back to screen frame
    vx_noisy = vx_noisy_dev * mouse_gain
    vy_noisy = vy_noisy_dev * mouse_gain
    # Set the mouse noise
    # Mouse noise to positions (screen frame)
    pos_dx_mouse, pos_dy_mouse, h_pos_x, h_pos_y = limb.mouse_noise(
        vx_noisy.copy(), vy_noisy.copy(), h_pos_x, h_pos_y, forearm, mouse_gain, interval
    )

    # Trapezoid baseline to estimate bias
    c_pos_dx_base = ((vx_noisy[1:] + vx_noisy[:-1]) / 2) * interval
    c_pos_dy_base = ((vy_noisy[1:] + vy_noisy[:-1]) / 2) * interval
    denom = max(pred_horizon * interval, np.finfo(float).eps)
    vel_mouse_noise_x = (np.sum(pos_dx_mouse) - np.sum(c_pos_dx_base)) / denom
    vel_mouse_noise_y = (np.sum(pos_dy_mouse) - np.sum(c_pos_dy_base)) / denom

    # Final motor noise (screen frame)
    c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out = limb.motor_noise(
        vx_noisy, vy_noisy, pred_horizon, vel_mouse_noise_x, vel_mouse_noise_y, [0, 0], interval
    )

    # Final guards
    for arr in (c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out):
        arr[~np.isfinite(arr)] = 0.0

    h_pos_x = float(h_pos_x) if np.isfinite(h_pos_x) else float(h_pos_x_ideal)
    h_pos_y = float(h_pos_y) if np.isfinite(h_pos_y) else float(h_pos_y_ideal)

    return c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out, h_pos_x, h_pos_y, \
           h_pos_delta_x, h_pos_delta_y


def single_step_motor_and_device_noise(
    c_vel_x, 
    c_vel_y, 
    h_pos_x, 
    h_pos_y, 
    nc, 
    interval, 
    forearm
):
    """
    Apply motor and device noise for a single step.
    
    Args:
        c_vel_x: float - Current cursor velocity (x) in screen frame (m/s)
        c_vel_y: float - Current cursor velocity (y) in screen frame (m/s)
        h_pos_x: float - Current hand position (x) in physical space (m)
        h_pos_y: float - Current hand position (y) in physical space (m)
        nc: list[float] - [nc[0], nc[1]] motor noise coefficients [directional, perpendicular]
        interval: float - Time step (seconds)
        forearm: float - Forearm length (meters) for coordinate transformation
    
    Returns:
        tuple: (c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out, h_pos_x_out, h_pos_y_out, 
                h_pos_delta_x, h_pos_delta_y)
        - c_pos_dx: float - Position delta (x) for one step (m)
        - c_pos_dy: float - Position delta (y) for one step (m)
        - c_vel_x_out: float - Noisy velocity (x) after one step (m/s)
        - c_vel_y_out: float - Noisy velocity (y) after one step (m/s)
        - h_pos_x_out: float - Updated hand position (x) (m)
        - h_pos_y_out: float - Updated hand position (y) (m)
        - h_pos_delta_x: float - Hand position delta (x) (m)
        - h_pos_delta_y: float - Hand position delta (y) (m)
    """
    # Convert to numpy arrays for compatibility with existing functions
    # We need arrays of length 2: [current_vel, next_vel]
    # For single step, we'll use [current_vel, current_vel] initially
    vx = np.array([float(c_vel_x), float(c_vel_x)], dtype=float)
    vy = np.array([float(c_vel_y), float(c_vel_y)], dtype=float)
    
    # Compute speed magnitude
    speed = float(np.sqrt(c_vel_x**2 + c_vel_y**2))
    
    # Get mouse gain (single value)
    mouse_gain_val = float(mouse.gain_func_can(speed))
    
    # Handle edge cases
    tiny = np.finfo(float).tiny
    if not np.isfinite(mouse_gain_val) or mouse_gain_val < tiny:
        mouse_gain_val = tiny
    
    inv_gain = 1.0 / mouse_gain_val
    
    # Convert to device frame
    vx_dev = vx * inv_gain
    vy_dev = vy * inv_gain
    
    # Get ideal hand position (first pass, for computing ideal hand delta)
    # Create mouse_gain array for compatibility
    mouse_gain_arr = np.array([mouse_gain_val, mouse_gain_val], dtype=float)
    _, _, h_pos_x_ideal, h_pos_y_ideal = limb.mouse_noise(
        vx_dev.copy(), vy_dev.copy(), h_pos_x, h_pos_y, forearm, mouse_gain_arr, interval
    )
    h_pos_delta_x_ideal = h_pos_x_ideal - h_pos_x
    h_pos_delta_y_ideal = h_pos_y_ideal - h_pos_y
    
    # Apply motor noise in device frame (single step)
    # motor_noise expects arrays and modifies them in place
    # We'll process index 1 (the "next" step)
    v = np.array([vx_dev[1], vy_dev[1]], dtype=float)
    v_norm = np.linalg.norm(v)
    
    if v_norm == 0:
        vel_dir = np.array([0.0, 0.0], dtype=float)
    else:
        vel_dir = v / v_norm
    
    vel_per = np.array([-vel_dir[1], vel_dir[0]], dtype=float)
    
    vel_mag = abs(v_norm)
    noise_dir = float(nc[0] * vel_mag * np.random.normal(0, 1))
    noise_per = float(nc[1] * vel_mag * np.random.normal(0, 1))
    
    vel_noisy_dev = v + noise_dir * vel_dir + noise_per * vel_per
    vx_noisy_dev = float(vel_noisy_dev[0])
    vy_noisy_dev = float(vel_noisy_dev[1])
    
    # Convert back to screen frame
    vx_noisy = vx_noisy_dev * mouse_gain_val
    vy_noisy = vy_noisy_dev * mouse_gain_val
    
    # Apply mouse noise (single step)
    # Compute position delta using trapezoidal integration
    # dx = (v_current + v_next) / 2 * interval
    dx = (c_vel_x + vx_noisy) / 2.0 * interval
    dy = (c_vel_y + vy_noisy) / 2.0 * interval
    
    # Get hand orientation
    hand_ori_prev = mouse.get_hand_orientation(np.array((h_pos_x, h_pos_y)), forearm)
    
    # Apply mouse rotation effects
    h_pos_x_out = float(h_pos_x)
    h_pos_y_out = float(h_pos_y)
    
    if abs(dx) > 1e-10 or abs(dy) > 1e-10:
        hand_pos_prev = np.array((h_pos_x, h_pos_y), dtype=float)
        hand_dx, hand_dy = mouse.rot_mat(dx, dy, -hand_ori_prev)
        h_pos_x_out += float(hand_dx)
        h_pos_y_out += float(hand_dy)
        
        # Compute cursor displacement with mouse rotation
        mouse_dx, mouse_dy = mouse.get_cursor_displacement(
            hand_pos_prev, 
            np.array((h_pos_x_out, h_pos_y_out), dtype=float), 
            forearm, 
            mouse_gain_val
        )
        dx = float(mouse_dx)
        dy = float(mouse_dy)
    
    # Final position deltas
    c_pos_dx = float(dx)
    c_pos_dy = float(dy)
    
    # Final velocities
    c_vel_x_out = float(vx_noisy)
    c_vel_y_out = float(vy_noisy)
    
    # Final guards
    if not np.isfinite(c_pos_dx):
        c_pos_dx = 0.0
    if not np.isfinite(c_pos_dy):
        c_pos_dy = 0.0
    if not np.isfinite(c_vel_x_out):
        c_vel_x_out = 0.0
    if not np.isfinite(c_vel_y_out):
        c_vel_y_out = 0.0
    
    # Ensure hand positions are finite
    if not np.isfinite(h_pos_x_out):
        h_pos_x_out = float(h_pos_x_ideal)
    if not np.isfinite(h_pos_y_out):
        h_pos_y_out = float(h_pos_y_ideal)
    
    # Hand position deltas: actual change in hand position (output - input)
    # This represents the systematic noise effect from hand rotation and mouse transformation
    h_pos_delta_x = float(h_pos_x_out - h_pos_x)
    h_pos_delta_y = float(h_pos_y_out - h_pos_y)
    
    return c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out, h_pos_x_out, h_pos_y_out, \
           h_pos_delta_x, h_pos_delta_y