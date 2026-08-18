
import numpy as np
from .point_and_click_modules import mouse_module as mouse
from .point_and_click_modules import upper_limb_module as limb

def motor_and_device_noise(c_vel_x, c_vel_y, h_pos_x, h_pos_y, pred_horizon, nc, interval, forearm):
    """Apply motor and device noise across the prediction horizon."""
    assert len(c_vel_x) >= pred_horizon
    assert len(c_vel_y) >= pred_horizon
    vx = np.asarray(c_vel_x[:pred_horizon + 1], dtype=float)
    vy = np.asarray(c_vel_y[:pred_horizon + 1], dtype=float)
    speed = np.sqrt(vx**2 + vy**2)
    mouse_gain = np.asarray(mouse.gain_func_can(speed), dtype=float)

    tiny = np.finfo(float).tiny
    mouse_gain[~np.isfinite(mouse_gain)] = tiny
    mouse_gain = np.maximum(mouse_gain, tiny)
    inv_gain = 1.0 / mouse_gain

    vx_dev = vx * inv_gain
    vy_dev = vy * inv_gain

    _, _, h_pos_x_ideal, h_pos_y_ideal = limb.mouse_noise(vx_dev, vy_dev, h_pos_x, h_pos_y, forearm, mouse_gain, interval)
    h_pos_delta_x = h_pos_x_ideal - h_pos_x
    h_pos_delta_y = h_pos_y_ideal - h_pos_y

    _, _, vx_noisy_dev, vy_noisy_dev = limb.motor_noise(vx_dev, vy_dev, pred_horizon, 0, 0, nc, interval)

    vx_noisy = vx_noisy_dev * mouse_gain
    vy_noisy = vy_noisy_dev * mouse_gain
    pos_dx_mouse, pos_dy_mouse, h_pos_x, h_pos_y = limb.mouse_noise(
        vx_noisy.copy(), vy_noisy.copy(), h_pos_x, h_pos_y, forearm, mouse_gain, interval
    )

    # Trapezoid baseline to estimate mouse-noise bias
    c_pos_dx_base = ((vx_noisy[1:] + vx_noisy[:-1]) / 2) * interval
    c_pos_dy_base = ((vy_noisy[1:] + vy_noisy[:-1]) / 2) * interval
    denom = max(pred_horizon * interval, np.finfo(float).eps)
    vel_mouse_noise_x = (np.sum(pos_dx_mouse) - np.sum(c_pos_dx_base)) / denom
    vel_mouse_noise_y = (np.sum(pos_dy_mouse) - np.sum(c_pos_dy_base)) / denom

    c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out = limb.motor_noise(
        vx_noisy, vy_noisy, pred_horizon, vel_mouse_noise_x, vel_mouse_noise_y, [0, 0], interval
    )

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
    forearm,
    c_vel_x_prev=None,
    c_vel_y_prev=None,
):
    """Apply motor and device noise for a single step.

    Args:
        c_vel_x, c_vel_y: Planned cursor velocity at the END of the step
            (m/s); the signal-dependent motor noise is applied to this.
        h_pos_x, h_pos_y: Current hand position in physical space (m).
        nc: [directional, perpendicular] motor noise coefficients.
        interval: Time step (s).
        forearm: Forearm length (m) for coordinate transformation.
        c_vel_x_prev, c_vel_y_prev: Velocity at the START of the step (m/s).
            Used as the base of the trapezoidal position integral so that, in
            the zero-noise limit, the executed displacement equals the planner's
            true first-step displacement c_pos_dx[0] = (v_start + v_end)/2 * dt.
            Defaults to (c_vel_x, c_vel_y) for backward compatibility (which
            reproduces the old forward-Euler behaviour using the end velocity).

    Returns:
        (c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out,
         h_pos_x_out, h_pos_y_out, h_pos_delta_x, h_pos_delta_y).
    """
    if c_vel_x_prev is None:
        c_vel_x_prev = c_vel_x
    if c_vel_y_prev is None:
        c_vel_y_prev = c_vel_y
    # motor_noise expects [current, next] velocity pairs
    vx = np.array([float(c_vel_x), float(c_vel_x)], dtype=float)
    vy = np.array([float(c_vel_y), float(c_vel_y)], dtype=float)

    speed = float(np.sqrt(c_vel_x**2 + c_vel_y**2))
    mouse_gain_val = float(mouse.gain_func_can(speed))

    tiny = np.finfo(float).tiny
    if not np.isfinite(mouse_gain_val) or mouse_gain_val < tiny:
        mouse_gain_val = tiny

    inv_gain = 1.0 / mouse_gain_val

    vx_dev = vx * inv_gain
    vy_dev = vy * inv_gain

    # Ideal hand position (for the ideal-delta reference)
    mouse_gain_arr = np.array([mouse_gain_val, mouse_gain_val], dtype=float)
    _, _, h_pos_x_ideal, h_pos_y_ideal = limb.mouse_noise(
        vx_dev.copy(), vy_dev.copy(), h_pos_x, h_pos_y, forearm, mouse_gain_arr, interval
    )
    h_pos_delta_x_ideal = h_pos_x_ideal - h_pos_x
    h_pos_delta_y_ideal = h_pos_y_ideal - h_pos_y

    # Motor noise in device frame on the "next" step
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

    vx_noisy = vx_noisy_dev * mouse_gain_val
    vy_noisy = vy_noisy_dev * mouse_gain_val

    # Trapezoidal integration between the step's START velocity and the (noisy)
    # END velocity: dx = (v_start + v_end_noisy) / 2 * interval. Using the start
    # velocity as the base (rather than the end velocity twice) makes this the
    # zero-noise-consistent counterpart of the planner's c_pos_dx[0].
    dx = (c_vel_x_prev + vx_noisy) / 2.0 * interval
    dy = (c_vel_y_prev + vy_noisy) / 2.0 * interval

    hand_ori_prev = mouse.get_hand_orientation(np.array((h_pos_x, h_pos_y)), forearm)

    h_pos_x_out = float(h_pos_x)
    h_pos_y_out = float(h_pos_y)

    if abs(dx) > 1e-10 or abs(dy) > 1e-10:
        hand_pos_prev = np.array((h_pos_x, h_pos_y), dtype=float)
        # dx/dy are CURSOR-space displacements; the hand moves in DEVICE space
        # (cursor / gain). get_cursor_displacement below multiplies by the gain
        # again, so feeding cursor-space dx here made the executed step
        # gain-times (1.2x-4.8x, growing with speed) larger than planned while
        # the velocity state stayed at the planned value — a planner/plant
        # gain mismatch that produced systematic overshoot.
        hand_dx, hand_dy = mouse.rot_mat(dx * inv_gain, dy * inv_gain, -hand_ori_prev)
        h_pos_x_out += float(hand_dx)
        h_pos_y_out += float(hand_dy)

        mouse_dx, mouse_dy = mouse.get_cursor_displacement(
            hand_pos_prev,
            np.array((h_pos_x_out, h_pos_y_out), dtype=float),
            forearm,
            mouse_gain_val
        )
        dx = float(mouse_dx)
        dy = float(mouse_dy)

    c_pos_dx = float(dx)
    c_pos_dy = float(dy)

    c_vel_x_out = float(vx_noisy)
    c_vel_y_out = float(vy_noisy)

    if not np.isfinite(c_pos_dx):
        c_pos_dx = 0.0
    if not np.isfinite(c_pos_dy):
        c_pos_dy = 0.0
    if not np.isfinite(c_vel_x_out):
        c_vel_x_out = 0.0
    if not np.isfinite(c_vel_y_out):
        c_vel_y_out = 0.0

    if not np.isfinite(h_pos_x_out):
        h_pos_x_out = float(h_pos_x_ideal)
    if not np.isfinite(h_pos_y_out):
        h_pos_y_out = float(h_pos_y_ideal)

    h_pos_delta_x = float(h_pos_x_out - h_pos_x)
    h_pos_delta_y = float(h_pos_y_out - h_pos_y)

    return c_pos_dx, c_pos_dy, c_vel_x_out, c_vel_y_out, h_pos_x_out, h_pos_y_out, \
           h_pos_delta_x, h_pos_delta_y