"""
Render comparison videos: human trajectory (top) vs model trajectory (bottom).

Two copies of the tunnel are drawn stacked vertically. The human cursor (blue)
animates in the upper tunnel and the model cursor (orange) in the lower tunnel.
Both play back in sync at real-time speed.
"""

import numpy as np
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle


# Colours
HUMAN_COLOR = "#2196F3"
MODEL_COLOR = "#FF9800"
TUNNEL_FILL = "#E8E8E8"
TUNNEL_EDGE = "#888888"
CURSOR_RADIUS_M = 0.004
BG_COLOR = "#FFFFFF"


def _interpolate_trajectory(trajectory, time_array, fps, total_duration):
    """Resample a trajectory to a fixed FPS timeline.

    Args:
        trajectory: [[x,y], ...] array
        time_array: time in seconds for each point (same length as trajectory)
        fps: target frames per second
        total_duration: total video duration in seconds

    Returns:
        (xs, ys) arrays of length n_frames
    """
    traj = np.array(trajectory, dtype=float)
    t = np.array(time_array, dtype=float)

    n_frames = max(1, int(total_duration * fps))
    t_target = np.linspace(0, total_duration, n_frames)

    # Clamp: after trajectory ends, hold final position
    fx = interp1d(t, traj[:, 0], kind="linear", bounds_error=False,
                  fill_value=(traj[0, 0], traj[-1, 0]))
    fy = interp1d(t, traj[:, 1], kind="linear", bounds_error=False,
                  fill_value=(traj[0, 1], traj[-1, 1]))

    return fx(t_target), fy(t_target), t_target


def _draw_tunnel(ax, tunnel_geom, label, label_color):
    """Draw static tunnel geometry on an axes."""
    left = tunnel_geom["left_boundary"]
    right = tunnel_geom["right_boundary"]

    # Filled polygon
    poly_x = np.concatenate([left[:, 0], right[::-1, 0]])
    poly_y = np.concatenate([left[:, 1], right[::-1, 1]])
    ax.fill(poly_x, poly_y, color=TUNNEL_FILL, zorder=1)

    # Boundary lines
    ax.plot(left[:, 0], left[:, 1], color=TUNNEL_EDGE, linewidth=1.5, zorder=2)
    ax.plot(right[:, 0], right[:, 1], color=TUNNEL_EDGE, linewidth=1.5, zorder=2)

    # Start and end markers
    cl = tunnel_geom["centerline"]
    ax.plot(cl[0, 0], cl[0, 1], "o", color="#4CAF50", markersize=8, zorder=5)
    ax.plot(cl[-1, 0], cl[-1, 1], "o", color="#F44336", markersize=8, zorder=5)

    # Label
    ax.text(0.02, 0.92, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=label_color,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Clean axes
    pad = 0.015
    x_min = min(left[:, 0].min(), right[:, 0].min()) - pad
    x_max = max(left[:, 0].max(), right[:, 0].max()) + pad
    y_min = min(left[:, 1].min(), right[:, 1].min()) - pad
    y_max = max(left[:, 1].max(), right[:, 1].max()) + pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_comparison_video(
    human_traj, human_timestamps,
    model_traj, model_interval,
    tunnel_geom, output_path,
    fps=30, speed_factor=1.0, title=None,
):
    """Render a side-by-side (stacked) comparison video.

    Args:
        human_traj: [[x,y], ...] in metres
        human_timestamps: [ms, ...] timestamps
        model_traj: [[x,y], ...] in metres
        model_interval: float, seconds per model step (typically 0.05)
        tunnel_geom: dict from build_tunnel_geometry
        output_path: path to output MP4
        fps: video frame rate
        speed_factor: playback speed multiplier (1.0 = real-time)
        title: optional title string
    """
    # Compute time arrays in seconds
    ts_human = np.array(human_timestamps, dtype=float)
    ts_human = (ts_human - ts_human[0]) / 1000.0  # ms -> s from start
    dur_human = ts_human[-1]

    n_model = len(model_traj)
    ts_model = np.arange(n_model) * model_interval
    dur_model = ts_model[-1]

    total_duration = max(dur_human, dur_model) / speed_factor
    n_frames = max(1, int(total_duration * fps))

    # Interpolate both to the video timeline
    hx, hy, t_frames = _interpolate_trajectory(
        human_traj, ts_human / speed_factor, fps, total_duration)
    mx, my, _ = _interpolate_trajectory(
        model_traj, ts_model / speed_factor, fps, total_duration)

    # Create figure: 1920x1080
    fig, (ax_h, ax_m) = plt.subplots(2, 1, figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)

    if title:
        fig.suptitle(title, fontsize=22, fontweight="bold", y=0.97)

    # Draw static tunnel geometry on both axes
    _draw_tunnel(ax_h, tunnel_geom, "Human", HUMAN_COLOR)
    _draw_tunnel(ax_m, tunnel_geom, "Model", MODEL_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.94] if title else [0, 0, 1, 1])

    # Dynamic elements
    trail_h, = ax_h.plot([], [], color=HUMAN_COLOR, linewidth=2.0,
                         alpha=0.8, zorder=10, solid_capstyle="round")
    trail_m, = ax_m.plot([], [], color=MODEL_COLOR, linewidth=2.0,
                         alpha=0.8, zorder=10, solid_capstyle="round")

    cursor_h = Circle((hx[0], hy[0]), CURSOR_RADIUS_M,
                       facecolor=HUMAN_COLOR, edgecolor="white",
                       linewidth=1.5, zorder=15)
    cursor_m = Circle((mx[0], my[0]), CURSOR_RADIUS_M,
                       facecolor=MODEL_COLOR, edgecolor="white",
                       linewidth=1.5, zorder=15)
    ax_h.add_patch(cursor_h)
    ax_m.add_patch(cursor_m)

    # Time text
    time_text = ax_h.text(0.98, 0.92, "", transform=ax_h.transAxes,
                          fontsize=14, ha="right", va="top",
                          fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", alpha=0.8))

    def init():
        trail_h.set_data([], [])
        trail_m.set_data([], [])
        return trail_h, trail_m, cursor_h, cursor_m, time_text

    def update(frame):
        # Trails (show full path up to current frame)
        trail_h.set_data(hx[:frame + 1], hy[:frame + 1])
        trail_m.set_data(mx[:frame + 1], my[:frame + 1])

        # Cursors
        cursor_h.set_center((hx[frame], hy[frame]))
        cursor_m.set_center((mx[frame], my[frame]))

        # Time
        t = t_frames[frame] * speed_factor
        time_text.set_text(f"{t:.2f} s")

        return trail_h, trail_m, cursor_h, cursor_m, time_text

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1000 / fps, blit=True)

    writer = FFMpegWriter(fps=fps, bitrate=6000,
                          extra_args=["-vcodec", "libx264",
                                      "-pix_fmt", "yuv420p"])

    print(f"  Rendering {n_frames} frames at {fps} FPS ({total_duration:.1f}s) -> {output_path}")
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Done: {output_path}")
