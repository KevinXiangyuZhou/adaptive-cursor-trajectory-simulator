"""
Generate comparison videos for UIST 2026 video submission.

Usage:
    python -m eval.video.make_videos                    # generate all videos
    python -m eval.video.make_videos --trial 9          # single trial, both participants
    python -m eval.video.make_videos --pid PA --trial 9  # single participant + trial

Output directory: eval/video/output/
"""

import argparse
from pathlib import Path

from .generate_trajectories import (
    load_human_trajectory,
    regenerate_model_trajectory,
    find_best_pair,
    build_tunnel_geometry,
    TRIAL_CONDITIONS,
)
from .render_video import render_comparison_video

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Participant aliases
PARTICIPANTS = {
    "PA": "P69295fa08b858d074685e348",
    "PB": "P6981093d5da8632454f038ab",
}

# Video manifest: conditions that demonstrate key behaviors from the script
VIDEO_SPECS = [
    # Sinusoidal W=50mm — smooth steering, race-tracing
    {"alias": "PA", "trial": 5, "desc": "Sinusoidal W=50mm"},
    {"alias": "PB", "trial": 5, "desc": "Sinusoidal W=50mm"},
    # Corner W=20mm — stop-and-go at sharp turns
    {"alias": "PA", "trial": 7, "desc": "Corner W=20mm"},
    {"alias": "PB", "trial": 7, "desc": "Corner W=20mm"},
    # Corner W=40mm — different strategies (A=wall-follow, B=corner-cut)
    {"alias": "PA", "trial": 9, "desc": "Corner W=40mm"},
    {"alias": "PB", "trial": 9, "desc": "Corner W=40mm"},
    # Gentle sinusoidal W=50mm — aggressive cutting
    {"alias": "PA", "trial": 20, "desc": "Gentle Sin W=50mm"},
    {"alias": "PB", "trial": 20, "desc": "Gentle Sin W=50mm"},
]


def generate_one_video(alias, pid, trial_id, round_num=None, speed_factor=1.0,
                       n_model_runs=5):
    """Generate a single comparison video.

    If round_num is None, automatically selects the best-matching
    (human_round, model_run) pair based on completion time and trajectory RMSE.
    """
    cond = TRIAL_CONDITIONS[trial_id]
    print(f"\n{'='*60}")
    print(f"Generating: {alias} — {cond['label']}")
    print(f"{'='*60}")

    # Build tunnel geometry
    print("  Building tunnel geometry...")
    geom = build_tunnel_geometry(trial_id)

    if round_num is not None:
        # Use specified round
        print(f"  Loading human trajectory (round {round_num})...")
        human = load_human_trajectory(pid, trial_id, round_num)
        print("  Regenerating model trajectory...")
        model = regenerate_model_trajectory(pid, trial_id)
        info_tag = f"r{round_num}"
    else:
        # Find best-matching pair
        print(f"  Finding best (human, model) pair from {n_model_runs} runs...")
        human, model, info = find_best_pair(pid, trial_id, n_model_runs=n_model_runs)
        info_tag = f"r{info['human_round']}_m{info['model_run']}"

    # Render video
    output_path = OUTPUT_DIR / f"{alias}_t{trial_id}_{info_tag}.mp4"
    title = f"Participant {alias[-1]} — {cond['label']}"

    render_comparison_video(
        human_traj=human["trajectory"],
        human_timestamps=human["timestamps"],
        model_traj=model["trajectory"],
        model_interval=model["interval"],
        tunnel_geom=geom,
        output_path=output_path,
        fps=30,
        speed_factor=speed_factor,
        title=title,
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory comparison videos")
    parser.add_argument("--trial", type=int, help="Generate only this trial ID")
    parser.add_argument("--pid", type=str, help="Generate only for this participant alias (PA or PB)")
    parser.add_argument("--round", type=int, default=None,
                        help="Human round to use (default: auto-select best match)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (default: 1.0)")
    parser.add_argument("--all-trials", action="store_true",
                        help="Generate videos for all 25 trial conditions")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_trials:
        specs = []
        for alias in ["PA", "PB"]:
            for tid in TRIAL_CONDITIONS:
                specs.append({"alias": alias, "trial": tid,
                              "round": args.round, "desc": TRIAL_CONDITIONS[tid]["label"]})
    elif args.trial and args.pid:
        specs = [{"alias": args.pid, "trial": args.trial,
                  "round": args.round, "desc": TRIAL_CONDITIONS[args.trial]["label"]}]
    elif args.trial:
        specs = [s for s in VIDEO_SPECS if s["trial"] == args.trial]
        if not specs:
            specs = [{"alias": a, "trial": args.trial,
                      "round": args.round, "desc": TRIAL_CONDITIONS[args.trial]["label"]}
                     for a in ["PA", "PB"]]
    elif args.pid:
        specs = [s for s in VIDEO_SPECS if s["alias"] == args.pid]
    else:
        specs = VIDEO_SPECS

    generated = []
    for spec in specs:
        alias = spec["alias"]
        pid = PARTICIPANTS[alias]
        trial_id = spec["trial"]
        round_num = spec.get("round", args.round)  # None = auto-select best pair
        try:
            path = generate_one_video(alias, pid, trial_id, round_num, args.speed)
            generated.append(path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} videos in {OUTPUT_DIR}/")
    for p in generated:
        print(f"  {p.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
