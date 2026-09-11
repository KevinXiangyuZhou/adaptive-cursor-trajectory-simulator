# pooled8-991900f

`default.json` is the ONE pooled persona of run
`mpcc-full-pooled8-s42-20260910-1540-991900f` (code 991900f, seed 42):
mpcc / full model, fitted jointly on the 8-participant gaze cohort
(p01 p02 p03 p04 p06 p07 p08 p10; participants_10p.txt) with the held-out
split protocol. CMA-ES searched jerk, contour, constraint, goal, D0; the
speed model is `gam_traversal` with no path, i.e. the shipped pooled
artifact `hcs_package/models/gam_traversal_10p.pkl`.

Staged with `cluster/stage_persona.py` (add_noise true, replan latency CV
0.89) from
`<run>/fit/stages/pooled8/pooled8_anchor_config_s42.json`; identical to the
run's own `personas/P*.json` apart from `_fit`. Every participant of
eval-14p is evaluated with this file (run_eval.py resolves `default.json`
when no `{pid}.json` exists), so nothing in it is fitted to the 14
participants.
