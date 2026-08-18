# Great Lakes Setup & Job Submission

## 1. Clone repo on login node

```bash
cd /home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation
```

## 2. Run setup

```bash
bash setup.sh
```

This creates `venv/`, `logs/`, result directories, and installs dependencies.

## 3. Submit fitting jobs

```bash
sbatch fit_all_participants.sh
```

- Array job: 1 per participant (14 total)
- 12 CPUs, 1g/CPU, 13h limit each
- Outputs: `eval/model_fitting/results/{PID}_gam_config_s42.json` + `.pkl`
- Monitor: `squeue -u xiangyz`

## 4. Submit baseline fitting jobs

```bash
sbatch fit_baseline_all_participants.sh
```

- Array job: 1 per participant (14 total)
- 12 CPUs, 0.5g/CPU, 13h limit each
- Outputs: `eval/baseline_fitting/results/{PID}_baseline_config_s42.json` + `_baseline_fit_s42.json`

## 5. Submit eval jobs (after both fitting jobs complete)

```bash
sbatch eval_all_participants.sh
```

- Array job: 1 per participant (14 total)
- 12 CPUs, 2g/CPU, 4h limit each
- Flags: `--per-participant --include-baseline --rounds 3`
- Outputs: `eval/experiment-main/results/participant_{PID}/`

## 6. Check results

```bash
# Job status
squeue -u xiangyz

# Logs
ls logs/fit_*.out logs/eval_*.out

# Fitting results
ls eval/model_fitting/results/*.json

# Eval results
cat eval/experiment-main/results/aggregate_metrics.json
```

## Notes

- Eval depends on fitting outputs — submit eval only after all fitting jobs finish.
- To rerun a single participant: `sbatch --array=N fit_all_participants.sh` where N is the line number in `participants.txt`.
- To cancel: `scancel <job_id>` or `scancel -u xiangyz`.
