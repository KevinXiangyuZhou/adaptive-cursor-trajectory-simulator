"""Merge the per-participant eval-main runs of the pooled persona into ONE
eval-main folder with cross-participant aggregate law plots.

Reads eval/eval-main/results/eval-main-pooled10-{p}/ for each pool participant,
concatenates the per-run CSVs (Steering, Fitts, ID4SCS, C2U), and regenerates
the aggregate outputs (steering_law.pdf, fitts_regression*.png/json, condition
summaries) with run_eval's own writers, so each law-plot point is one task
condition averaged across ALL pool participants — the paper-figure methodology.

Output: eval/eval-anchor-drive/results/eval-main-pooled10-ALL/
Usage:  python merge_eval_main.py [--pool p04 p06 p07 p09 p10]
"""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "eval-main"))
import probe_anchor as pa   # noqa: F401 (sys.path)
import run_eval as em

SRC = HERE / "results" / "pooled10"
OUT = HERE / "results" / "pooled10" / "eval-main-pooled10-ALL"

BOOL = {"True": True, "False": False, "": None}


def read_rows(path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v in BOOL:
                    row[k] = BOOL[v]
                    continue
                try:
                    fv = float(v)
                    row[k] = int(fv) if fv == int(fv) and "." not in v and "e" not in v.lower() else fv
                except (ValueError, TypeError):
                    row[k] = v
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="*", default=["p04", "p06", "p07", "p09", "p10"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    em._redirect_output_paths(OUT) if hasattr(em, "_redirect_output_paths") else None
    # run_eval exposes the dirs as module globals; point them at OUT
    em.STEERING_DIR = OUT / "Steering"; em.FITTS_DIR = OUT / "Fitts"
    em.ID4SCS_DIR = OUT / "ID4SCS"; em.C2U_DIR = OUT / "ConstrainedToUnconstrained"

    def gather(rel):
        rows = []
        for p in a.pool:
            rows += read_rows(SRC / f"eval-main-pooled10-{p}" / rel)
        return rows

    st = gather("Steering/steering_results.csv")
    st_sum = gather("Steering/steering_condition_summary.csv")
    em.write_steering_outputs(st, st_sum)
    ft = gather("Fitts/fitts_results.csv")
    ft_sum = gather("Fitts/fitts_condition_summary.csv")
    em.write_fitts_outputs(ft, ft_sum)
    for direction, sub in (("wide_to_narrow", "ID4SCS/wide_to_narrow"),
                            ("narrow_to_wide", "ID4SCS/narrow_to_wide")):
        rows = gather(f"{sub}/{direction}_results.csv")
        sums = gather(f"{sub}/{direction}_condition_summary.csv")
        if rows:
            em.write_id4scs_outputs(rows, sums, direction, OUT / sub, direction)
    c2u = gather("ConstrainedToUnconstrained/c2u_results.csv")
    c2u_sum = gather("ConstrainedToUnconstrained/c2u_condition_summary.csv")
    if c2u:
        (OUT / "ConstrainedToUnconstrained").mkdir(parents=True, exist_ok=True)
        em._write_dict_rows_csv(c2u, OUT / "ConstrainedToUnconstrained" / "c2u_results.csv")
        em._write_dict_rows_csv(c2u_sum, OUT / "ConstrainedToUnconstrained" / "c2u_condition_summary.csv")
    print(f"merged {len(st)} steering rows, {len(ft)} fitts rows from {len(a.pool)} participants -> {OUT}")


if __name__ == "__main__":
    main()
