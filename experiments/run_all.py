"""Phase 6: run the experimental grid.

    5 datasets x 4 models x 5 policies x 5 seeds = 500 runs

Every run is written to `results/grid/grid.csv` the moment it finishes, so the
grid can be stopped and restarted at any point; completed runs are skipped.
Per-adaptation event logs go to `results/grid/events/`, failures to
`results/grid/grid_failures.csv`, and progress to `results/grid/run.log`.

Runs are ordered seed-first, so once seed 0 finishes every (dataset, model) cell
has one full replicate of all five policies. Check the selector against
FixedSchedule at that point with `experiments/margin_report.py`.

Examples:
    python experiments/run_all.py                      # the full grid
    python experiments/run_all.py --seeds 0            # seed 0 only
    python experiments/run_all.py --dry-run            # show the plan
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import GRID_MODELS  # noqa: E402
from policies import POLICY_KEYS  # noqa: E402
from runner import RunConfig, completed_keys, grid_plan, run_grid  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--datasets", nargs="+", default=list(ds.STREAMS), choices=ds.STREAMS)
    parser.add_argument("--models", nargs="+", default=list(GRID_MODELS), choices=GRID_MODELS)
    parser.add_argument("--policies", nargs="+", default=list(POLICY_KEYS), choices=POLICY_KEYS)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "grid")
    parser.add_argument("--no-energy", action="store_true", help="skip CodeCarbon (secondary metric)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = grid_plan(args.seeds, args.datasets, args.models, args.policies)
    done = completed_keys(args.out / "grid.csv")
    todo = [k for k in plan if k not in done]

    if args.dry_run:
        print(f"{len(plan)} runs planned, {len(plan) - len(todo)} complete, {len(todo)} to run")
        for key in todo[:10]:
            print("  ", key)
        if len(todo) > 10:
            print(f"   ... and {len(todo) - 10} more")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "run.log"

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    config = RunConfig(track_energy=not args.no_energy)
    started = time.perf_counter()
    result = run_grid(args.out, seeds=args.seeds, datasets=args.datasets, models=args.models,
                      policies=args.policies, config=config, log=log)
    log(f"done in {(time.perf_counter() - started) / 60:.1f} min: {result}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
