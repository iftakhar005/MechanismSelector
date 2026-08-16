"""Phase 1 acceptance check: every loader returns correctly shaped arrays.

Prints and records shape, feature count, class count and licence for each
stream. Run from the repo root:  python experiments/verify_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402


def licence_of(name: str) -> str:
    """River exposes no licence field; report the upstream source instead."""
    sources = {
        "elec2": "river.datasets.Elec2 (Harries 1999, NSW electricity market)",
        "insects_abrupt": "river.datasets.Insects(abrupt_balanced) (Souza et al. 2020)",
        "insects_gradual": "river.datasets.Insects(gradual_balanced) (Souza et al. 2020)",
        "insects_incremental": "river.datasets.Insects(incremental_balanced) (Souza et al. 2020)",
        "covtype": "sklearn.datasets.fetch_covtype (Blackard & Dean 1999, UCI)",
    }
    return sources[name]


def main() -> int:
    records = []
    failures = []

    for name in ds.STREAMS:
        print(f"\n=== {name} ===", flush=True)
        try:
            info = ds.describe(name)
        except Exception as exc:  # noqa: BLE001 - report, don't mask
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append((name, repr(exc)))
            continue

        info["source"] = licence_of(name)
        records.append(info)
        print(f"  rows        : {info['n_rows']:,}")
        print(f"  features    : {info['n_features']}")
        print(f"  classes     : {info['n_classes']}")
        print(f"  balance     : {info['class_balance']}")
        print(f"  dtype       : {info['dtype']}   has_nan: {info['has_nan']}")
        print(f"  source      : {info['source']}")

    out = ROOT / "results" / "dataset_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nManifest written to {out}")

    print("\n--- PHASE 1 GATE ---")
    if failures:
        for name, exc in failures:
            print(f"  FAIL {name}: {exc}")
        return 1
    if len(records) != len(ds.STREAMS):
        print(f"  FAIL: {len(records)}/{len(ds.STREAMS)} streams loaded")
        return 1
    print(f"  PASS: all {len(records)} streams loaded with correct shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
