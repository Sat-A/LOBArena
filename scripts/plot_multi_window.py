#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LOBArena.evaluate.pipeline import generate_multi_window_plots_from_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multi-window plots from an existing summary JSON.")
    p.add_argument("--summary_path", required=True, help="Path to multi_window_summary.json")
    p.add_argument("--plot_dir", default="", help="Optional output plot directory (defaults to <summary_dir>/plots)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = generate_multi_window_plots_from_summary(summary_path=args.summary_path, out_dir=args.plot_dir)
    print(f"[LOBArena] Multi-window plots written: {out['plots_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
