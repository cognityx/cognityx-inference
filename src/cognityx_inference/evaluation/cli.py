"""Inspect a finite hardware-boundary evaluation plan."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from cognityx_inference.evaluation.boundary import BoundaryConfig, BoundaryPlanner
from cognityx_inference.presentation import render_human


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args(argv)
    with args.config.open("rb") as source:
        document = tomllib.load(source)
    search = document["search"]
    axes = {name: tuple(values) for name, values in (search.get("axes") or {}).items()}
    config = BoundaryConfig(
        axes=axes,
        ordered_axes=tuple(search.get("ordered_axes") or axes),
        monotonic_axes=tuple(search.get("monotonic_axes") or ()),
    )
    trials = BoundaryPlanner(config).trials()
    result = {
        "configured_trial_count": len(trials),
        "load_groups": len({trial.load_identity for trial in trials}),
        "axes": {name: list(values) for name, values in axes.items()},
    }
    if args.human:
        print(render_human(result))
    else:
        print(json.dumps(result, indent=2))
    if not args.plan:
        raise SystemExit(
            "Execution wiring requires an inference workload; use --plan for now."
        )


if __name__ == "__main__":
    main()
