#!/usr/bin/env python3
"""Fail if installing a requirement would change any already-installed package.

Simulates what a student gets from a plain ``pip install`` (no constraints), using
``pip install --dry-run --report``. Run it inside the Colab runtime image:

  python scripts/check_no_reinstall.py ".[all]"
  python scripts/check_no_reinstall.py colabutilscs4337[all] --upgrade
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from colabutilscs4337._install import (  # noqa: E402
    InstallError,
    dry_run,
    requirement_name,
    snapshot,
)

SELF = "colabutilscs4337"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requirements", nargs="*", default=[".[all]"])
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args()

    installed = snapshot()
    try:
        plan = dry_run(args.requirements, installed, constraints=None, upgrade=args.upgrade)
    except InstallError as exc:
        print(exc)
        return 2

    requested = {requirement_name(r) for r in args.requirements if not r.startswith((".", "/"))} | {
        SELF
    }
    changed = {n: v for n, v in plan.changed.items() if n not in requested}

    for name, version in sorted(plan.added.items()):
        print(f"  + {name} {version}")
    for name, (old, new) in sorted(plan.changed.items()):
        marker = "~" if name in requested else "!"
        print(f"  {marker} {name} {old} -> {new}")

    if changed:
        print(
            f"FAIL: installing {args.requirements} would change {len(changed)} "
            "preinstalled package(s) (marked !)."
        )
        return 1
    print(f"OK: {len(plan.added)} package(s) added, no preinstalled package changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
