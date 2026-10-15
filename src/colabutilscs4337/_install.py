"""Locked installer: add packages without changing anything already installed.

``install("tools", "cam")`` snapshots the environment, pins every installed
distribution with a constraints file, dry-runs pip to see exactly what would happen,
and only then installs. If pip would upgrade, downgrade, or reinstall anything that
is already present (torch, fastai, numpy, ...), it refuses unless
``allow_changes=True``. When ``uv`` is available (Colab preinstalls it) the dry run
and the install use uv, which is several times faster; pip is the fallback.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ._compat import COLAB_PROVIDED

PACKAGE = "colabutilscs4337"

EXTRAS: dict[str, list[str]] = {
    "tools": ["torchinfo>=1.8,<2", "netron>=7.0", "onnx>=1.18,<1.23", "onnxscript>=0.3,<0.8"],
    "cam": ["grad-cam>=1.5"],
    "fastai": ["fastai>=2.7"],
}
EXTRAS["all"] = EXTRAS["tools"] + EXTRAS["cam"]

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_UV_LINE_RE = re.compile(r"^\s*([+-])\s+([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)\s*$")
BACKENDS = ("auto", "uv", "pip")


class InstallError(RuntimeError):
    pass


class UnresolvableError(InstallError):
    """The request cannot be satisfied (with the current pins); retrying will not help."""


def normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(req: str) -> str:
    match = _NAME_RE.match(req)
    if not match:
        raise ValueError(f"Cannot parse a package name from requirement {req!r}")
    return normalize(match.group(1))


def resolve_targets(targets: Iterable[str]) -> list[str]:
    """Expand extra names (``"tools"``) into requirement strings; pass others through."""
    reqs: list[str] = []
    for target in targets:
        reqs.extend(EXTRAS.get(target, [target]))
    return list(dict.fromkeys(reqs))


def _pip(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "pip", *args, "--disable-pip-version-check"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise InstallError(
            f"Command failed: {' '.join(cmd)}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    return proc


def _uv_path() -> str | None:
    return shutil.which("uv")


def _uv(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ``uv pip <args>`` against this kernel's interpreter (not a virtualenv)."""
    cmd = [_uv_path() or "uv", "pip", *args, "--python", sys.executable]
    env = {**os.environ, "NO_COLOR": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and proc.returncode != 0:
        raise InstallError(
            f"Command failed: {' '.join(cmd)}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    return proc


def choose_backend(backend: str = "auto") -> str:
    """``"uv"`` if requested or available (``"auto"``), else ``"pip"``."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "pip":
        return "pip"
    if _uv_path():
        return "uv"
    if backend == "uv":
        raise InstallError("uv is not installed; use backend='pip' or `pip install uv`.")
    return "pip"


def snapshot() -> dict[str, str]:
    """Installed distributions as ``{normalized_name: version}``.

    Uses ``pip list --format=freeze`` because ``pip freeze`` prints Colab's torch as a
    ``torch @ https://...`` URL, which is not usable as a constraint.
    """
    out = _pip("list", "--format=freeze").stdout
    installed: dict[str, str] = {}
    for line in out.splitlines():
        name, sep, version = line.strip().partition("==")
        if sep and name:
            installed[normalize(name)] = version
    return installed


def constraints_text(installed: dict[str, str], exclude: Iterable[str] = ()) -> str:
    skip = {normalize(n) for n in exclude} | {PACKAGE}
    return "".join(f"{n}=={v}\n" for n, v in sorted(installed.items()) if n not in skip)


@dataclass
class Plan:
    added: dict[str, str] = field(default_factory=dict)
    changed: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def is_noop(self) -> bool:
        return not self.added and not self.changed


def parse_report(report: dict, installed: dict[str, str]) -> Plan:
    """Turn a ``pip install --dry-run --report`` JSON report into added/changed packages."""
    plan = Plan()
    for item in report.get("install", []):
        meta = item.get("metadata", {})
        name, version = normalize(meta.get("name", "")), meta.get("version", "")
        if not name:
            continue
        if name in installed:
            plan.changed[name] = (installed[name], version)
        else:
            plan.added[name] = version
    return plan


def parse_uv_dry_run(output: str, installed: dict[str, str]) -> Plan:
    """Turn ``uv pip install --dry-run`` output (`` + name==1.0`` lines) into a :class:`Plan`."""
    plan = Plan()
    for line in output.splitlines():
        match = _UV_LINE_RE.match(line)
        if not match or match.group(1) != "+":
            continue
        name, version = normalize(match.group(2)), match.group(3)
        if name in installed:
            plan.changed[name] = (installed[name], version)
        else:
            plan.added[name] = version
    return plan


def _install_args(
    reqs: list[str], constraints: Path | None, upgrade: bool, backend: str = "pip"
) -> list[str]:
    args = ["install"]
    if constraints is not None:
        args += ["-c", str(constraints)]
    if upgrade and backend == "uv":
        # uv's --upgrade would upgrade every dependency too; limit it to the request.
        for name in dict.fromkeys(requirement_name(r) for r in reqs):
            args += ["--upgrade-package", name]
    elif upgrade:
        args.append("--upgrade")
    return [*args, *reqs]


_UNRESOLVABLE_HINT = (
    "\nThese packages cannot be installed without changing packages that are "
    "already installed. Nothing was changed. Pass allow_changes=True to override "
    "(may break Colab's preinstalled torch/fastai)."
)


def dry_run(
    reqs: list[str],
    installed: dict[str, str],
    constraints: Path | None,
    upgrade: bool = False,
    backend: str = "pip",
) -> Plan:
    hint = _UNRESOLVABLE_HINT if constraints is not None else ""
    if backend == "uv":
        proc = _uv(*_install_args(reqs, constraints, upgrade, "uv"), "--dry-run", check=False)
        output = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            error = UnresolvableError if "No solution found" in output else InstallError
            raise error(f"uv could not resolve {reqs}:\n{output[-4000:]}{hint}")
        return parse_uv_dry_run(output, installed)
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        args = _install_args(reqs, constraints, upgrade)
        proc = _pip(*args, "--dry-run", "--quiet", "--report", str(report_path), check=False)
        if proc.returncode != 0:
            error = UnresolvableError if "ResolutionImpossible" in proc.stderr else InstallError
            raise error(f"pip could not resolve {reqs}:\n{proc.stderr[-4000:]}{hint}")
        return parse_report(json.loads(report_path.read_text()), installed)


def _pip_check_problems() -> set[str]:
    proc = _pip("check", check=False)
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


@dataclass
class InstallResult:
    requested: list[str]
    added: dict[str, str]
    changed: dict[str, tuple[str, str]]
    new_problems: list[str]
    restart_needed: bool
    backend: str = "pip"

    def __str__(self) -> str:
        lines = []
        if self.added:
            lines.append("Added: " + ", ".join(f"{n} {v}" for n, v in sorted(self.added.items())))
        if self.changed:
            lines.append(
                "Changed: "
                + ", ".join(f"{n} {a} -> {b}" for n, (a, b) in sorted(self.changed.items()))
            )
        if not lines:
            lines.append("Already installed: " + ", ".join(self.requested))
        for problem in self.new_problems:
            lines.append(f"pip check: {problem}")
        if self.restart_needed:
            lines.append("Restart the session (Runtime > Restart session) to use the new versions.")
        return "\n".join(lines)


def _imported_dists(names: Iterable[str]) -> bool:
    loaded = {normalize(m.split(".")[0]) for m in sys.modules}
    return any(normalize(n) in loaded for n in names)


def install(
    *targets: str,
    allow_changes: bool = False,
    upgrade: bool = False,
    dry: bool = False,
    verbose: bool = True,
    backend: str = "auto",
) -> InstallResult:
    """Install extras (``"cam"``, ``"fastai"``, ``"all"``) or any requirement strings.

    Every already-installed package is pinned to its current version, so this only
    ever adds packages. If that is impossible, nothing is installed and
    :class:`InstallError` explains why. ``allow_changes=True`` lifts the pins.
    ``dry=True`` reports what would happen without installing.

    ``backend="auto"`` uses ``uv`` when it is installed (it is in Colab) and ``pip``
    otherwise; if uv fails, pip is tried. ``upgrade=True`` upgrades only the
    requested packages, never their dependencies.

    Examples
    --------
    >>> cu.install("cam")                    # Grad-CAM++ and friends
    >>> cu.install("torchmetrics", dry=True)         # preview: what would be added?
    >>> cu.install("torchmetrics", upgrade=True)     # newer torchmetrics only
    >>> cu.install("torchmetrics", backend="pip")    # force pip
    """
    if not targets:
        targets = ("all",)
    reqs = resolve_targets(targets)
    names = [requirement_name(r) for r in reqs]
    chosen = choose_backend(backend)

    before = snapshot()
    problems_before = _pip_check_problems()

    with tempfile.TemporaryDirectory() as tmp:
        constraints: Path | None = None
        if not allow_changes:
            constraints = Path(tmp) / "colab-constraints.txt"
            constraints.write_text(constraints_text(before, exclude=names))

        try:
            plan = dry_run(reqs, before, constraints, upgrade=upgrade, backend=chosen)
        except InstallError as err:
            if chosen != "uv" or backend == "uv" or isinstance(err, UnresolvableError):
                raise
            warnings.warn(f"uv failed, retrying with pip:\n{err}", stacklevel=2)
            chosen = "pip"
            plan = dry_run(reqs, before, constraints, upgrade=upgrade, backend=chosen)
        protected = {normalize(d) for d in COLAB_PROVIDED}
        blocked = {n: v for n, v in plan.changed.items() if n not in names or n in protected}
        if blocked and not allow_changes:
            changes = ", ".join(f"{n} {a} -> {b}" for n, (a, b) in blocked.items())
            raise InstallError(
                f"Refusing to install {reqs}: it would change already-installed packages "
                f"({changes}). Pass allow_changes=True to override."
            )

        if dry or plan.is_noop:
            result = InstallResult(reqs, plan.added, plan.changed, [], False, chosen)
        else:
            if chosen == "uv":
                _uv(*_install_args(reqs, constraints, upgrade, "uv"))
            else:
                _pip(*_install_args(reqs, constraints, upgrade))
            after = snapshot()
            added = {n: v for n, v in after.items() if n not in before}
            changed = {
                n: (before[n], v) for n, v in after.items() if n in before and before[n] != v
            }
            new_problems = sorted(_pip_check_problems() - problems_before)
            result = InstallResult(
                reqs, added, changed, new_problems, _imported_dists(changed), chosen
            )

    if verbose:
        prefix = f"[dry run, {chosen}] " if dry else ""
        print(prefix + str(result))
    return result
