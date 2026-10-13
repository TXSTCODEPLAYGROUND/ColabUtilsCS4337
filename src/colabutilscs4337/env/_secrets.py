"""Load API keys and tokens into ``os.environ`` without pasting them into notebooks.

Sources, in order:

1. Already set in ``os.environ``.
2. Colab Secrets (the key icon in Colab's left sidebar).
3. A ``.env`` file: ``env_file=`` if given, else ``./.env``, else
   ``MyDrive/.colabutilscs4337/.env`` when Drive is mounted.
4. A hidden ``getpass`` prompt, only when ``prompt=True``.

Values are never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import _colab
from .._colab import SecretAccessError

DRIVE_ENV_FILE = Path(".colabutilscs4337") / ".env"

_HOW_TO_ADD = (
    "In Colab: click the key icon in the left sidebar > 'Add new secret', enter the name "
    "and value, and switch on 'Notebook access'. Elsewhere: set the environment variable "
    "or add NAME=value to a .env file."
)


class MissingSecretError(KeyError):
    def __str__(self) -> str:
        return f"Secret(s) not found: {', '.join(self.args[0])}. {_HOW_TO_ADD}"


def read_env_file(path: str | os.PathLike) -> dict[str, str]:
    """Parse a ``.env`` file (``NAME=value`` lines; ``#`` comments, quotes, ``export`` ok)."""
    values: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    return values


def _env_files(env_file: str | os.PathLike | None) -> list[Path]:
    if env_file is not None:
        return [Path(env_file)]
    candidates = [Path.cwd() / ".env"]
    if _colab.is_drive_mounted() or not _colab.is_colab():
        candidates.append(_colab.drive_root() / DRIVE_ENV_FILE)
    return [p for p in candidates if p.is_file()]


def _lookup(name: str, files: list[Path], cache: dict[Path, dict[str, str]]) -> tuple[str, str]:
    """Return ``(value, source)``; value is ``""`` when not found."""
    if os.environ.get(name):
        return os.environ[name], "environment"
    value = _colab.colab_secret(name)
    if value:
        return value, "Colab Secrets"
    for path in files:
        if path not in cache:
            cache[path] = read_env_file(path)
        if cache[path].get(name):
            return cache[path][name], str(path)
    return "", ""


def get_secret(
    name: str,
    default: str | None = None,
    *,
    prompt: bool = False,
    env_file: str | os.PathLike | None = None,
) -> str | None:
    """Return one secret's value (does not modify ``os.environ``).

    Example::

        token = cu.env.get_secret("HF_TOKEN")
        key = cu.env.get_secret("OPENAI_API_KEY", prompt=True)  # asks if not found
    """
    value, _ = _lookup(name, _env_files(env_file), {})
    if not value and prompt:
        import getpass

        value = getpass.getpass(f"{name}: ").strip()
    return value or default


def load_secrets(
    *names: str,
    env_file: str | os.PathLike | None = None,
    prompt: bool = False,
    required: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, str | None]:
    """Put secrets into ``os.environ`` so libraries (Hugging Face, W&B, Kaggle) find them.

    With no ``names``, every entry of the ``.env`` file is loaded. Returns
    ``{name: source}`` where ``source`` is where it was found (``None`` if missing).
    Prints a short found/missing summary, never the values.

    Examples::

        cu.env.load_secrets("HF_TOKEN", "WANDB_API_KEY")
        cu.env.load_secrets("KAGGLE_USERNAME", "KAGGLE_KEY", required=True)
        cu.env.load_secrets(env_file="/content/drive/MyDrive/keys.env")   # load a whole file
        cu.env.load_secrets("OPENAI_API_KEY", prompt=True)                 # ask if missing
    """
    files = _env_files(env_file)
    cache: dict[Path, dict[str, str]] = {}
    if not names:
        for path in files:
            cache[path] = read_env_file(path)
        names = tuple(dict.fromkeys(k for path in files for k in cache[path]))

    result: dict[str, str | None] = {}
    no_access: list[str] = []
    for name in names:
        if os.environ.get(name) and not overwrite:
            result[name] = "environment"
            continue
        # overwrite=True: look past the current environment value, keep it as a fallback.
        previous = os.environ.pop(name, None) if overwrite else None
        try:
            value, source = _lookup(name, files, cache)
        except SecretAccessError:
            no_access.append(name)
            value, source = "", ""
        if not value and previous:
            value, source = previous, "environment"
        if not value and prompt:
            import getpass

            value = getpass.getpass(f"{name}: ").strip()
            source = "prompt" if value else ""
        if value:
            os.environ[name] = value
        result[name] = source or None

    if verbose:
        for name, source in result.items():
            print(f"  {'ok' if source else '--'}  {name}" + (f"  ({source})" if source else ""))
        for name in no_access:
            print(
                f"  Colab has '{name}' but this notebook can't read it: turn on "
                "'Notebook access' for it in the Secrets panel (key icon)."
            )
    missing = [n for n, s in result.items() if s is None]
    if missing and required:
        raise MissingSecretError(missing)
    if missing and verbose:
        print(f"  Missing: {', '.join(missing)}. {_HOW_TO_ADD}")
    return result
