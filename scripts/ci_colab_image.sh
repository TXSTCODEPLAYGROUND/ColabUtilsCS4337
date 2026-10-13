#!/usr/bin/env bash
# Run inside Google's Colab runtime image (see .github/workflows/ci.yml):
#   docker run --rm --entrypoint bash -v "$PWD:/src" -w /src \
#     us-docker.pkg.dev/colab-images/public/cpu-runtime:latest scripts/ci_colab_image.sh
set -euxo pipefail

export MPLBACKEND=Agg
git config --global --add safe.directory "$PWD" || true
python3 --version

python3 -m pip list --format=freeze > /tmp/before.txt

# 1. A plain student install must not change any preinstalled package.
python3 scripts/check_no_reinstall.py ".[all]"
python3 -m pip install --disable-pip-version-check ".[all]"
python3 -m pip list --format=freeze > /tmp/after.txt

# 2. Verify after the fact: only additions (and colabutilscs4337 itself).
python3 - <<'EOF'
def load(path):
    return dict(line.strip().split("==", 1) for line in open(path) if "==" in line)

before, after = load("/tmp/before.txt"), load("/tmp/after.txt")
changed = {
    k: (before[k], v)
    for k, v in after.items()
    if k in before and before[k] != v and k.lower() != "colabutilscs4337"
}
assert not changed, f"Preinstalled packages changed: {changed}"
print("Added:", sorted(set(after) - set(before)))
EOF

python3 -c "import pytest" 2>/dev/null || python3 -m pip install -q pytest
python3 -c "import colabutilscs4337 as cu; cu.env.check()"

# Test the installed package, not the source tree.
cd /tmp
python3 -m pytest -q -p no:cacheprovider "$OLDPWD/tests"
