#!/usr/bin/env python3
"""Create the pinned local environment without overwriting existing source changes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check pins and installed dependencies without changing anything")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise SystemExit("Use Python 3.12 or newer.")
    pins = json.loads((ROOT / "configs/tools.lock.json").read_text())
    for name, pin in pins.items():
        if not pin["install"]:
            continue
        path = ROOT / "vendor" / name
        if not path.exists() and not args.verify_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", pin["repository"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", pin["revision"]], check=True)
        if not path.exists():
            raise SystemExit(f"Missing source: {path}")
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if revision != pin["revision"] or dirty:
            raise SystemExit(f"{name} differs from the source lock; inspect it rather than overwriting it.")
        print(f"{name}: {revision}", flush=True)
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists() and not args.verify_only:
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    if not python.exists():
        raise SystemExit("Missing .venv; run without --verify-only first.")
    if not args.verify_only:
        subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements-local.txt"], cwd=ROOT, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    exact_versions = dict(line.strip().split("==", 1) for line in
        (ROOT / "requirements-local.txt").read_text().splitlines()
        if "==" in line and not line.lstrip().startswith("#"))
    verify = """
import importlib, importlib.metadata, json, pathlib, sys
assert sys.version_info >= (3, 12), 'Existing venv needs Python 3.12+'
root = pathlib.Path(sys.argv[1])
for name, expected in json.loads(sys.argv[2]).items():
    actual = importlib.metadata.version(name)
    assert actual == expected, f'{name}: installed {actual}, required {expected}'
for package, directory in [('safetune', 'SafeTune'), ('auditkit', 'AuditKIT')]:
    module = importlib.import_module(package)
    assert pathlib.Path(module.__file__).resolve().is_relative_to((root / 'vendor' / directory).resolve()), f'{package}: imported unexpected source'
import torch
print('Verified exact package pins, editable source paths, and venv Python:', sys.version.split()[0])
print('MPS available:', torch.backends.mps.is_available())
"""
    subprocess.run([str(python), "-c", verify, str(ROOT), json.dumps(exact_versions)], check=True)


if __name__ == "__main__":
    main()
