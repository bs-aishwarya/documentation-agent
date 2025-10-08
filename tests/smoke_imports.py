"""
Simple smoke test to validate package imports when running from the repo root.

Run from the project root:
    python tests/smoke_imports.py

This script appends `src` to sys.path to mirror editable installs and prints
the result of importing the config utilities.
"""
import sys
import importlib
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    print(f"Added to sys.path: {src_path}")

    try:
        cfg = importlib.import_module("utils.config")
        print("Imported utils.config successfully")
        # try calling a small helper if present
        if hasattr(cfg, "get_config"):
            conf = cfg.get_config()
            print("get_config() returned:", type(conf))
        else:
            print("utils.config has no get_config() function; module keys:", dir(cfg)[:20])
    except Exception as e:
        print("Import failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
