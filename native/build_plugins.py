"""Builds plugins.dll from every .cpp file in plugins/ (see native/nd_function.hpp).

Usage:  python native/build_plugins.py
Requires g++ in PATH. global-opt is not needed.
"""
import sys
from pathlib import Path

from build import compile_dll

HERE = Path(__file__).resolve().parent
PLUGINS = HERE.parent / "plugins"
OUTPUT = HERE / "plugins.dll"


def main():
    sources = sorted(PLUGINS.rglob("*.cpp"))
    if not sources:
        sys.exit(f"No .cpp files found in {PLUGINS}")
    compile_dll(OUTPUT, [HERE / "plugin_capi.cpp", *sources], [HERE, PLUGINS])


if __name__ == "__main__":
    main()
