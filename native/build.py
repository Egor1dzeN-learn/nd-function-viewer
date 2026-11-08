"""Builds testfuncs.dll from the global-opt sources (GCGen + classic functions).

Usage:  python native/build.py [path to global-opt]
By default global-opt is looked up next to this project (../global-opt).
Requires g++ (e.g. MinGW bundled with CLion) in PATH.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GLOBAL_OPT = HERE.parent.parent / "global-opt"
OUTPUT = HERE / "testfuncs.dll"


def main():
    global_opt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GLOBAL_OPT
    gcgen = global_opt / "3rd_party" / "GCGen"
    if not gcgen.is_dir():
        sys.exit(f"GCGen not found: {gcgen}\nPass the path to global-opt: python native/build.py <path>")
    compiler = shutil.which("g++")
    if compiler is None:
        sys.exit("g++ not found in PATH")

    sources = [HERE / "testfuncs_capi.cpp", global_opt / "src" / "test_functions" / "classic_functions.cpp"]
    sources += sorted((gcgen / "src").glob("*.cpp"))
    for family in ("GKLS", "Grishagin", "Hill", "Shekel"):
        sources += sorted((gcgen / "sample_src" / family).glob("*.cpp"))

    command = [
        compiler, "-O2", "-std=c++17", "-shared", "-static", "-w", "-D_USE_MATH_DEFINES",
        f"-I{gcgen / 'src'}", f"-I{gcgen / 'sample_src'}", f"-I{global_opt / 'include'}",
        *map(str, sources),
    ]
    print(f"Compiling {len(sources)} files...")
    # The MinGW linker cannot write to non-ASCII paths, so link in a temporary directory.
    with tempfile.TemporaryDirectory() as tmp:
        built = Path(tmp) / OUTPUT.name
        subprocess.run([*command, "-o", str(built)], check=True)
        shutil.move(built, OUTPUT)
    print(f"Done: {OUTPUT}")


if __name__ == "__main__":
    main()
