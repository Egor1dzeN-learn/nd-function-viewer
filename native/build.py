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


def compile_dll(output: Path, sources: list, include_dirs: list, options: tuple = ()):
    """Compiles the sources into a DLL with g++; exits with an error message on failure."""
    compiler = shutil.which("g++")
    if compiler is None:
        sys.exit("g++ not found in PATH")
    command = [
        compiler, "-O2", "-std=c++17", "-shared", "-static", *options,
        *(f"-I{directory}" for directory in include_dirs), *map(str, sources),
    ]
    print(f"Compiling {len(sources)} files...")
    # The MinGW linker cannot write to non-ASCII paths, so link in a temporary directory.
    with tempfile.TemporaryDirectory() as tmp:
        built = Path(tmp) / output.name
        result = subprocess.run([*command, "-o", str(built)])
        if result.returncode != 0:
            sys.exit("Compilation failed")
        shutil.move(built, output)
    print(f"Done: {output}")


def main():
    global_opt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GLOBAL_OPT
    gcgen = global_opt / "3rd_party" / "GCGen"
    if not gcgen.is_dir():
        sys.exit(f"GCGen not found: {gcgen}\nPass the path to global-opt: python native/build.py <path>")

    sources =[HERE / "testfuncs_capi.cpp", global_opt / "src" / "test_functions" / "classic_functions.cpp"]
    sources += sorted((gcgen / "src").glob("*.cpp"))
    for family in ("GKLS", "Grishagin", "Hill", "Shekel"):
        sources += sorted((gcgen / "sample_src" / family).glob("*.cpp"))

    compile_dll(OUTPUT, sources, [gcgen / "src", gcgen / "sample_src", global_opt / "include"], ("-w", "-D_USE_MATH_DEFINES"))


if __name__ == "__main__":
    main()
