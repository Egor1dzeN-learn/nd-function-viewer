# nd-function-viewer

Plots functions of one to twelve variables:

- `f(x)` — a curve; the color of the line and the fill depends on the function value;
- `f(x, y)` — a 3D surface that can be rotated with the mouse. Lowlands are blue ("sea"), peaks are red ("mountains"); the scale on the right maps colors to values;
- `f(x, y, z)`, `f(x, y, z, w)` — two views of the slices (see below);
- `f(x1, ..., xN)`, `N > 4` — HyperSlice, a matrix of slices through a movable point (see below).

Function sources (tabs on the left):

- **Formula** — an arbitrary expression in `x`, `y`, `z`, `w` or `x1` … `x12` (`x1` is `x`, `x2` is `y`, ...);
  the last variable used sets the dimension;
- **Test problems** — functions from the [global-opt](https://github.com/Egor1dzeN/global-opt) project
  (this tab needs the native library, see [below](#building-the-test-function-library)):
  GCGen families — Hill (1D, 1000 problems), Shekel (1D, 1000), Hansen (1D, 20), Grishagin (2D, 100),
  GKLS (2–12D, 100 problems, Simple/Hard classes, ND/D/D2 types) — and classic functions
  (Rastrigin, Ackley, Eggholder, etc., up to 12D). The known global minimum is marked on the plot;
- **Plugins** — your own functions in Python or C++, see [below](#adding-your-own-function).

## Functions of three and four variables

Two variables go on the surface axes (the *Surface axes* list), the others are fixed:

- **Animated surface** — the slice of the function as a surface; each extra variable has a slider and a
  play button that sweeps it back and forth. In 4D one variable is animated while the other stays on its slider.
- **Slice grid** (dimensional stacking, LeBlanc et al., 1990) — all slices at once as heatmaps: in 4D the columns
  are `z` and the rows are `w`, in 3D the tiles go in reading order. Hovering over a tile shows the coordinates and
  the value of `f`; the wheel zooms.

All slices share one color and height scale, so they can be compared. If the minimum is known, the nearest slice is
moved exactly onto it (so the axis steps may be slightly uneven) and the view opens on that slice; the marker is
shown only on it. The whole grid is computed in advance: the surface resolution is reduced when the total exceeds
4 million points, as noted in the status bar.

## Functions of more than four variables

The HyperSlice view (van Wijk, van Liere, 1993) shows slices through one common *focus point*, marked by crosshairs:

- below the diagonal, the panel in column `i`, row `j` is a heatmap of `f` over `xi`, `xj`, with the other variables at the focus point;
- on the diagonal, the panel is the graph of `f` along one variable.

Dragging inside a panel moves the focus point and recomputes all panels; its coordinates and `f` are listed in the top
right corner, and hovering shows the value under the cursor. The view opens on the known minimum if there is one,
otherwise on the center of the domain. A filled dot means the minimum lies on that slice; a ring is only its
projection onto the two axes of the panel.

Only visible slices are computed, so the number of points grows as `N²`. Keep in mind:

- each panel is a thin slice: a feature away from the focus point is not visible until the point moves there;
- the color scale is estimated from 100,000 random points and widens when a slice goes beyond it, so the colors may
  shift slightly while dragging; with *Clip outliers* it stays fixed;
- all variables share one range;
- *Points per axis* limits the resolution of a panel, which also never exceeds its size on screen (128 points at most).

## Running

Developed and tested on Windows with Python 3.14, numpy 2.5, PyQt6 6.11 and pyqtgraph 0.14.

```
git clone https://github.com/Egor1dzeN-learn/nd-function-viewer.git
cd nd-function-viewer
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

## Building the test function library

The **Formula** tab works right after installing the requirements. The **Test problems** tab needs
`native/testfuncs.dll`, which is not stored in the repository. It is built directly from the global-opt sources
(GCGen and `classic_functions.cpp`), so the values match the C++ project. You need `g++` in PATH
(for example, the MinGW bundled with CLion), and the build script currently targets Windows:

```
git clone --recursive https://github.com/Egor1dzeN/global-opt.git ..\global-opt
.venv\Scripts\python native\build.py            # global-opt is looked up in ..\global-opt
.venv\Scripts\python native\build.py D:\path\to\global-opt
```

## Adding your own function

**Once, without changing the code.** Open the **Formula** tab, type an expression in the `f =` field, set the ranges
and press Enter or **Plot**:

```
sin(x)*cos(y) + z^2
where(x > 0, x, -x) + y
x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2
```

The last variable used sets the dimension: a formula with `x5` is 5D even if `x1` … `x4` are not used.
The available functions are listed under the field.

**Permanently, in the Example list.** Add a line to `PRESETS` at the top of `main.py` and restart the application:

```python
PRESETS = [
    ...
    # (name in the list, formula, ranges of the variables)
    ("3D: My function", "sin(x) + y^2*z", [(-3, 3), (-2, 2), (0, 1)]),
]
```

Variables without a range use the range of `x`. Above 4D all variables share the first range.

**As a Python plugin.** Use this for functions that are too long for a formula or need the minimum marked on the
plot. Nothing has to be built.

1. Add a `.py` file to the `plugins/` folder and derive a class from `Function`:

   ```python
   import numpy as np

   from nd_function import Function, box


   class MyFunction(Function):
       name = "My Python function"  # must be unique, the class name by default
       min_dimension = 1            # supported dimensions, within 1 to 12
       max_dimension = 12

       def bounds(self, dimension):
           return box(dimension, -5, 5)

       def evaluate(self, x):
           return np.sum(x**2, axis=1)

       # Optional: mark the global minimum on the plot.
       def minimum(self, dimension):
           return [0.0] * dimension, 0.0
   ```

2. Restart the application and open the **Plugins** tab.

`evaluate` receives all points at once as an array of shape `(n, dimension)` and returns `n` values, so use numpy
columns such as `x[:, 0]` instead of looping over the points. To give each variable its own range, return
`[(-6, 6), (-3, 3)]` from `bounds`. One file can hold several classes, and `plugins/example_functions.py` is a working
example. Files and classes whose names start with `_` are skipped. If a plugin cannot be loaded, the error is shown in
the tab and the others keep working.

**As a C++ plugin.** Use this for functions that are expensive to compute. It needs only `g++` in PATH, global-opt is
not required.

1. Add a `.cpp` file to the `plugins/` folder, derive a class from `nfv::Function` and register it:

   ```cpp
   #include "nd_function.hpp"

   class MyFunction : public nfv::Function {
   public:
       std::string name() const override { return "My function"; }  // must be unique
       int min_dimension() const override { return 1; }             // supported dimensions, within 1 to 12
       int max_dimension() const override { return 12; }
       nfv::Bounds bounds(int dimension) const override { return nfv::box(dimension, -5.0, 5.0); }

       double evaluate(const nfv::Point &x) const override {
           double sum = 0.0;
           for (double coordinate: x) sum += coordinate * coordinate;
           return sum;
       }

       // Optional: mark the global minimum on the plot.
       bool minimum(int dimension, nfv::Point &point, double &value) const override {
           point.assign(dimension, 0.0);
           value = 0.0;
           return true;
       }
   };

   NFV_REGISTER_FUNCTION(MyFunction)
   ```

2. Close the application and build all plugins into `native/plugins.dll`:

   ```
   .venv\Scripts\python native\build_plugins.py
   ```

3. Start the application and open the **Plugins** tab.

One file can hold several classes, and `plugins/example_functions.cpp` is a working example. To give each variable its
own range, return `{{-6.0, 6.0}, {-3.0, 3.0}}` from `bounds`. Above 4D all variables must share one range.

## Files

- `main.py` — the application window and drawing;
- `hyperslice.py` — the HyperSlice view for `N > 4`;
- `function_parser.py` — safe formula parsing (whitelisted functions, no arbitrary code execution);
- `test_functions.py` — Python wrapper over `testfuncs.dll` and `plugins.dll` (ctypes) and the problem catalogs;
- `plugins/` — your Python and C++ functions;
- `nd_function.py` — the `Function` base class for Python plugins;
- `plugin_loader.py` — finds the Python and C++ plugins;
- `native/nd_function.hpp` — the `nfv::Function` base class for C++ plugins;
- `native/plugin_capi.cpp` — C interface to the plugins;
- `native/build_plugins.py` — plugin build script;
- `native/testfuncs_capi.cpp` — C interface to the GCGen problems and classic functions;
- `native/build.py` — build script of the test function library;
- `colormaps.py` — colormaps.

## Third-party software

The application uses [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) (GPL v3),
[pyqtgraph](https://github.com/pyqtgraph/pyqtgraph) (MIT), [NumPy](https://numpy.org/) (BSD) and
[PyOpenGL](https://github.com/mcfletch/pyopengl) (BSD). None of them are included in this repository.

The native library is built from [global-opt](https://github.com/Egor1dzeN/global-opt) (MIT) and
[GCGen](https://github.com/OptimLLab/GCGen) (MIT, University of Nizhny Novgorod). GCGen includes the GKLS generator:
M. Gaviano, D. E. Kvasov, D. Lera, Ya. D. Sergeyev, *Algorithm 829: Software for Generation of Classes of Test
Functions with Known Local and Global Minima for Global Optimization*, ACM TOMS 29(4), 2003, pp. 469–480.

## License

Copyright (C) 2026 Egor1dzeN

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version (`GPL-3.0-or-later`). It is distributed WITHOUT ANY WARRANTY. See [LICENSE](LICENSE) for the full text.
