"""Colormaps: function value (0..1 after normalization) -> color."""
import pyqtgraph as pg

# Physical map palette: lowlands are blue (sea), the middle is green/yellow, peaks are red.
_CUSTOM = {
    "Terrain": [
        (0.00, (8, 29, 88)),
        (0.18, (34, 94, 168)),
        (0.35, (65, 182, 196)),
        (0.48, (120, 198, 121)),
        (0.62, (255, 237, 111)),
        (0.78, (253, 141, 60)),
        (1.00, (189, 0, 38)),
    ],
    "Cool–warm": [
        (0.0, (59, 76, 192)),
        (0.5, (221, 221, 221)),
        (1.0, (180, 4, 38)),
    ],
}
_BUILTIN = {"Turbo": "turbo", "Viridis": "viridis", "Plasma": "plasma"}

COLORMAP_NAMES = list(_CUSTOM) + list(_BUILTIN)


def get_colormap(name: str, alpha: int = 255) -> pg.ColorMap:
    if name in _CUSTOM:
        stops = _CUSTOM[name]
        return pg.ColorMap([p for p, _ in stops], [(*c, alpha) for _, c in stops])
    cmap = pg.colormap.get(_BUILTIN[name])
    if alpha == 255:
        return cmap
    pos, colors = cmap.getStops(mode="byte")
    colors = colors.copy()
    colors[:, 3] = alpha
    return pg.ColorMap(pos, colors)
