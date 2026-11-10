"""YOLO label parsing that handles both box lines and polygon (segment) lines.

The merged dataset mixes the two: most lines are `cls cx cy w h`, but some
(233 in val alone) are `cls x1 y1 x2 y2 x3 y3 ...` polygons. Reading a polygon
line as a box gives a wrong crop, so always go through `parse_line`.
"""
from __future__ import annotations


def parse_line(line: str):
    """Return (cls, cx, cy, w, h) in normalised coords, or None for bad lines."""
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:
        return (cls, *vals)
    if len(vals) % 2:            # odd number of coordinates: not a valid polygon
        return None
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
