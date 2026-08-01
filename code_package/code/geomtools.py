"""
Minimal geometry / rasterization helpers built on numpy + scikit-image,
substituting for rasterio.features.rasterize / gdal_rasterize which are
not available in this container (no GDAL/rasterio/geopandas/shapely).
"""
import numpy as np
from skimage.draw import polygon as sk_polygon

def _signed_area(ring):
    x = ring[:, 0]; y = ring[:, 1]
    return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

def polygons_with_holes(rings):
    """Group raw shapefile rings into [(exterior, [hole1, hole2,...]), ...]
    ESRI convention: exterior rings are clockwise (signed_area<0),
    holes are counter-clockwise (signed_area>0), holes follow their exterior."""
    polys = []
    for ring in rings:
        a = _signed_area(ring)
        if a < 0 or not polys:
            polys.append([ring, []])
        else:
            polys[-1][1].append(ring)
    return polys

class Grid:
    """Simple affine raster grid: world (x,y) -> pixel (row, col).
    origin_x/origin_y = world coords of the top-left pixel corner.
    pixel_size in meters (assumes north-up, square pixels)."""
    def __init__(self, origin_x, origin_y, pixel_size, n_rows, n_cols):
        self.ox, self.oy, self.px = origin_x, origin_y, pixel_size
        self.n_rows, self.n_cols = n_rows, n_cols

    def world_to_pixel(self, xy):
        col = (xy[:, 0] - self.ox) / self.px
        row = (self.oy - xy[:, 1]) / self.px
        return row, col

    def bounds(self):
        return (self.ox, self.oy - self.n_rows*self.px,
                self.ox + self.n_cols*self.px, self.oy)


def rasterize_features(features, grid, burn_field=None, default_value=1,
                        dtype=np.float32, combine='max'):
    """features: list of {'attrs':.., 'rings':[...]}  (raw output of read_shapefile)
    burn_field: attrs key whose numeric value is burned; if None, default_value used.
    combine: 'max' or 'overwrite' - how overlapping polygons are merged."""
    out = np.zeros((grid.n_rows, grid.n_cols), dtype=dtype)
    written = np.zeros((grid.n_rows, grid.n_cols), dtype=bool)
    for feat in features:
        rings = feat['rings']
        if not rings:
            continue
        value = default_value if burn_field is None else feat['attrs'].get(burn_field, default_value)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        for exterior, holes in polygons_with_holes(rings):
            r_ext, c_ext = grid.world_to_pixel(exterior)
            r0 = int(np.floor(min(r_ext.min(), 0) if r_ext.min() < 0 else r_ext.min()))
            r0 = max(0, int(np.floor(r_ext.min())))
            r1 = min(grid.n_rows, int(np.ceil(r_ext.max())) + 1)
            c0 = max(0, int(np.floor(c_ext.min())))
            c1 = min(grid.n_cols, int(np.ceil(c_ext.max())) + 1)
            if r1 <= r0 or c1 <= c0:
                continue
            h, w = r1 - r0, c1 - c0
            rr, cc = sk_polygon(r_ext - r0, c_ext - c0, shape=(h, w))
            mask = np.zeros((h, w), dtype=bool)
            mask[rr, cc] = True
            for hole in holes:
                r_h, c_h = grid.world_to_pixel(hole)
                rrh, cch = sk_polygon(r_h - r0, c_h - c0, shape=(h, w))
                hole_mask = np.zeros((h, w), dtype=bool)
                # clip hole coords that fall outside local window
                valid = (rrh >= 0) & (rrh < h) & (cch >= 0) & (cch < w)
                hole_mask[rrh[valid], cch[valid]] = True
                mask &= ~hole_mask
            sub_out = out[r0:r1, c0:c1]
            sub_written = written[r0:r1, c0:c1]
            if combine == 'max':
                sub_out[mask] = np.maximum(sub_out[mask], value)
            else:
                sub_out[mask] = value
            sub_written[mask] = True
    return out, written


def euclid_buffer_mask(binary_mask, pixel_size, distance_m):
    """Grid-space buffer: True within distance_m (meters) of any True cell,
    computed via a Euclidean distance transform (replaces vector .buffer())."""
    from scipy.ndimage import distance_transform_edt
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=bool)
    inv = ~binary_mask
    dist = distance_transform_edt(inv) * pixel_size
    return dist <= distance_m
