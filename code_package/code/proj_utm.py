import numpy as np

# GRS80 ellipsoid (ETRS89)
A = 6378137.0
F = 1 / 298.257222101
K0 = 0.9996
LON0 = np.radians(-3.0)   # UTM zone 30N central meridian
FALSE_E = 500000.0
FALSE_N = 0.0

def lonlat_to_utm30n(lon_deg, lat_deg):
    """Snyder (1987) transverse Mercator forward formulas. Accurate to a
    few millimeters within a UTM zone (+-3 deg of the central meridian)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e2 = 2*F - F**2
    ep2 = e2 / (1 - e2)
    N = A / np.sqrt(1 - e2*np.sin(lat)**2)
    T = np.tan(lat)**2
    C = ep2 * np.cos(lat)**2
    Aa = (lon - LON0) * np.cos(lat)

    M = A * (
        (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*lat)
        + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*lat)
        - (35*e2**3/3072) * np.sin(6*lat)
    )

    x = K0 * N * (Aa + (1 - T + C) * Aa**3 / 6
                  + (5 - 18*T + T**2 + 72*C - 58*ep2) * Aa**5 / 120)
    y = K0 * (M + N * np.tan(lat) * (
        Aa**2 / 2 + (5 - T + 9*C + 4*C**2) * Aa**4 / 24
        + (61 - 58*T + T**2 + 600*C - 330*ep2) * Aa**6 / 720))

    easting = x + FALSE_E
    northing = y + FALSE_N
    return easting, northing
