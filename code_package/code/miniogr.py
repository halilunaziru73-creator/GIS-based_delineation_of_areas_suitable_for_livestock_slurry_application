"""
Minimal pure-Python ESRI Shapefile (.shp/.dbf) reader.
No GDAL/OGR/pyshp/fiona/geopandas dependency - built for this environment.
Supports Polygon / PolygonZ / PolygonM shape types (5, 15, 25) which is all
that is present in the dataset used in this study.
"""
import struct
import numpy as np

SHAPE_POLYGON = 5
SHAPE_POLYGONZ = 15
SHAPE_POLYGONM = 25

def _read_dbf(path):
    with open(path, 'rb') as f:
        data = f.read()
    n_records = struct.unpack('<i', data[4:8])[0]
    header_len = struct.unpack('<h', data[8:10])[0]
    record_len = struct.unpack('<h', data[10:12])[0]
    # field descriptors start at byte 32, each 32 bytes, terminated by 0x0D
    fields = []
    pos = 32
    while data[pos:pos+1] != b'\x0d':
        name = data[pos:pos+11].split(b'\x00')[0].decode('latin-1')
        ftype = chr(data[pos+11])
        flen = data[pos+16]
        fields.append((name, ftype, flen))
        pos += 32
    records = []
    rec_start = header_len
    for i in range(n_records):
        rec = data[rec_start:rec_start+record_len]
        rec_start += record_len
        if rec[0:1] == b'*':  # deleted record
            records.append(None)
            continue
        offset = 1
        row = {}
        for name, ftype, flen in fields:
            raw = rec[offset:offset+flen]
            offset += flen
            val = raw.decode('latin-1', errors='replace').strip()
            if ftype in ('N', 'F') and val not in ('',):
                try:
                    val = float(val) if ('.' in val) else int(val)
                except ValueError:
                    pass
            row[name] = val
        records.append(row)
    return records, fields

def _signed_area(ring):
    x = ring[:, 0]; y = ring[:, 1]
    return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

def read_shapefile(shp_path, dbf_path=None):
    """Returns list of dicts: {'attrs':{...}, 'rings':[ring_xy_array,...]}
    rings are raw rings straight from the file (exterior CW / holes CCW,
    per ESRI convention). Grouping into polygons-with-holes is done by
    the caller (see geomtools.polygons_with_holes)."""
    if dbf_path is None:
        dbf_path = shp_path[:-4] + '.dbf'
    attr_records, fields = _read_dbf(dbf_path)

    with open(shp_path, 'rb') as f:
        buf = f.read()
    file_len_words = struct.unpack('>i', buf[24:28])[0]
    file_len = file_len_words * 2
    shape_type = struct.unpack('<i', buf[32:36])[0]
    assert shape_type in (SHAPE_POLYGON, SHAPE_POLYGONZ, SHAPE_POLYGONM), \
        f"unsupported shape type {shape_type} in {shp_path}"

    pos = 100
    idx = 0
    out = []
    while pos < file_len:
        rec_num, content_len_words = struct.unpack('>ii', buf[pos:pos+8])
        pos += 8
        content_len = content_len_words * 2
        content = buf[pos:pos+content_len]
        pos += content_len
        st = struct.unpack('<i', content[0:4])[0]
        rings = []
        if st != 0:  # not null shape
            num_parts, num_points = struct.unpack('<ii', content[36:44])
            parts = np.frombuffer(content[44:44+4*num_parts], dtype='<i4')
            pts_start = 44 + 4*num_parts
            pts = np.frombuffer(content[pts_start:pts_start+16*num_points], dtype='<f8')
            pts = pts.reshape(-1, 2)
            parts = list(parts) + [num_points]
            for i in range(num_parts):
                ring = pts[parts[i]:parts[i+1]]
                if len(ring) >= 3:
                    rings.append(ring)
        attrs = attr_records[idx] if idx < len(attr_records) else {}
        out.append({'attrs': attrs, 'rings': rings})
        idx += 1
    return out
