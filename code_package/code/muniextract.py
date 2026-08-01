import struct
import numpy as np
from miniogr import _read_dbf

def extract_municipality(shp_path, dbf_path, shx_path, name_field, name_value):
    recs, fields = _read_dbf(dbf_path)
    matches = [i for i, r in enumerate(recs) if r and r.get(name_field) == name_value]
    if not matches:
        raise ValueError(f'{name_value} not found')
    idx = matches[0]
    with open(shx_path, 'rb') as f:
        shx = f.read()
    rec_off, rec_len_w = struct.unpack('>ii', shx[100+idx*8:100+idx*8+8])
    offset_bytes = rec_off * 2
    with open(shp_path, 'rb') as f:
        f.seek(offset_bytes)
        rec_num, content_len_w = struct.unpack('>ii', f.read(8))
        content = f.read(content_len_w * 2)
    st = struct.unpack('<i', content[0:4])[0]
    num_parts, num_points = struct.unpack('<ii', content[36:44])
    parts = np.frombuffer(content[44:44+4*num_parts], dtype='<i4')
    pts_start = 44 + 4*num_parts
    pts = np.frombuffer(content[pts_start:pts_start+16*num_points], dtype='<f8').reshape(-1, 2)
    parts = list(parts) + [num_points]
    rings = [pts[parts[i]:parts[i+1]] for i in range(num_parts)]
    return {'attrs': recs[idx], 'rings': rings}
