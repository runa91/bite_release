
import numpy as np 

def get_triangle_faces_from_pyvista_poly(poly):
    """Fetch all triangle faces."""
    stream = poly.faces
    tris = []
    i = 0
    while i < len(stream):
        n = stream[i]
        if n != 3:
            i += n + 1
            continue
        stop = i + n + 1
        tris.append(stream[i+1:stop])
        i = stop
    return np.array(tris)