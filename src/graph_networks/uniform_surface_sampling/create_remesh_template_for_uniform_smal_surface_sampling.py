
# python src/graph_networks/uniform_surface_sampling/create_remesh_template_for_uniform_smal_surface_sampling.py

import numpy as np
import pyacvd
import pyvista as pv
import trimesh
import pickle as pkl

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from combined_model.helper import get_triangle_faces_from_pyvista_poly


ROOT_OUT_PATH = ...........
ROOT_PATH_MESH = ...........

n_points = 25000    # 6000     # 4000
name = 'my_smpl_39dogsnorm_Jr_4_dog_remesh25000'        # 6000'     # 'my_smpl_39dogsnorm_Jr_4_dog_remesh4000'


# load smal mesh (could also be loaded using SMAL class, this is just the SMAL dog template)
path_mesh = ROOT_PATH_MESH + 'mesh_downsampling_meshesmy_smpl_39dogsnorm_Jr_4_dog_template_downsampled0.obj'
my_mesh = trimesh.load_mesh(path_mesh, process=False,  maintain_order=True)
verts = my_mesh.vertices
faces = my_mesh.faces

# read smal dog mesh with pyvista
mesh_pv = pv.read(path_mesh)
clus = pyacvd.Clustering(mesh_pv)

# remesh the surface (see https://github.com/pyvista/pyacvd)
clus.subdivide(3)
clus.cluster(n_points)          # clus.cluster(20000)
remesh = clus.create_mesh()
remesh_points_of_interest = np.asarray(remesh.points)

# save the resulting mesh 
# remesh.save(ROOT_OUT_PATH + name + '.ply')
remesh_triangle_faces = get_triangle_faces_from_pyvista_poly(remesh)
remesh_tri = trimesh.Trimesh(vertices=remesh_points_of_interest, faces=remesh_triangle_faces, process=False,  maintain_order=True)
remesh_tri.export(ROOT_OUT_PATH + name + '.obj')

# get barycentric coordinates 
points_closest, dists_closest, faceid_closest = trimesh.proximity.closest_point(my_mesh, remesh_points_of_interest)
barys_closest = trimesh.triangles.points_to_barycentric(my_mesh.vertices[my_mesh.faces[faceid_closest]], points_closest)     # , method='cramer') 

# test that we can get the vertex location of the remeshes mesh back
#   -> similarly we will be able to calculate new vertex locations for a deformed smal mesh
verts_closest = np.einsum('ij,ijk->ik', barys_closest, my_mesh.vertices[my_mesh.faces[faceid_closest]])

# save all relevant (and more) information
remeshing_dict = {'remeshed_name': name + '.obj', 'is_symmetric': 'no', 'remeshed_verts':  np.asarray(remesh.points), 'smal_mesh': path_mesh, 'points_closest': points_closest, 'dists_closest': dists_closest, 'faceid_closest': faceid_closest, 'barys_closest': barys_closest, 'string_to_get_new_point_locations_from_smal': 'verts_closest = np.einsum(ij,ijk->ik, barys_closest, smal_mesh.vertices[smal_mesh.faces[faceid_closest]])'}
with open(ROOT_OUT_PATH + name + '_info.pkl', 'wb') as f: 
    pkl.dump(remeshing_dict, f)

