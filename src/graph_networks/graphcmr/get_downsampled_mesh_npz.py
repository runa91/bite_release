
# try to use aenv_conda3  (maybe also export PYOPENGL_PLATFORM=osmesa)
# python src/graph_networks/graphcmr/get_downsampled_mesh_npz.py

# see https://github.com/nkolot/GraphCMR/issues/35


from __future__ import print_function
# import mesh_sampling
from psbody.mesh import Mesh, MeshViewer, MeshViewers
import numpy as np
import json
import os
import copy
import argparse
import pickle
import time
import sys
import trimesh



sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from barc_for_bite.src.graph_networks.graphcmr.pytorch_coma_mesh_operations import generate_transform_matrices
from barc_for_bite.src.configs.SMAL_configs import SMAL_MODEL_CONFIG
from barc_for_bite.src.smal_pytorch.smal_model.smal_torch_new import SMAL

SMAL_MODEL_TYPE = '39dogs_diffsize'        # '39dogs_diffsize'     # '39dogs_norm'  # 'barc'
smal_model_path = SMAL_MODEL_CONFIG[SMAL_MODEL_TYPE]['smal_model_path']

output_path = os.path.join(os.path.dirname(__file__), '../', '../', 'my_output_folder')  
data_path_root = output_path + "/graph_networks/graphcmr/data/"

smal_dog_model_name = os.path.basename(smal_model_path).split('.pkl')[0]    # 'my_smpl_SMBLD_nbj_v3'
suffix = "_template"
template_obj_path = data_path_root + smal_dog_model_name + suffix + ".obj"

print("Loading smal .. ")
print(SMAL_MODEL_TYPE)
print(smal_model_path)

smal = SMAL(smal_model_type=SMAL_MODEL_TYPE, template_name='neutral')
smal_verts = smal.v_template.detach().cpu().numpy()     # (3889, 3)
smal_faces = smal.f                                     # (7774, 3)
smal_trimesh = trimesh.base.Trimesh(vertices=smal_verts, faces=smal_faces, process=False,  maintain_order=True)
smal_trimesh.export(file_obj=template_obj_path)  # file_type='obj')


print("Loading data .. ")
reference_mesh_file = template_obj_path # 'data/barc_neutral_vertices.obj'      # 'data/smpl_neutral_vertices.obj'
reference_mesh = Mesh(filename=reference_mesh_file)

# ds_factors = [4, 4]     # ds_factors = [4,1]	# Sampling factor of the mesh at each stage of sampling
ds_factors = [4, 4, 4, 4]
print("Generating Transform Matrices ..")


# Generates adjecency matrices A, downsampling matrices D, and upsamling matrices U by sampling
# the mesh 4 times. Each time the mesh is sampled by a factor of 4

# M,A,D,U = mesh_sampling.generate_transform_matrices(reference_mesh, ds_factors)
M,A,D,U = generate_transform_matrices(reference_mesh, ds_factors)

# REMARK: there is a warning:
#   lib/graph_networks/graphcmr/../../../lib/graph_networks/graphcmr/pytorch_coma_mesh_operations.py:237: FutureWarning: `rcond` parameter will 
#   change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
#   To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.


print(type(A))
np.savez(data_path_root + 'mesh_downsampling_' + smal_dog_model_name + suffix + '.npz', A = A, D = D, U = U)
np.savez(data_path_root + 'meshes/' + 'mesh_downsampling_meshes' + smal_dog_model_name + suffix + '.npz', M = M)

for ind_m, my_mesh in enumerate(M):
    new_suffix = '_template_downsampled' + str(ind_m)
    my_mesh_tri = trimesh.Trimesh(vertices=my_mesh.v, faces=my_mesh.f, process=False,  maintain_order=True)
    my_mesh_tri.export(data_path_root + 'meshes/' + 'mesh_downsampling_meshes' + smal_dog_model_name + new_suffix + '.obj')





