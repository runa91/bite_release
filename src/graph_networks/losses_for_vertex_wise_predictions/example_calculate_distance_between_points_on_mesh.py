
"""
code adapted from: https://github.com/mikedh/trimesh/blob/main/examples/shortest.py
shortest.py
----------------
Given a mesh and two vertex indices find the shortest path
between the two vertices while only traveling along edges
of the mesh.
"""

# python src/graph_networks/losses_for_vertex_wise_predictions/calculate_distance_between_points_on_mesh.py


import os
import sys
import glob
import csv
import json
import shutil
import numpy as np
import trimesh
import networkx as nx



ROOT_PATH_MESH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/src/graph_networks/graphcmr/data/meshes/'
ROOT_PATH_ANNOT = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/data/stanext_related_data/ground_contact_annotations/stage3/'
STAN_V12_ROOT_DIR = '/ps/scratch/nrueegg/new_projects/Animals/data/dog_datasets/Stanford_Dogs_Dataset/StanfordExtra_V12/'
IMG_V12_DIR = STAN_V12_ROOT_DIR + 'StanExtV12_Images/'		
ROOT_OUT_PATH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/src/graph_networks/losses_for_vertex_wise_predictions/debugging_results/'


def read_csv(csv_file):
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        row_list = [{h:x for (h,x) in zip(headers,row)} for row in reader]
    return row_list

images_with_gc_labelled  = ['n02093991-Irish_terrier/n02093991_2874.jpg',
                            'n02093754-Border_terrier/n02093754_1062.jpg',
                            'n02092339-Weimaraner/n02092339_1672.jpg',
                            'n02096177-cairn/n02096177_4916.jpg',
                            'n02110185-Siberian_husky/n02110185_725.jpg',
                            'n02110806-basenji/n02110806_761.jpg',
                            'n02094433-Yorkshire_terrier/n02094433_2474.jpg',
                            'n02097474-Tibetan_terrier/n02097474_8796.jpg',
                            'n02099601-golden_retriever/n02099601_2495.jpg']


# ----- PART 1: load all ground contact annotations
gc_annot_csv = ROOT_PATH_ANNOT + 'my_gcannotations_qualification.csv'
gc_row_list = read_csv(gc_annot_csv)
json_acceptable_string = (gc_row_list[0]['vertices']).replace("'", "\"")
gc_dict = json.loads(json_acceptable_string)


# ----- PART 2: load and prepare the mesh
'''
from smal_pytorch.smal_model.smal_torch_new import SMAL
smal = SMAL()
verts = smal.v_template.detach().cpu().numpy()
faces = smal.faces.detach().cpu().numpy()
'''
path_mesh = ROOT_PATH_MESH + 'mesh_downsampling_meshesmy_smpl_39dogsnorm_Jr_4_dog_template_downsampled0.obj'
my_mesh = trimesh.load_mesh(path_mesh, process=False,  maintain_order=True)
verts = my_mesh.vertices
faces = my_mesh.faces
# edges without duplication
edges = my_mesh.edges_unique
# the actual length of each unique edge
length = my_mesh.edges_unique_length
# create the graph with edge attributes for length (option A)
#   g = nx.Graph()
#   for edge, L in zip(edges, length): g.add_edge(*edge, length=L)
# you can create the graph with from_edgelist and
# a list comprehension (option B)
ga = nx.from_edgelist([(e[0], e[1], {'length': L}) for e, L in zip(edges, length)])


# ----- PART 3: calculate the distances between all vertex pairs
calc_dist_mat = False
if calc_dist_mat:
    # calculate distances between all possible vertex pairs
    # shortest_path = nx.shortest_path(ga, source=ind_v0, target=ind_v1, weight='length')
    # shortest_dist = nx.shortest_path_length(ga, source=ind_v0, target=ind_v1, weight='length')
    dis = dict(nx.shortest_path_length(ga, weight='length', method='dijkstra'))
    vertex_distances = np.zeros((n_verts_smal, n_verts_smal))
    for ind_v0 in range(n_verts_smal):
        print(ind_v0)
        for ind_v1 in range(ind_v0, n_verts_smal):
            vertex_distances[ind_v0, ind_v1] = dis[ind_v0][ind_v1]
            vertex_distances[ind_v1, ind_v0] = dis[ind_v0][ind_v1]
    # save those distances
    np.save(ROOT_OUT_PATH + 'all_vertex_distances.npy', vertex_distances)
    vert_dists = vertex_distances
else:
    vert_dists = np.load(ROOT_OUT_PATH + 'all_vertex_distances.npy')


# ----- PART 4: prepare contact annotation
n_verts_smal = 3889
for ind_img in range(len(images_with_gc_labelled)):     #  range(len(gc_dict.keys())):
    name = images_with_gc_labelled[ind_img]
    print('work on image ' + name)
    gc_info_raw = gc_dict['bite/' + name]      # a list with all vertex numbers that are in ground contact
    gc_vertices = []
    gc_info_np = np.zeros((n_verts_smal))
    for ind_v in gc_info_raw:
        if ind_v < n_verts_smal:
            gc_vertices.append(ind_v)
            gc_info_np[ind_v] = 1
    # save a visualization of those annotations 
    vert_colors = np.repeat(255*gc_info_np[:, None], 3, 1)
    my_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,  maintain_order=True)
    my_mesh.visual.vertex_colors = vert_colors
    my_mesh.export(ROOT_OUT_PATH + (name.split('/')[1]).replace('.jpg', '_withgc.obj'))
    img_path = IMG_V12_DIR + name
    shutil.copy(img_path, ROOT_OUT_PATH + name.split('/')[1])

    # ----- PART 5: calculate for each vertex the distance to the closest element of the other group
    non_gc_vertices = list(set(range(n_verts_smal)) - set(gc_vertices))
    print('vertices in contact: ' + str(len(gc_vertices)))
    print('vertices without contact: ' + str(len(non_gc_vertices)))
    vertex_overview = np.zeros((n_verts_smal, 3))   # first: no-contact=0 contact=1     second: index of vertex     third: dist
    vertex_overview[:, 0] = gc_info_np
    # loop through all contact vertices
    for ind_v in gc_vertices:
        min_length = 100
        for ind_v_ps in non_gc_vertices:    # possible solution
            # this_path = nx.shortest_path(ga, source=ind_v, target=ind_v_ps, weight='length')
            # this_length = nx.shortest_path_length(ga, source=ind_v, target=ind_v_ps, weight='length')
            this_length = vert_dists[ind_v, ind_v_ps]
            if this_length < min_length:
                min_length = this_length
                vertex_overview[ind_v, 1] = ind_v_ps
                vertex_overview[ind_v, 2] = this_length
    # loop through all non-contact vertices
    for ind_v in non_gc_vertices:
        min_length = 100
        for ind_v_ps in gc_vertices:    # possible solution
            # this_path = nx.shortest_path(ga, source=ind_v, target=ind_v_ps, weight='length')
            # this_length = nx.shortest_path_length(ga, source=ind_v, target=ind_v_ps, weight='length')
            this_length = vert_dists[ind_v, ind_v_ps]
            if this_length < min_length:
                min_length = this_length
                vertex_overview[ind_v, 1] = ind_v_ps
                vertex_overview[ind_v, 2] = this_length
    # save a colored mesh
    my_mesh_dists = my_mesh.copy()
    scale_0 = (vertex_overview[vertex_overview[:, 0]==0, 2]).max()
    scale_1 = (vertex_overview[vertex_overview[:, 0]==1, 2]).max()
    vert_col = np.zeros((n_verts_smal, 3)) 
    vert_col[vertex_overview[:, 0]==0, 1] = vertex_overview[vertex_overview[:, 0]==0, 2] * 255 / scale_0     # green
    vert_col[vertex_overview[:, 0]==1, 0] = vertex_overview[vertex_overview[:, 0]==1, 2] * 255 / scale_1     # red
    my_mesh_dists.visual.vertex_colors = np.uint8(vert_col)
    my_mesh_dists.export(ROOT_OUT_PATH + (name.split('/')[1]).replace('.jpg', '_withgcdists.obj'))




import pdb; pdb.set_trace()











