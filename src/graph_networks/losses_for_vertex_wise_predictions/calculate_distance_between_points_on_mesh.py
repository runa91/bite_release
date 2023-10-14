
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
import tqdm
import numpy as np
import pickle as pkl
import trimesh
import networkx as nx





def read_csv(csv_file):
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        row_list = [{h:x for (h,x) in zip(headers,row)} for row in reader]
    return row_list


def load_all_template_mesh_distances(root_out_path, filename='all_vertex_distances.npy'):
    vert_dists = np.load(root_out_path + filename)
    return vert_dists


def prepare_graph_from_template_mesh_and_calculate_all_distances(path_mesh, root_out_path, calc_dist_mat=False):
    # root_out_path = ROOT_OUT_PATH
    '''
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    smal = SMAL()
    verts = smal.v_template.detach().cpu().numpy()
    faces = smal.faces.detach().cpu().numpy()
    '''
    # path_mesh = ROOT_PATH_MESH + 'mesh_downsampling_meshesmy_smpl_39dogsnorm_Jr_4_dog_template_downsampled0.obj'
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
    # calculate the distances between all vertex pairs
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
        np.save(root_out_path + 'all_vertex_distances.npy', vertex_distances)
        vert_dists = vertex_distances
    else:
        vert_dists = np.load(root_out_path + 'all_vertex_distances.npy')
    return ga, vert_dists


def calculate_vertex_overview_for_gc_annotation(name, gc_info_raw, vert_dists, root_out_path_vis=None, verts=None, faces=None, img_v12_dir=None):
    # input:
    #   root_out_path_vis = ROOT_OUT_PATH
    #   img_v12_dir = IMG_V12_DIR
    #   name = images_with_gc_labelled[ind_img]
    #   gc_info_raw = gc_dict['bite/' + name] 
    # output:
    #    vertex_overview: np array of shape (n_verts_smal, 3) with [first: no-contact=0 contact=1     second: index of vertex     third: dist]
    n_verts_smal = 3889
    gc_vertices = []
    gc_info_np = np.zeros((n_verts_smal))
    for ind_v in gc_info_raw:
        if ind_v < n_verts_smal:
            gc_vertices.append(ind_v)
            gc_info_np[ind_v] = 1
    # save a visualization of those annotations 
    if root_out_path_vis is not None:
        my_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,  maintain_order=True)
    if img_v12_dir is not None and root_out_path_vis is not None:
        vert_colors = np.repeat(255*gc_info_np[:, None], 3, 1)
        my_mesh.visual.vertex_colors = vert_colors
        my_mesh.export(root_out_path_vis + (name).replace('.jpg', '_withgc.obj'))
        img_path = img_v12_dir + name
        shutil.copy(img_path, root_out_path_vis + name)
    # calculate for each vertex the distance to the closest element of the other group
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
    if root_out_path_vis is not None:
        # save a colored mesh
        my_mesh_dists = my_mesh.copy()
        scale_0 = (vertex_overview[vertex_overview[:, 0]==0, 2]).max()
        scale_1 = (vertex_overview[vertex_overview[:, 0]==1, 2]).max()
        vert_col = np.zeros((n_verts_smal, 3)) 
        vert_col[vertex_overview[:, 0]==0, 1] = vertex_overview[vertex_overview[:, 0]==0, 2] * 255 / scale_0     # green
        vert_col[vertex_overview[:, 0]==1, 0] = vertex_overview[vertex_overview[:, 0]==1, 2] * 255 / scale_1     # red
        my_mesh_dists.visual.vertex_colors = np.uint8(vert_col)
        my_mesh_dists.export(root_out_path_vis + (name).replace('.jpg', '_withgcdists.obj'))
    return vertex_overview








def main():

    ROOT_PATH_MESH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/src/graph_networks/graphcmr/data/meshes/'
    ROOT_PATH_ANNOT = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage3/'
    IMG_V12_DIR = '/ps/scratch/nrueegg/new_projects/Animals/data/dog_datasets/Stanford_Dogs_Dataset/StanfordExtra_V12/StanExtV12_Images/'		
    # ROOT_OUT_PATH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/src/graph_networks/losses_for_vertex_wise_predictions/debugging_results/'
    ROOT_OUT_PATH = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage3/'
    ROOT_OUT_PATH_VIS = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage3/vis/'
    ROOT_OUT_PATH_DISTSGCNONGC = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage3/vertex_distances_gc_nongc/'
    ROOT_PATH_ALL_VERT_DIST_TEMPLATE  = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/'

    # load all vertex distances
    path_mesh = ROOT_PATH_MESH + 'mesh_downsampling_meshesmy_smpl_39dogsnorm_Jr_4_dog_template_downsampled0.obj'
    my_mesh = trimesh.load_mesh(path_mesh, process=False,  maintain_order=True)
    verts = my_mesh.vertices
    faces = my_mesh.faces
    # vert_dists, ga = prepare_graph_from_template_mesh_and_calculate_all_distances(path_mesh, ROOT_OUT_PATH, calc_dist_mat=False)
    vert_dists = load_all_template_mesh_distances(ROOT_PATH_ALL_VERT_DIST_TEMPLATE, filename='all_vertex_distances.npy')




    all_keys = []
    gc_dict = {}
    # data/stanext_related_data/ground_contact_annotations/stage3/main_partA1667_20221021_140108.csv
    # for csv_file in ['main_partA500_20221018_131139.csv', 'pilot_20221017_104201.csv', 'my_gcannotations_qualification.csv']:        
    # for csv_file in ['main_partA1667_20221021_140108.csv', 'main_partA500_20221018_131139.csv', 'pilot_20221017_104201.csv', 'my_gcannotations_qualification.csv']:
    for csv_file in ['main_partA1667_20221021_140108.csv', 'main_partA500_20221018_131139.csv', 'main_partB20221023_150926.csv', 'pilot_20221017_104201.csv', 'my_gcannotations_qualification.csv']:
        # load all ground contact annotations
        gc_annot_csv = ROOT_PATH_ANNOT + csv_file   # 'my_gcannotations_qualification.csv'
        gc_row_list = read_csv(gc_annot_csv)
        for ind_row in range(len(gc_row_list)):
            json_acceptable_string = (gc_row_list[ind_row]['vertices']).replace("'", "\"")
            gc_dict_temp = json.loads(json_acceptable_string)
            all_keys.extend(gc_dict_temp.keys())
            gc_dict.update(gc_dict_temp)
        print(len(gc_dict.keys()))

    print('number of labeled images: ' + str(len(gc_dict.keys())))      # WHY IS THIS ONLY 699?

    import pdb; pdb.set_trace()


    # prepare and save contact annotations including distances
    vertex_overview_dict = {}
    for ind_img, name_ingcdict in enumerate(gc_dict.keys()):     #  range(len(gc_dict.keys())):
        name = name_ingcdict.split('bite/')[1]
        # name = images_with_gc_labelled[ind_img]
        print('work on image ' + str(ind_img) + ': ' + name)
        # gc_info_raw = gc_dict['bite/' + name]      # a list with all vertex numbers that are in ground contact
        gc_info_raw = gc_dict[name_ingcdict]      # a list with all vertex numbers that are in ground contact

        if not os.path.exists(ROOT_OUT_PATH_VIS + name.split('/')[0]): os.makedirs(ROOT_OUT_PATH_VIS + name.split('/')[0])
        if not os.path.exists(ROOT_OUT_PATH_DISTSGCNONGC + name.split('/')[0]): os.makedirs(ROOT_OUT_PATH_DISTSGCNONGC + name.split('/')[0])

        vertex_overview = calculate_vertex_overview_for_gc_annotation(name, gc_info_raw, vert_dists, root_out_path_vis=ROOT_OUT_PATH_VIS, verts=verts, faces=faces, img_v12_dir=None)
        np.save(ROOT_OUT_PATH_DISTSGCNONGC + name.replace('.jpg', '_gc_vertdists_overview.npy'), vertex_overview)

        vertex_overview_dict[name.split('.')[0]] = {'gc_vertdists_overview': vertex_overview, 'gc_index_list': gc_info_raw}



        

    # import pdb; pdb.set_trace()

    with open(ROOT_OUT_PATH + 'gc_annots_overview_stage3complete_withtraintestval_xx.pkl', 'wb') as fp:
        pkl.dump(vertex_overview_dict, fp)













if __name__ == "__main__":
    main()







