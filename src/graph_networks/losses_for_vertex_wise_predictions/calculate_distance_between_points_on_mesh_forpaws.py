
"""
code adapted from: https://github.com/mikedh/trimesh/blob/main/examples/shortest.py
shortest.py
----------------
Given a mesh and two vertex indices find the shortest path
between the two vertices while only traveling along edges
of the mesh.
"""

# python src/graph_networks/losses_for_vertex_wise_predictions/calculate_distance_between_points_on_mesh_forpaws.py


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


def summarize_results_stage2b(row_list, display_worker_performance=False):
    # four catch trials are included in every batch
    annot_n02088466_3184 = {'paw_rb': 0, 'paw_rf': 1, 'paw_lb': 1, 'paw_lf': 1, 'additional_part': 0, 'no_contact': 0}
    annot_n02100583_9922 = {'paw_rb': 1, 'paw_rf': 0, 'paw_lb': 0, 'paw_lf': 0, 'additional_part': 0, 'no_contact': 0}
    annot_n02105056_2798 = {'paw_rb': 1, 'paw_rf': 1, 'paw_lb': 1, 'paw_lf': 1, 'additional_part': 1, 'no_contact': 0}
    annot_n02091831_2288 = {'paw_rb': 0, 'paw_rf': 1, 'paw_lb': 1, 'paw_lf': 0, 'additional_part': 0, 'no_contact': 0}
    all_comments = []
    all_annotations = {}
    for row in row_list:
        all_comments.append(row['Answer.submitComments'])
        worker_id = row['WorkerId']
        if display_worker_performance:
            print('----------------------------------------------------------------------------------------------')
            print('Worker ID: ' + worker_id)
        n_wrong = 0
        n_correct = 0
        for ind in range(0, len(row['Answer.submitValuesNotSure'].split(';')) - 1):
            input_image = (row['Input.images'].split(';')[ind]).split('StanExtV12_Images/')[-1]
            paw_rb = row['Answer.submitValuesRightBack'].split(';')[ind] 
            paw_rf = row['Answer.submitValuesRightFront'].split(';')[ind] 
            paw_lb = row['Answer.submitValuesLeftBack'].split(';')[ind] 
            paw_lf = row['Answer.submitValuesLeftFront'].split(';')[ind]  
            addpart = row['Answer.submitValuesAdditional'].split(';')[ind] 
            no_contact = row['Answer.submitValuesNoContact'].split(';')[ind]  
            unsure = row['Answer.submitValuesNotSure'].split(';')[ind] 
            annot = {'paw_rb': paw_rb, 'paw_rf': paw_rf, 'paw_lb': paw_lb, 'paw_lf': paw_lf, 
                    'additional_part': addpart, 'no_contact': no_contact, 'not_sure': unsure, 
                    'worker_id': worker_id}     # , 'input_image': input_image}
            if ind == 0:
                gt = annot_n02088466_3184
            elif ind == 1:
                gt = annot_n02105056_2798
            elif ind == 2:
                gt = annot_n02091831_2288
            elif ind == 3:
                gt = annot_n02100583_9922
            else:
                pass
            if ind < 4:
                for key in gt.keys():
                    if str(annot[key]) == str(gt[key]):
                        n_correct += 1
                    else:
                        if display_worker_performance:
                            print(input_image)
                            print(key + ':[ expected: ' + str(gt[key]) + '   predicted: ' + str(annot[key]) + ' ]')
                        n_wrong += 1
            else:
                all_annotations[input_image] = annot
        if display_worker_performance:
            print('n_correct: ' + str(n_correct))
            print('n_wrong: ' + str(n_wrong))
    return all_annotations, all_comments






def main():

    ROOT_PATH_MESH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/src/graph_networks/graphcmr/data/meshes/'
    ROOT_PATH_ANNOT = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage2b/'
    IMG_V12_DIR = '/ps/scratch/nrueegg/new_projects/Animals/data/dog_datasets/Stanford_Dogs_Dataset/StanfordExtra_V12/StanExtV12_Images/'		
    # ROOT_OUT_PATH = '/is/cluster/work/nrueegg/icon_pifu_related/barc_for_bite/src/graph_networks/losses_for_vertex_wise_predictions/debugging_results/'
    ROOT_OUT_PATH = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage2b/'
    ROOT_OUT_PATH_VIS = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage2b/vis/'
    ROOT_OUT_PATH_DISTSGCNONGC = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/stage2b/vertex_distances_gc_nongc/'
    ROOT_PATH_ALL_VERT_DIST_TEMPLATE  = STANEXT_RELATED_DATA_ROOT_DIR + '/ground_contact_annotations/'

    # load all vertex distances
    path_mesh = ROOT_PATH_MESH + 'mesh_downsampling_meshesmy_smpl_39dogsnorm_Jr_4_dog_template_downsampled0.obj'
    my_mesh = trimesh.load_mesh(path_mesh, process=False,  maintain_order=True)
    verts = my_mesh.vertices
    faces = my_mesh.faces
    # vert_dists, ga = prepare_graph_from_template_mesh_and_calculate_all_distances(path_mesh, ROOT_OUT_PATH, calc_dist_mat=False)
    vert_dists = load_all_template_mesh_distances(ROOT_PATH_ALL_VERT_DIST_TEMPLATE, filename='all_vertex_distances.npy')





    # paw vertices:
    # left and right is a bit different, but that is ok (we will anyways mirror data at training time)
    right_front_paw = [3829,+3827,+3825,+3718,+3722,+3723,+3743,+3831,+3719,+3726,+3716,+3724,+3828,+3717,+3721,+3725,+3832,+3830,+3720,+3288,+3740,+3714,+3826,+3715,+3728,+3712,+3287,+3284,+3727,+3285,+3742,+3291,+3710,+3697,+3711,+3289,+3730,+3713,+3739,+3282,+3738,+3708,+3709,+3741,+3698,+3696,+3308,+3695,+3706,+3700,+3707,+3306,+3305,+3737,+3304,+3303,+3307,+3736,+3735,+3250,+3261,+3732,+3734,+3733,+3731,+3729,+3299,+3297,+3298,+3295,+3293,+3296,+3294,+3292,+3312,+3311,+3314,+3309,+3290,+3313,+3410,+3315,+3411,+3412,+3316,+3421,+3317,+3415,+3445,+3327,+3328,+3283,+3343,+3326,+3325,+3330,+3286,+3399,+3398,+3329,+3446,+3400,+3331,+3401,+3281,+3332,+3279,+3402,+3419,+3407,+3356,+3358,+3357,+3280,+3354,+3277,+3278,+3346,+3347,+3377,+3378,+3345,+3386,+3379,+3348,+3384,+3418,+3372,+3276,+3275,+3374,+3274,+3373,+3375,+3369,+3371,+3376,+3273,+3396,+3397,+3395,+3388,+3360,+3370,+3361,+3394,+3387,+3420,+3359,+3389,+3272,+3391,+3393,+3390,+3392,+3363,+3362,+3367,+3365,+3705,+3271,+3704,+3703,+3270,+3269,+3702,+3268,+3224,+3267,+3701,+3225,+3699,+3265,+3264,+3266,+3263,+3262,+3249,+3228,+3230,+3251,+3301,+3300,+3302,+3252]
    right_back_paw = [3472,+3627,+3470,+3469,+3471,+3473,+3626,+3625,+3475,+3655,+3519,+3468,+3629,+3466,+3476,+3624,+3521,+3654,+3657,+3838,+3518,+3653,+3839,+3553,+3474,+3516,+3656,+3628,+3834,+3535,+3630,+3658,+3477,+3520,+3517,+3595,+3522,+3597,+3596,+3501,+3534,+3503,+3478,+3500,+3479,+3502,+3607,+3499,+3608,+3496,+3605,+3609,+3504,+3606,+3642,+3614,+3498,+3480,+3631,+3610,+3613,+3506,+3659,+3660,+3632,+3841,+3661,+3836,+3662,+3633,+3663,+3664,+3634,+3635,+3486,+3665,+3636,+3637,+3666,+3490,+3837,+3667,+3493,+3638,+3492,+3495,+3616,+3644,+3494,+3835,+3643,+3833,+3840,+3615,+3650,+3668,+3652,+3651,+3645,+3646,+3647,+3649,+3648,+3622,+3617,+3448,+3621,+3618,+3623,+3462,+3464,+3460,+3620,+3458,+3461,+3463,+3465,+3573,+3571,+3467,+3569,+3557,+3558,+3572,+3570,+3556,+3585,+3593,+3594,+3459,+3566,+3592,+3567,+3568,+3538,+3539,+3555,+3537,+3536,+3554,+3575,+3574,+3583,+3541,+3550,+3576,+3581,+3639,+3577,+3551,+3582,+3580,+3552,+3578,+3542,+3549,+3579,+3523,+3526,+3598,+3525,+3600,+3640,+3599,+3601,+3602,+3603,+3529,+3604,+3530,+3533,+3532,+3611,+3612,+3482,+3481,+3505,+3452,+3455,+3456,+3454,+3457,+3619,+3451,+3450,+3449,+3591,+3589,+3641,+3584,+3561,+3587,+3559,+3488,+3484,+3483]
    left_front_paw = [1791,+1950,+1948,+1790,+1789,+1746,+1788,+1747,+1949,+1944,+1792,+1945,+1356,+1775,+1759,+1777,+1787,+1946,+1757,+1761,+1745,+1943,+1947,+1744,+1309,+1786,+1771,+1354,+1774,+1765,+1767,+1768,+1772,+1763,+1770,+1773,+1769,+1764,+1766,+1758,+1760,+1762,+1336,+1333,+1330,+1325,+1756,+1323,+1755,+1753,+1749,+1754,+1751,+1321,+1752,+1748,+1750,+1312,+1319,+1315,+1313,+1317,+1318,+1316,+1314,+1311,+1310,+1299,+1276,+1355,+1297,+1353,+1298,+1300,+1352,+1351,+1785,+1784,+1349,+1783,+1782,+1781,+1780,+1779,+1778,+1776,+1343,+1341,+1344,+1339,+1342,+1340,+1360,+1335,+1338,+1362,+1357,+1361,+1363,+1458,+1337,+1459,+1456,+1460,+1493,+1332,+1375,+1376,+1331,+1374,+1378,+1334,+1373,+1494,+1377,+1446,+1448,+1379,+1449,+1329,+1327,+1404,+1406,+1405,+1402,+1328,+1426,+1432,+1434,+1403,+1394,+1395,+1433,+1425,+1286,+1380,+1466,+1431,+1290,+1401,+1381,+1427,+1450,+1393,+1430,+1326,+1396,+1428,+1397,+1429,+1398,+1420,+1324,+1422,+1417,+1419,+1421,+1443,+1418,+1423,+1444,+1442,+1424,+1445,+1495,+1440,+1441,+1468,+1436,+1408,+1322,+1435,+1415,+1439,+1409,+1283,+1438,+1416,+1407,+1437,+1411,+1413,+1414,+1320,+1273,+1272,+1278,+1469,+1463,+1457,+1358,+1464,+1465,+1359,+1372,+1391,+1390,+1455,+1447,+1454,+1467,+1453,+1452,+1451,+1383,+1345,+1347,+1348,+1350,+1364,+1392,+1410,+1412]
    left_back_paw = [1957,+1958,+1701,+1956,+1951,+1703,+1715,+1702,+1700,+1673,+1705,+1952,+1955,+1674,+1699,+1675,+1953,+1704,+1954,+1698,+1677,+1671,+1672,+1714,+1706,+1676,+1519,+1523,+1686,+1713,+1692,+1685,+1543,+1664,+1712,+1691,+1959,+1541,+1684,+1542,+1496,+1663,+1540,+1497,+1499,+1498,+1500,+1693,+1665,+1694,+1716,+1666,+1695,+1501,+1502,+1696,+1667,+1503,+1697,+1504,+1668,+1669,+1506,+1670,+1508,+1510,+1507,+1509,+1511,+1512,+1621,+1606,+1619,+1605,+1513,+1620,+1618,+1604,+1633,+1641,+1642,+1607,+1617,+1514,+1632,+1614,+1689,+1640,+1515,+1586,+1616,+1516,+1517,+1603,+1615,+1639,+1585,+1521,+1602,+1587,+1584,+1601,+1623,+1622,+1631,+1598,+1624,+1629,+1589,+1687,+1625,+1599,+1630,+1569,+1570,+1628,+1626,+1597,+1627,+1590,+1594,+1571,+1568,+1567,+1574,+1646,+1573,+1645,+1648,+1564,+1688,+1647,+1643,+1649,+1650,+1651,+1577,+1644,+1565,+1652,+1566,+1578,+1518,+1524,+1583,+1582,+1520,+1581,+1522,+1525,+1549,+1551,+1580,+1552,+1550,+1656,+1658,+1554,+1657,+1659,+1548,+1655,+1690,+1660,+1556,+1653,+1558,+1661,+1544,+1662,+1654,+1547,+1545,+1527,+1560,+1526,+1678,+1679,+1528,+1708,+1707,+1680,+1529,+1530,+1709,+1546,+1681,+1710,+1711,+1682,+1532,+1531,+1683,+1534,+1533,+1536,+1538,+1600,+1553]





    all_keys = []
    gc_dict = {}
    vertex_overview_nocontact = {}
    # data/stanext_related_data/ground_contact_annotations/stage3/main_partA1667_20221021_140108.csv
    for csv_file in ['Stage2b_finalResults.csv']:
        # load all ground contact annotations
        gc_annot_csv = ROOT_PATH_ANNOT + csv_file   # 'my_gcannotations_qualification.csv'
        gc_row_list = read_csv(gc_annot_csv)
        all_annotations, all_comments = summarize_results_stage2b(gc_row_list, display_worker_performance=False)
        for key, value in all_annotations.items():
            if value['not_sure'] == '0':
                if value['no_contact'] == '1':
                    vertex_overview_nocontact[key.split('.')[0]] = {'gc_vertdists_overview': 'no contact', 'gc_index_list': None}
                else:
                    all_contact_vertices = []
                    if value['paw_rf'] == '1':
                        all_contact_vertices.extend(right_front_paw)
                    if value['paw_rb'] == '1':
                        all_contact_vertices.extend(right_back_paw)
                    if value['paw_lf'] == '1':
                        all_contact_vertices.extend(left_front_paw)
                    if value['paw_lb'] == '1':
                        all_contact_vertices.extend(left_back_paw)
                    gc_dict[key] = all_contact_vertices
    print('number of labeled images: ' + str(len(gc_dict.keys())))     
    print('number of images without contact: ' + str(len(vertex_overview_nocontact.keys())))      

    # prepare and save contact annotations including distances
    vertex_overview_dict = {}
    for ind_img, name_ingcdict in enumerate(gc_dict.keys()):     #  range(len(gc_dict.keys())):
        name = name_ingcdict      # name_ingcdict.split('bite/')[1]
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

    with open(ROOT_OUT_PATH + 'gc_annots_overview_stage2b_contact_complete_xx.pkl', 'wb') as fp:
        pkl.dump(vertex_overview_dict, fp)

    with open(ROOT_OUT_PATH + 'gc_annots_overview_stage2b_nocontact_complete_xx.pkl', 'wb') as fp:
        pkl.dump(vertex_overview_nocontact, fp)











if __name__ == "__main__":
    main()







