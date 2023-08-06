
import torch
import numpy as np


'''
def keyp_rep_error_l1(smpl_keyp_2d, keyp_hourglass, keyp_hourglass_scores, thr_kp=0.3):
    # step 1: make sure that the hg prediction and barc are close
    with torch.no_grad():
        kp_weights = keyp_hourglass_scores
        kp_weights[keyp_hourglass_scores<thr_kp] = 0
    loss_keyp_rep = torch.mean((torch.abs((smpl_keyp_2d - keyp_hourglass)/512)).sum(dim=2)*kp_weights[:, :, 0])
    return loss_keyp_rep

def keyp_rep_error(smpl_keyp_2d, keyp_hourglass, keyp_hourglass_scores, thr_kp=0.3):
    # step 1: make sure that the hg prediction and barc are close
    with torch.no_grad():
        kp_weights = keyp_hourglass_scores
        kp_weights[keyp_hourglass_scores<thr_kp] = 0
    # losses['kp_reproj']['value'] = torch.mean((((smpl_keyp_2d - keyp_reproj_init)/512)**2).sum(dim=2)*kp_weights[:, :, 0])
    loss_keyp_rep = torch.mean((((smpl_keyp_2d - keyp_hourglass)/512)**2).sum(dim=2)*kp_weights[:, :, 0])
    return loss_keyp_rep
'''

def leg_sideway_error(optimed_pose_with_glob):
    assert optimed_pose_with_glob.shape[1] == 35
    leg_indices_right = np.asarray([7, 8, 9, 10, 17, 18, 19, 20])      # front, back
    leg_indices_left = np.asarray([11, 12, 13, 14, 21, 22, 23, 24])     # front, back
    # leg_indices_right = np.asarray([8, 9, 10, 18, 19, 20])      # front, back
    # leg_indices_left = np.asarray([12, 13, 14, 22, 23, 24])     # front, back
    x0_rotmat = optimed_pose_with_glob   # (1, 35, 3, 3)
    x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
    x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]
    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec[2] = -1
    x0_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3))@vec
    x0_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3))@vec
    loss_pose_legs_side = (x0_legs_left[:, 1]**2).mean() + (x0_legs_right[:, 1]**2).mean()
    return loss_pose_legs_side


def leg_torsion_error(optimed_pose_with_glob):
    leg_indices_right = np.asarray([7, 8, 9, 10, 17, 18, 19, 20])      # front, back
    leg_indices_left = np.asarray([11, 12, 13, 14, 21, 22, 23, 24])     # front, back
    x0_rotmat = optimed_pose_with_glob   # (1, 35, 3, 3)
    x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
    x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]
    vec_x = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec_x[0] = 1      # in x direction
    x_x_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3))@vec_x
    x_x_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3))@vec_x
    loss_pose_legs_torsion = (x_x_legs_left[:, 1]**2).mean() + (x_x_legs_right[:, 1]**2).mean()
    return loss_pose_legs_torsion


def frontleg_walkingdir_error(optimed_pose_with_glob):
    # this prior should only be used for standing poses!
    leg_indices_right = np.asarray([7, 8, 9, 10])      # front, back
    leg_indices_left = np.asarray([11, 12, 13, 14])     # front, back
    relevant_back_indices = np.asarray([1, 2, 3, 4, 5, 6])      # np.asarray([6])             # back joint in the front
    x0_rotmat = optimed_pose_with_glob   # (1, 35, 3, 3)
    x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
    x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]
    x0_rotmat_back = x0_rotmat[:, relevant_back_indices, :, :]
    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec[2] = -1     # vector down
    x0_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3))@vec
    x0_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3))@vec
    x0_back = x0_rotmat_back.reshape((-1, 3, 3))@vec
    loss_pose_legs_side = (x0_legs_left[:, 0]**2).mean() + (x0_legs_right[:, 0]**2).mean() + (x0_back[:, 0]**2).mean()  # penalize movement to front
    return loss_pose_legs_side


def tail_sideway_error(optimed_pose_with_glob):
    tail_indices = np.asarray([25, 26, 27, 28, 29, 30, 31])      
    x0_rotmat = optimed_pose_with_glob   # (1, 35, 3, 3)
    x0_rotmat_tail = x0_rotmat[:, tail_indices, :, :]
    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    '''vec[2] = -1    
    x0_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec
    loss_pose_tail_side = (x0_tail[:, 1]**2).mean()'''
    vec[0] = -1    
    x0_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec
    loss_pose_tail_side = (x0_tail[:, 1]**2).mean()
    return loss_pose_tail_side


def tail_torsion_error(optimed_pose_with_glob):
    tail_indices = np.asarray([25, 26, 27, 28, 29, 30, 31])      
    x0_rotmat = optimed_pose_with_glob   # (1, 35, 3, 3)
    x0_rotmat_tail = x0_rotmat[:, tail_indices, :, :]
    vec_x = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    '''vec_x[0] = 1      # in x direction
    x_x_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec_x
    loss_pose_tail_torsion = (x_x_tail[:, 1]**2).mean()'''
    vec_x[2] = 1      # in y direction
    x_x_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec_x
    loss_pose_tail_torsion = (x_x_tail[:, 1]**2).mean()
    return loss_pose_tail_torsion


def spine_sideway_error(optimed_pose_with_glob):
    tail_indices = np.asarray([1, 2, 3, 4, 5, 6])   # was wrong      
    x0_rotmat = optimed_pose_with_glob   # (1, 35, 3, 3)
    x0_rotmat_tail = x0_rotmat[:, tail_indices, :, :]
    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec[0] = -1    
    x0_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec
    loss_pose_tail_side = (x0_tail[:, 1]**2).mean()
    return loss_pose_tail_side


def spine_torsion_error(optimed_pose_with_glob):
    tail_indices = np.asarray([1, 2, 3, 4, 5, 6])      
    x0_rotmat = optimed_pose_with_glob   # (1, 35, 3, 3)
    x0_rotmat_tail = x0_rotmat[:, tail_indices, :, :]
    vec_x = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec_x[2] = 1    # vec_x[0] = 1      # in z direction
    x_x_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec_x
    loss_pose_tail_torsion = (x_x_tail[:, 1]**2).mean()     # (x_x_tail[:, 1]**2).mean()
    return loss_pose_tail_torsion


def fit_plane(points_npx3):
    # remarks:
    #   visualization of the plane: debug_code/curve_fitting_v2.py
    #   theory: https://www.ltu.se/cms_fs/1.51590!/svd-fitting.pdf
    #   remark: torch.svd is depreciated
    # new plane equation:
    #   a(x−x0)+b(y−y0)+c(z−z0)=0
    #   ax+by+cz=d with  d=ax0+by0+cz0
    #   z = (d-ax-by)/c
    #   here:
    #   a, b, c describe the plane normal 
    #   d can be calculated (from a, b, c, x0, y0, z0)
    #   (x0, y0, z0) are the coordinates of a point on the 
    #     plane, for example points_centroid
    #   (x, y, z) are the coordinates of a query point on the plane 
    # 
    # points_npx3: (n_points, 3) 
    # REMARK: this loss is not yet for batches!
    # import pdb; pdb.set_trace()
    # print('this loss is not yet for batches!')
    assert (points_npx3.ndim == 2)
    assert (points_npx3.shape[1] == 3)
    points = torch.transpose(points_npx3, 0, 1)       # (3, n_points)
    points_centroid = torch.mean(points, dim=1)
    input_svd = points - points_centroid[:, None] 
    U_svd, sigma_svd, V_svd = torch.svd(input_svd, compute_uv=True)
    plane_normal = U_svd[:, 2]
    plane_squaredsumofdists = sigma_svd[2]
    error = plane_squaredsumofdists
    return points_centroid, plane_normal, error


def paws_to_groundplane_error(vertices, return_details=False):
    # list of feet vertices (some of them)
    #   remark: we did annotate left indices and find the right insices using sym_ids_dict
    # REMARK: this loss is not yet for batches!
    # import pdb; pdb.set_trace()
    # print('this loss is not yet for batches!')
    list_back_left = [1524, 1517, 1512, 1671, 1678, 1664, 1956, 1680, 1685, 1602, 1953, 1569]
    list_front_left = [1331, 1327, 1332, 1764, 1767, 1747, 1779, 1789, 1944, 1339, 1323, 1420]
    list_back_right = [3476, 3469, 3464, 3623, 3630, 3616, 3838, 3632, 3637, 3554, 3835, 3521]
    list_front_right = [3283, 3279, 3284, 3715, 3718, 3698, 3730, 3740, 3826, 3291, 3275, 3372]
    assert vertices.shape[0] == 3889
    assert vertices.shape[1] == 3
    all_paw_vert_idxs = list_back_left + list_front_left + list_back_right + list_front_right
    verts_paws = vertices[all_paw_vert_idxs, :]
    plane_centroid, plane_normal, error = fit_plane(verts_paws)
    if return_details:
        return plane_centroid, plane_normal, error
    else:
        return error

def groundcontact_error(vertices, gclabels, return_details=False):
    # import pdb; pdb.set_trace()
    # REMARK: this loss is not yet for batches!
    import pdb; pdb.set_trace()
    print('this loss is not yet for batches!')
    assert vertices.shape[0] == 3889
    assert vertices.shape[1] == 3
    verts_gc = vertices[gclabels, :]
    plane_centroid, plane_normal, error = fit_plane(verts_gc)
    if return_details:
        return plane_centroid, plane_normal, error
    else:
        return error



