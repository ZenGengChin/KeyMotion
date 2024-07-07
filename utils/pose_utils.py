import torch 

import numpy as np


from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d

from utils.skeleton import Skeleton
from utils.quaternion import qinv, qmul, qrot, quaternion_to_cont6d

SMPL_K_TREE = [[0, 2, 5, 8, 11],  # right leg
                [0, 1, 4, 7, 10],   # left leg
                [0, 3, 6, 9, 12, 15],  # torso
                [9, 14, 17, 19, 21],   # right arm
                [9, 13, 16,  18, 20]]  # left arm

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

# This version has only 22 joints

SMPL_K_ANIME_TREE = [[0, 1, 2, 3, 4],   # left leg
                [0, 5, 6, 7, 8],  # right leg
                [0, 9, 10, 11, 12, 13],  # torso
                [11, 14, 15,  16, 17, 18],  # left arm
                [11, 19, 20, 21, 22, 23 ]]   # right arm
# This version has only 24 joints starts from Pelvis


t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])
COLORS = {
    'pink': [.6, .0, .4],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .2, .1],
    'brown-light': [0.654, 0.396, 0.164],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [1., .2, 0],

    'grey': [.7, .7, .7],
    'grey-blue': [0.345, 0.580, 0.713],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}

SMPL_SKEL = [[0,1], [0,2], [0,3],
             [1,4], [2,5], [3,6], [4,7],
             [5,8], [6,9], [7,10], [8,11],
             [9,12],[9,13], [9,14],[12,15],
             [13,16],[14,17],[16,18],[17,19],
             [18,20],[19,21],[20,22],[21,23]]

def render_smpl_params_hw(bm, body_parms, trans=None, rot_body=None, imw=400, imh=400, 
                          cam_t=[0, 1.2, 7], cam_r=[0, 0, 0]):
    '''
    Modified from render_smpl_params with flexible hw.
    :param bm: pytorch body model with batch_size 1
    :param pose_body: Nx21x3
    :param trans: Nx3
    :param betas: Nxnum_betas
    :return: N x 400 x 400 x 3
    '''
    import trimesh
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from body_visualizer.tools.mesh_tools import rotateXYZ
    from utils.meshviewer import AdjustableMeshViewer as AMV

    mv = AMV(width=imw, height=imh, yfov=np.pi / 5,use_offscreen=True)
    mv.set_cam_trans(cam_t, cam_r)
    faces = c2c(bm.f)
    
    v = c2c(bm(**body_parms).v)
    trans = trans.reshape((-1,1,3)).cpu().detach().numpy()
    v = v + trans
    
    
    T, num_verts = v.shape[:-1]
    images = []
    for fIdx in range(T):
        verts = v[fIdx]
        if rot_body is not None:
            verts = rotateXYZ(verts, rot_body)
        mesh = trimesh.base.Trimesh(verts, faces, vertex_colors=num_verts*COLORS['red'])
                                    #face_colors=faces.shape[-1]*COLORS['red'])

        mv.set_meshes([mesh], 'static')

        images.append(mv.render())

    return np.array(images).reshape(T, imw, imh, 3)

def rotation_6d_to_axis_angle(sixd:torch.Tensor):
    """ transform rotation 6d to axis angle representation.
    Args:
        sixd (torch.Tensor): [B, L, 6d]

    Returns:
        torch.Tensor: [B, L, 3] 
    """
    return matrix_to_axis_angle(rotation_6d_to_matrix(sixd))

def rotation_6d_to_quaternion(sixd: torch.Tensor):
    """ transform rotation 6d to axis angle representation.
    Args:
        sixd (torch.Tensor): [B, L, 6d]

    Returns:
        torch.Tensor: [B, L, 4]
    """
    return matrix_to_quaternion(rotation_6d_to_matrix(sixd))


def humanml_to_quaternion(humanml_motion: torch.Tensor):
    """
    for 
    Args:
        humanml_motion (torch.Tensor): [B,L,263]
    """
    njoints = 22 # fixed
    


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions