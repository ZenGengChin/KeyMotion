# from model.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl

#  class npy2obj:
#     def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=False):
#         self.npy_path = npy_path
#         self.motions = np.load(self.npy_path, allow_pickle=True)
#         if self.npy_path.endswith('.npz'):
#             self.motions = self.motions['arr_0']
#         self.motions = self.motions[None][0]
#         self.rot2xyz = Rotation2xyz(device='cpu')
#         self.faces = self.rot2xyz.smpl_model.faces
#         self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
#         self.opt_cache = {}
#         self.sample_idx = sample_idx
#         self.total_num_samples = self.motions['num_samples']
#         self.rep_idx = rep_idx
#         self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
#         self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
#         self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=None)

#         if self.nfeats == 3:
#             print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
#             motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
#             self.motions['motion'] = motion_tensor.cpu().numpy()
#         elif self.nfeats == 6:
#             self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
#         self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
#         self.real_num_frames = self.motions['lengths'][self.absl_idx]

#         self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
#                                      pose_rep='rot6d', translation=True, glob=True,
#                                      jointstype='vertices',
#                                      # jointstype='smpl',  # for joint locations
#                                      vertstrans=True)
#         self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
#         self.vertices += self.root_loc

#     def get_vertices(self, sample_i, frame_i):
#         return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

#     def get_trimesh(self, sample_i, frame_i):
#         return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
#                        faces=self.faces)

#     def save_obj(self, save_path, frame_i):
#         mesh = self.get_trimesh(0, frame_i)
#         with open(save_path, 'w') as fw:
#             mesh.export(fw, 'obj')
#         return save_path
    
#     def save_npy(self, save_path):
#         data_dict = {
#             'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
#             'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
#             'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
#             'faces': self.faces,
#             'vertices': self.vertices[0, :, :, :self.real_num_frames],
#             'text': self.motions['text'][0],
#             'length': self.real_num_frames,
#         }
#         np.save(save_path, data_dict)


from torch import Tensor
from body.body_model import BodyModel
from utils.pose_utils import recover_from_ric
from utils.pose_utils import rotation_6d_to_axis_angle
import trimesh
import os


class feats2obj(object):
   def __init__(self, 
                obj_path: str = 'obj/',
                nfeats: int = 263,
                njoints: int = 22,
                sexual:str = 'female', # female male, neural
                nframe:int = 196,
                device:str = 'cuda'
                ) -> None:
       """ Redundant features to object file. 

       Args:
           features (Tensor): _description_
           sub_path (str): _description_
           obj_path (str, optional): _description_. Defaults to 'obj'.
       """
       self.sexual = sexual
       if sexual == 'neutral':
           self.bm_fname='body/body_models/smplh/neutral/model.npz'
       elif sexual == 'female':
           self.bm_fname='body/body_models/smplh/female/model.npz'
       elif sexual == 'neutral':
           self.bm_fname='body/body_models/smplh/male/model.npz'
           
       self.objpath = obj_path
       self.nfeats = nfeats
       self.njoints = njoints
       self.device = device        
       self.bm = BodyModel(bm_fname=self.bm_fname).to(device)
       

       
   def generate_obj(self, features: Tensor, path:str, text:list):
       """
       Args:
           features (Tensor): [B, L, E]
           path (str): path in the obj path
       """
       self.j2s = joints2smpl(num_frames=features.shape[0]*features.shape[1],
                              device_id=0,
                              cuda=True)
       
       #if not os.path.exists(os.path.join(self.objpath, path)):
       #    os.mkdir(os.path.join(self.objpath, path))
       
       B, L, _ = features.shape
       joints = recover_from_ric(features, joints_num=self.njoints)
       joints = joints.reshape((B*L, -1, 3))
       #print(joints.shape, 'joints')
       
       smpl_input = self.j2s.joint2smpl(input_joints=joints.cuda(), init_params=None)
       smpl_input = smpl_input.reshape((B, L, -1))
       #print(smpl_input.shape)
       # visualize_motion_batch(
       #  batch=smpl_input.unsqueeze(0), rows=1, cols=1, interval=1/20, device='cuda', rep='6d')
       trans = smpl_input[:,:,:3].reshape((-1,3)) # translation x, y, z
       poses_6d = smpl_input[:, :, 3:].reshape((B*L,self.njoints,6))
       poses_aa = rotation_6d_to_axis_angle(poses_6d).reshape((B*L, -1))
       
       vertices = self.bm.forward(
           root_orient=poses_aa[:,:3],
           trans=trans,
           pose_body=poses_aa[:,3:]
           ).v.reshape((B* L, -1,3))
       
       T, num_verts = vertices.shape[:-1]
       for id in range(B):
           if not os.path.exists(self.objpath+path+'_'+str(id)+'/'):
               os.mkdir(self.objpath+path+'_'+str(id)+'/')
           
           with open(self.objpath+path+'_'+str(id)+'/'+'text.txt', 'w') as tp:
               tp.write(text[id])
               
           for fIdx in range(L):
               verts = vertices[id*L + fIdx]
               verts = verts.detach().cpu().numpy()
               mesh = trimesh.base.Trimesh(verts, self.bm.f.detach().cpu().numpy(), vertex_colors=num_verts)

               with open(self.objpath+path+'_'+str(id)+'/'+str(fIdx)+'.obj', 'w') as fp:
                   mesh.export(fp, 'obj')
