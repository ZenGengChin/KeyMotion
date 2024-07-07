from body_visualizer.tools.vis_tools import imagearray2file
from torch import Tensor
import numpy as np
from body.body_model import BodyModel
import cv2

from utils.pose_utils import render_smpl_params_hw
from utils.pose_utils import rotation_6d_to_axis_angle


def get_smpl_render_img(amass_pose:np.ndarray, 
                        amass_trans:np.ndarray,
                        bm_fname:str='body/body_models/smplh/neutral/model.npz',
                        device:str='cuda',
                        imh:int=400, 
                        imw:int=400):
    """_summary_

    Args:
        amass_pose (tensor): amass_pose data, np.darray in the shape of 
            [B, J*3]. J = 22 for body pose, and can add hand if J = 52.
        amass_trans (tensor): amass translation data. in the shape of [B, 3]
        bm_fname (str, optional): SMPL body model.
            Defaults to 'body/body_models/smplh/neutral/model.npz'.
        device (str, optional): device. Defaults to 'cuda'.
        imh (int, optional): rendered image height. Defaults to 400.
        imw (int, optional): rendered iamge width. Defaults to 400.
    
    Returns:
        img: a list of np.array. each array within is shaped by [h, w, 3]
    """
    bm = BodyModel(bm_fname=bm_fname).to(device)
    
    if amass_pose.shape[-1] < 156:
        pose_hand = None
    else:
        pose_hand = amass_pose[:, 66:]
    
    images = render_smpl_params_hw(
        bm=bm,body_parms={
        'root_orient':amass_pose[:, 0:3],
        'pose_body':amass_pose[0:len(amass_pose), 3:66],
        'pose_hand': pose_hand
        #'betas':beta
        },
        trans=amass_trans,
        imw=imw, imh=imh).reshape(1,1,len(amass_pose),imh,imw,3)

    img = imagearray2file(images)
    return img


def show_multi_image(img_group, rows:int, cols:int,interval=1/30):
    """show multiple image sequences as video, with rows and cols.
    Args:
        img_group (list): list of [nparray] or list of list of nparray,
            whose shape should be [rows*cols, L, [H, W, 3]]
        rows (int): row number
        cols (int): colum number
        interval (float): 1/fps. Defaults to 1/30.

    * Caution, before you put img_group make sure that the L is of the same size.
    
    Return:
        None: display animation in a cv2 canvas.
    """
    if len(img_group) != rows * cols:
            raise ValueError("Number of images must match row and col.")

    height, width = img_group[0][0].shape[:2]
    
    canvas = np.zeros((height*rows,width*cols, 3), dtype=np.uint8)
    
    while True:
        for i in range(len(img_group[0])):
            for r in range(rows):
                for c in range(cols):
                    img = img_group[r*cols + c][i]            
                    img = img.astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    canvas[r*height:(r+1)*height, c*width:(c+1)*width] = img
            cv2.imshow('plot', canvas)
            cv2.waitKey(int(interval * 1000))
        cv2.destroyAllWindows()

def visualize_motion_batch(batch:Tensor, 
                           rows:int, 
                           cols:int, 
                           interval=1/30,
                           bm_fname:str='body/body_models/smplh/neutral/model.npz',
                           device:str='cuda',
                           rep = '6d'):
    """
    visualize motion batch for 6d

    Args:
        batch (Tensor): motion 6d rep, shape [B, L, 135] 135 = 3 + 6 * 22
        rows (int): video rows number
        cols (int): video cols number
        interval (float, optional):. Defaults to 1/30.
        bm_fname (str, optional): Defaults to 'body/body_models/smplh/neutral/model.npz'.
        device (str, optional): Defaults to 'cuda'.
    """
    motion_list = []
    canvas_height = 900
    canvas_width = 1200
    B, L, E = batch.shape
    for i in range(batch.shape[0]):
        trans = batch[i,:,0:3]
        if rep == '6d':
            poses_6d = batch[i, :, 3:]
            poses_6d = poses_6d.reshape((L,22,6))
            poses = rotation_6d_to_axis_angle(poses_6d)
            poses = poses.reshape((L, 22*3))
            if trans.device != device:
                trans = trans.to(device)
            if poses.device != device:
                poses = poses.to(device)
        elif rep == 'aa':
            poses = batch[i,:,3:]
            poses = poses.reshape((L, 22*3))
            if trans.device != device:
                trans = trans.to(device)
            if poses.device != device:
                poses = poses.to(device)
        rendered_img = get_smpl_render_img(
            amass_pose=poses,
            amass_trans=trans,
            bm_fname=bm_fname,
            device=device,
            imh= canvas_height // rows,
            imw= canvas_width // cols
        )
        motion_list.append(rendered_img)
    show_multi_image(motion_list,rows=rows,cols=cols,interval=interval)



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_point_scatter_animation(animation_data, output_file):
    L, J, _ = animation_data.shape

    # Extract X, Y, and Z coordinates for each animation frame
    X = animation_data[:, :, 0]
    Y = animation_data[:, :, 1]
    Z = animation_data[:, :, 2]

    def update_plot(frame):
        ax.clear()
        ax.scatter(X[frame], Y[frame], Z[frame], c='b', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'Frame: {frame}')
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim3d((0,2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_frames = L
    ani = FuncAnimation(fig, update_plot, frames=num_frames, interval=100)

    # Save the animation as a video file
    ani.save(output_file, writer='ffmpeg')