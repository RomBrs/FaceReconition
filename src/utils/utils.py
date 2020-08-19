import numpy as np
import glob
import os

def convert_coords(kp_tmp, faces, size = [112,112]):
    """
    in:
        @kp_tmp - keypoints of face on MTCNN input image 
        @faces - bboxes of faces on MTCNN input image
        @size - what face size will be used in face recognition

    out:
        @kp - keypoints projected on resized face 
    """
    kp = kp_tmp.copy()
    kp[:,:,0] = kp[:,:,0] - faces[:,0][:,None]
    kp[:,:,1] = kp[:,:,1] - faces[:,1][:,None]
    w = faces[:,2] - faces[:,0]
    h = faces[:,3] - faces[:,1]
    scale_w = w / size[0]
    scale_h = h / size[1]
    kp = kp/ np.concatenate( [scale_w[:,None], scale_h[:,None]] , axis = 1)[:,None,:]
    return kp


def is_position_bad(yaw_pitch_roll):
    if yaw_pitch_roll[0] >= 30 or yaw_pitch_roll[0] <= -30 or \
        yaw_pitch_roll[1] >= 30 or yaw_pitch_roll[1] <= -30 or \
        yaw_pitch_roll[2] >= 30 or yaw_pitch_roll[2] <= -30:
        return True
    else:
        return False


def clear_directory(dir):
    files = glob.glob( os.path.join(dir, "*") )
    for f in files:
        os.remove(f)