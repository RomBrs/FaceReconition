import torch 
from numba import jit
import numpy as np


class HeadPositionDetector:
    def __init__(self,  cfg):
        self.cfg = cfg
        self.model = torch.jit.load(self.cfg.HEAD_POSITION.MODEL_FPATH, self.cfg.DEVICE)

    # @jit(nopython=True)
    def _features_from_kp(self, kp):
        out = []
        for p1 in range(kp.shape[0]):
            for p2 in range(p1+1,kp.shape[0]):
                out.extend(kp[p1] - kp[p2])
                out.append( np.sqrt(((kp[p1] - kp[p2])**2).sum()) )
        return out

    def get_yaw_pitch_roll(self, mtcnn_kp):
        features = []
        for i in mtcnn_kp:
            features.append( self._features_from_kp( i ) )
        output = self.model( torch.Tensor( features ).to(self.cfg.DEVICE) )
        return output.cpu().detach().numpy()



