import insightface
import numpy as np
import cv2
import torch
from skimage import transform as trans

import albumentations as albu
from albumentations import Compose
from albumentations.pytorch import ToTensor

class Embedding:
    def __init__(self, model_type, cfg):
        self.cfg = cfg
        if model_type == 'iresnet34':
            self.model = insightface.iresnet34(pretrained = True).to(self.cfg.DEVICE)

        elif model_type == 'iresnet50':
            self.model = insightface.iresnet50(pretrained = False).to(self.cfg.DEVICE)

        elif model_type == 'iresnet100':
            self.model = insightface.iresnet100(pretrained = False).to(self.cfg.DEVICE)
            
        self.model.eval()
        self.transform = self.transform()
    
    def transform(self):
        return Compose([
            albu.LongestMaxSize(np.max(self.cfg.INSIGHTFACE.PREPROCESS.IMAGE_SIZE),
                                interpolation=cv2.INTER_LINEAR, always_apply=False, p=1),
            albu.PadIfNeeded(min_height=self.cfg.INSIGHTFACE.PREPROCESS.IMAGE_SIZE[0],
                             min_width=self.cfg.INSIGHTFACE.PREPROCESS.IMAGE_SIZE[1], 
                             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=False, p=1.0),
            ToTensor(num_classes=1, sigmoid=False, normalize={'mean': self.cfg.INSIGHTFACE.PREPROCESS.MEAN,
                                                              'std': self.cfg.INSIGHTFACE.PREPROCESS.STD})
        ])
        
    def _preprocess(self, img, bbox=None, landmark=None, **kwargs):
      if isinstance(img, str):
        img = read_image(img, **kwargs)
      M = None
      image_size = []
      str_image_size = kwargs.get('image_size', '')
      if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
          image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2
        assert image_size[0]==112
        assert image_size[0]==112 or image_size[1]==96
      if landmark is not None:
        assert len(image_size)==2
        src = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
          src[:,0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

      if M is None:
        if bbox is None: #use center crop
          det = np.zeros(4, dtype=np.int32)
          det[0] = int(img.shape[1]*0.0625)
          det[1] = int(img.shape[0]*0.0625)
          det[2] = img.shape[1] - det[0]
          det[3] = img.shape[0] - det[1]
        else:
          det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
          ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret 
      else: #do align using landmark
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped
        
    def __call__(self, image, bounding_boxes, probabilites , points):
        output = {}
        for idx in range(probabilites.shape[0]):
            warped = self._preprocess(image, bbox=bounding_boxes[idx], landmark = points[idx], 
                                image_size=self.cfg.INSIGHTFACE.PREPROCESS.SIZE)
            transformed = self.transform(image = warped)["image"] 
            with torch.no_grad():
                features =self.model( transformed.unsqueeze(0).to(self.cfg.DEVICE) )
            output[idx] = (warped, features.cpu().numpy() )
        return output
        