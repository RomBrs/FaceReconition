
import cv2
from facenet_pytorch import MTCNN
import numpy as np


class DetectorMTCNN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model =  MTCNN(keep_all=True, device=self.cfg.DEVICE)
        
    def _preprocess_img(self, image, input_img_type):
        
        def _pad_if_needed(img, size):
            h,w = img.shape[:2]
            pad_h = int( (size - h) / 2 )
            pad_w = int( (size - w) / 2 )
            img_out = np.zeros( (size, size, 3) , dtype=np.uint8 )
            img_out[pad_h: pad_h + h, pad_w: pad_w + w, :] = img
            return img_out
        
        if input_img_type == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pad = _pad_if_needed(image, max(image.shape) )
        image_resized = cv2.resize(image_pad, self.cfg.MTCNN.TRANSFORM.SIZE)
        return image_resized
    
    def __call__(self, image, input_img_type = "BGR", prob_cutoff = 0.8):
        image = self._preprocess_img(image, input_img_type)
        bounding_boxes, probabilites , points = self.model.detect( image, landmarks=True)

        # index = np.where( probabilites > prob_cutoff)
        # probabilites = probabilites[index]
        # bounding_boxes = bounding_boxes[index]
        # points = points[index]

        return image, bounding_boxes, probabilites , points
        

