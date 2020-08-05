
from models.detection import DetectorMTCNN
from models.recognition import Embedding

import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from config import _C  as cfg

# Video
cap = cv2.VideoCapture(0)

# Models (detection + recognition)
detector = DetectorMTCNN(cfg)
embedding = Embedding('iresnet34', cfg)

# Score base
base_imgs_path = "C:/Users/rusrob/Python_files/SAS/29_FAMILY/DemoIpy/base_embeddings/"
imgs_fpath = []
for dr in os.listdir(base_imgs_path):
    imgs_fpath.extend( glob.glob(os.path.join(base_imgs_path,dr) + "/*") )
    
base_emb = []
base_face = []
base_name = []
for im_path in tqdm(imgs_fpath):
    try:
        img = cv2.imread(im_path.replace("\\","/"))
        image, bounding_boxes, probabilites , points = detector(img, input_img_type = "BGR", prob_cutoff = 0.9)
        out = embedding( image, bounding_boxes, probabilites , points)
        for i in out:
            base_name.append( os.path.basename(os.path.dirname(im_path)) )
            base_emb.append( out[i][1] )
            base_face.append( out[i][0] )
    except Exception as e:
        print(e)
            
base_emb = np.array(base_emb).squeeze(1)
print("Base size: ", len(base_emb))

while True:
    _, frame = cap.read()
    
    # cv2.imwrite("roman.jpg", frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        prob_cutoff = 0.5
        image, bounding_boxes, probabilites , points = detector(frame, input_img_type = "BGR", prob_cutoff = prob_cutoff)
        out = embedding( image, bounding_boxes, probabilites , points)
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for idx in range(probabilites.shape[0]):

            # Face keypoints
            for point in points[idx]:
                x, y = point
                cv2.circle(rgb, (x, y), 4, (255, 0, 0), -1)
            
            # Face rectangle
            x1,y1,x2,y2 = bounding_boxes[idx]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Face probability        
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            color = (0, 255, 0)  
            thickness = 2
            rgb = cv2.putText(rgb, str(np.round(probabilites[idx],decimals=2)), (x1, y1), font,  
                            fontScale, color, thickness, cv2.LINE_AA) 
            
            # # Plot aligned face 
            for i in range(len(out)):
                size = 80
                rgb[0:size,
                    i*size:(i+1)*size, :] \
                     = cv2.cvtColor(cv2.resize(out[i][0], (size, size)), cv2.COLOR_RGB2BGR)

        # Find closest embeddings
        index_max = probabilites.argmax()
        emb = out[index_max][1]
        face = out[index_max][0]
        distances = np.sqrt(((emb - base_emb)**2).sum(1))
        distances_argsort = np.argsort(distances)

        # # Plot aligned face 
        for i in range(3):
            size = 80
            rgb[rgb.shape[0] - size: rgb.shape[0],
                i*size:(i+1)*size, :] \
                    = cv2.cvtColor(cv2.resize(base_face[distances_argsort[i]], (size, size)), cv2.COLOR_RGB2BGR)
            
            # Face probability        
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 255, 100)  
            thickness = 1
            rgb = cv2.putText(rgb, str(np.round(distances[distances_argsort[i]], decimals=2)),
                             ((i)*size, rgb.shape[0] - size), font,  
                            fontScale, color, thickness, cv2.LINE_AA) 
            rgb = cv2.putText(rgb,  base_name[distances_argsort[i]],
                             ((i)*size, rgb.shape[0] - size-15), font,  
                            fontScale, color, thickness, cv2.LINE_AA) 
                    
    except Exception as e:
        print(e)
        rgb = frame.copy()

    cv2.imshow("SAS", rgb)

    key = cv2.waitKey(1)
    if key == 27:
        break
