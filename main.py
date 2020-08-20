
from src.models.detection import DetectorMTCNN
from src.models.recognition import Embedding
from src.utils.blur_detection import is_blurred
from src.utils.head_position import HeadPositionDetector
from src.utils.utils import convert_coords, is_position_bad, clear_directory, get_colors, plot_pose_cube, draw_axis

from repos.sort.sort import Sort

import time
from sklearn.neighbors import NearestNeighbors

import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from config import _C  as cfg

# Clear contents of directory with filtered embeddings 
clear_directory(cfg.DIR_BAD_IMAGES)

# Video
cap = cv2.VideoCapture(0)

# Models (detection + recognition + head position + tracker)
detector = DetectorMTCNN(cfg)
embedding = Embedding('iresnet34', cfg)
head_position = HeadPositionDetector(cfg)
mot_tracker  = Sort( **cfg.TRACKER.SORT.ARGS)

# Score base
base_imgs_path = "C:/Users/rusrob/Python_files/SAS/29_FAMILY/DemoIpy/FaceReconition/base_embeddings/"
imgs_fpath = []
for dr in os.listdir(base_imgs_path):
    imgs_fpath.extend( glob.glob(os.path.join(base_imgs_path,dr) + "/*") )
    
if cfg.TEST_MODE:
    imgs_fpath = imgs_fpath[:10]

base_emb = []
base_face = []
base_name = []
for im_path in tqdm(imgs_fpath):
    try:
        img = cv2.imread(im_path.replace("\\","/"))
        image, bounding_boxes, probabilites , points = detector(img, input_img_type = "BGR", prob_cutoff = cfg.MTCNN.CUTOFF_FOR_EMBEDDINGS_IN_BASE)
        out = embedding( image, bounding_boxes, probabilites , points)

        points_projected = convert_coords(points, bounding_boxes, size = cfg.INSIGHTFACE.PREPROCESS.IMAGE_SIZE)
        yaw_pitch_roll = head_position.get_yaw_pitch_roll( points_projected )

        for idx, i in enumerate(out):
            filtered_face_prefix = ""
            # Filter by probability
            if probabilites[idx] < cfg.MTCNN.CUTOFF_FOR_EMBEDDINGS_IN_BASE:
                cv2.imwrite( os.path.join( cfg.DIR_BAD_IMAGES, f"prob_{probabilites[idx]:.2f}_" + str(idx) + "__" + os.path.basename(im_path) ),
                              cv2.cvtColor(out[i][0], cv2.COLOR_RGB2BGR) )
                continue

            if is_position_bad( yaw_pitch_roll[idx] ):
                print("Bad position of face!")
                print(im_path)
                filtered_face_prefix += "P_"
                # continue

            if is_blurred( out[i][0], cfg.FILTER_FACES.var_limit ):
                print("Image is too blurry!")
                print(im_path)
                filtered_face_prefix += "B_"
                # continue

            # save bad image
            if filtered_face_prefix != "" and cfg.DIR_BAD_IMAGES != "":
                cv2.imwrite( os.path.join( cfg.DIR_BAD_IMAGES, filtered_face_prefix + str(idx) + "__" + os.path.basename(im_path) ),
                              cv2.cvtColor(out[i][0], cv2.COLOR_RGB2BGR) )
                continue

            base_name.append( os.path.basename(os.path.dirname(im_path)) )
            base_emb.append( out[i][1] )
            base_face.append( out[i][0] )
    except Exception as e:
        print(e)
            
base_emb = np.array(base_emb).squeeze(1)
print("Base size: ", len(base_emb))

# Fit KNN
knn = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(base_emb)

person_color = {}
while True:
    _, frame = cap.read()
    
    # cv2.imwrite("roman.jpg", frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        image, bounding_boxes, probabilites , points = detector(frame, input_img_type = "BGR", prob_cutoff = cfg.MTCNN.CUTOFF_FOR_INPUT_EMBEDDINGS)
        out = embedding( image, bounding_boxes, probabilites , points)
        dets = np.concatenate( [bounding_boxes, probabilites[:,None] ] , axis=1) # for tracker

        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Project KP on associated face (for face pose detection)
        points_projected = convert_coords(points, bounding_boxes, size = cfg.INSIGHTFACE.PREPROCESS.IMAGE_SIZE)
        yaw_pitch_roll = head_position.get_yaw_pitch_roll( points_projected )

        trackers = mot_tracker.update(dets)

        for idx in range(probabilites.shape[0]):

            # Face keypoints
            for point in points[idx]:
                x, y = point
                cv2.circle(rgb, (x, y), 4, (255, 0, 0), -1)
            
            # Face rectangle
            x1,y1,x2,y2 = bounding_boxes[idx]
            if cfg.PLOT_RECTANGLE:
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Face position
            if cfg.PLOT_POSE == 2:
                rgb = plot_pose_cube(rgb, -yaw_pitch_roll[idx][0], -yaw_pitch_roll[idx][1], -yaw_pitch_roll[idx][2],
                        int((x1 + x2) / 2)  ,int((y1 + y2) / 2))
            if cfg.PLOT_POSE == 1:
                rgb = draw_axis(rgb, -yaw_pitch_roll[idx][0], -yaw_pitch_roll[idx][1], -yaw_pitch_roll[idx][2],
                        int((x1 + x2) / 2)  ,int((y1 + y2) / 2))

            # Face probability        
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 255, 0)  
            thickness = 2
            rgb = cv2.putText(rgb, " prob: " + str(np.round(probabilites[idx],decimals=2)) , (x1, (y1-10).astype(np.float32) ), font,  
                            fontScale, color, thickness, cv2.LINE_AA) 
            # Above image
            # rgb = cv2.putText(rgb,  " var: " + str( np.round( np.var(cv2.Laplacian(out[idx][0],cv2.CV_64F)), decimals = 2 ) ), (x1, (y1-30).astype(np.float32) ), font,  
            #                 0.5, (80,90,200), 1, cv2.LINE_AA) 
            # rgb = cv2.putText(rgb,  "yaw: {0:.2f}, pitch: {1:.2f}, roll: {2:.2f}".format( *(yaw_pitch_roll[idx]).tolist() ) , (x1, (y1-50).astype(np.float32) ), font,  
            #                 0.5, (200,122,50), 1, cv2.LINE_AA) 
            # Near border
            rgb = cv2.putText(rgb,  " var: " + str( np.round( np.var(cv2.Laplacian(out[idx][0],cv2.CV_64F)), decimals = 2 ) ), (0, int(rgb.shape[0]/2 - 30) ), font,  
                            0.5, (80,90,200), 1, cv2.LINE_AA) 
            rgb = cv2.putText(rgb,  "yaw: {0:.2f}".format( *(yaw_pitch_roll[idx]).tolist() ) , (0, int(rgb.shape[0]/2 - 10) ), font,  
                            0.5, (255,0,0), 1, cv2.LINE_AA) 
            rgb = cv2.putText(rgb,  "pitch: {1:.2f}".format( *(yaw_pitch_roll[idx]).tolist() ) , (0, int(rgb.shape[0]/2 + 10) ), font,  
                            0.5, (0,255,0), 1, cv2.LINE_AA) 
            rgb = cv2.putText(rgb,  "roll: {2:.2f}".format( *(yaw_pitch_roll[idx]).tolist() ) , (0, int(rgb.shape[0]/2 + 30) ), font,  
                            0.5, (255,255,0), 1, cv2.LINE_AA) 

            # TODO: не учитывается, что может быть несоклько лиц на экране!
            if is_position_bad( yaw_pitch_roll[idx] ):
                rgb = cv2.putText(rgb,  "BAD POSE!" , (int(rgb.shape[1]/2 -50), int(rgb.shape[0]/2) ), font,  
                                1, (0,0,255), 1, cv2.LINE_AA)       

            if is_blurred( out[idx][0], cfg.FILTER_FACES.var_limit ):
                rgb = cv2.putText(rgb,  "TOO BLURRED!" , (int(rgb.shape[1]/2 - 50), int(rgb.shape[0]/2 + 50) ), font,  
                                1, (0,0,255), 1, cv2.LINE_AA)   

            
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

        t_start = time.time()
        distances = np.sqrt(((emb - base_emb)**2).sum(1))
        distances_argsort = np.argsort(distances)
        t_end = time.time()

        t_start_knn = time.time()
        distances_knn, indexes_knn = knn.kneighbors( emb ,n_neighbors=3)
        t_end_knn = time.time()

        print(f"NP - KNN: {t_end - t_start - (t_end_knn - t_start_knn)}, Num inequalities in indexes: { (indexes_knn != distances_argsort[:3]).sum() }")
        # print(f"KNN: {t_end_knn - t_start_knn} NP: {t_end - t_start}, Num inequalities in indexes: { (indexes_knn != distances_argsort[:3]).sum() }")
        # print(distances[distances_argsort][:3])
        # print(distances_knn[:3])


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


        # Plot tracks
        if cfg.PLOT_TRACKERS:
            for i in trackers:
                idx = i[-1]
                if idx not in person_color:
                    person_color[idx] = get_colors()
                idx_color = person_color[idx]
                cv2.rectangle(rgb, tuple(i[:2].astype(int)),  tuple(i[2:4].astype(int)), tuple(idx_color.tolist()), 3 )
                rgb = cv2.putText(rgb,  str(idx),
                                    tuple(i[2:4].astype(int)+3), font,  
                                fontScale, tuple(idx_color.tolist()), 3, cv2.LINE_AA) 
                    
    except Exception as e:
        print(e)
        rgb = frame.copy()

    cv2.imshow("SAS", rgb)

    key = cv2.waitKey(1)
    if key == 27:
        break
