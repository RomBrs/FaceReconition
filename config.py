from yacs.config import CfgNode as CN

_C = CN()
_C.DEVICE = 'cpu'
_C.TEST_MODE = 0
_C.DIR_BAD_IMAGES = "C:/Users/rusrob/Python_files/SAS/29_FAMILY/DemoIpy/FaceReconition/filtered_faces" # "" - if do not use

_C.FILTER_FACES = CN()
_C.FILTER_FACES.var_limit = 30

_C.MTCNN = CN()
_C.MTCNN.TRANSFORM = CN()
_C.MTCNN.TRANSFORM.SIZE = (512, 512)
_C.MTCNN.CUTOFF_FOR_EMBEDDINGS_IN_BASE = 0.95
_C.MTCNN.CUTOFF_FOR_INPUT_EMBEDDINGS = 0.5

_C.INSIGHTFACE = CN()
_C.INSIGHTFACE.PREPROCESS = CN()
_C.INSIGHTFACE.PREPROCESS.MEAN = [0.5] * 3  #  [0.485, 0.456, 0.406]
_C.INSIGHTFACE.PREPROCESS.STD = [0.5 * 256 / 255] * 3  #  [0.229, 0.224, 0.225]
_C.INSIGHTFACE.PREPROCESS.IMAGE_SIZE = (112, 112)
_C.INSIGHTFACE.PREPROCESS.SIZE = '112, 112'

_C.HEAD_POSITION = CN()
# _C.HEAD_POSITION.MODEL_FPATH = "C:/Users/rusrob/Python_files/SAS/29_FAMILY/DemoIpy/FaceReconition/jit_models/yaw_pitch_roll/YPR_mtcnn_5points.jit"
_C.HEAD_POSITION.MODEL_FPATH = "C:/Users/rusrob/Python_files/SAS/29_FAMILY/DemoIpy/FaceReconition/jit_models/yaw_pitch_roll/YPR_mtcnn_5points_pandora.jit"


# cfg = _C