import cv2
import numpy as np

def is_blurred(image, var_limit):
    """ Laplace
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = np.var(laplacian)
    if variance < var_limit:
        return True
    else:
        return False