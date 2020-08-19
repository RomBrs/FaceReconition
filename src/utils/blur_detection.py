import cv2
import numpy as np

def is_blurred(image, var_limit, return_var_limit = False):
    """ Laplace
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = np.var(laplacian)
    if variance < var_limit:
        if return_var_limit:
            return True, variance    
        return True
    else:
        if return_var_limit:
            return False, variance
        return False