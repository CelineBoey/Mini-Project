import cv2 as cv
# from scipy.spatial.distance import euclidean
# import math
import numpy as np
import matplotlib.pyplot as plt

def show_img(window_name, img, adjust=False):
    """3 arguments: window name, source images, boolean to adjust to screen size"""
    if adjust:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    else:
        cv.namedWindow(window_name)

    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def plt_img(img, title=None):
    if len(img.shape)==3:
        plt.imshow(img)
    elif len(img.shape)==2:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        raise Exception('Incorrect image dimension')
        
    if title is not None:
        plt.title(title)
        
    plt.xticks([]), plt.yticks([])
