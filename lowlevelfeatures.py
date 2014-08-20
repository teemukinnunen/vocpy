import os
import numpy as np
import cv2


def extract_color_histogram(I, bins=256):
    "Computes a colour histogram for an input image I (I=NxMx3)"

    [ry,rx] = np.histogram(I[:,:,0], bins=bins, range=(0,255))
    [gy,gx] = np.histogram(I[:,:,1], bins=bins, range=(0,255))
    [by,bx] = np.histogram(I[:,:,2], bins=bins, range=(0,255))

    h = np.hstack((ry,gy,by))

    return h

# Compute color histogram for a image
def colorhistogram_compute(imageFile):
    I = cv2.imread(imageFile)
    r, g, b = cv2.split(I)
    [r, x] = np.histogram(r, bins=np.arange(0, 256))
    [g, x] = np.histogram(g, bins=np.arange(0, 256))
    [b, x] = np.histogram(b, bins=np.arange(0, 256))

    return np.hstack((r, g, b))

def colorhistogram_save(dataDir, imageName, h):
    descfile = dataDir + '/colorhistograms/' + imageName + '.rgbhist.npy'
    descpath = os.path.dirname(descfile)
    if os.path.exists(descpath) == False:
        os.makedirs(descpath)
    np.save(descfile, h)