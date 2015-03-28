#!/usr/bin/env python2

# -*- coding: utf-8 -*-

NUMTHREADS = 4
PICRES = 50

import os
import sys
import argparse
import numpy as np
import time
import pylab
import sklearn.preprocessing
import scipy.misc as sm

# Append vocpy scripts
sys.path.append('..' + os.path.sep)
from datasets import *

# Import SOMPY
#TODO: fix path if necessary ;)
sys.path.append(os.path.join('..', '..', 'SOMPY'))
from SOMPY import SOM

def codebookhistogram_matrix_save(cbm, imgList, dataDir, detector, descriptor, codebookmethod, codebooksize):
    "Saves codebook histograms matrix"

    dataFile = os.path.join(dataDir, 'codehistograms_matrices',
        detector + '+' + descriptor + codebookmethod + '+' + str(codebooksize))

    if not os.path.exists(os.path.dirname(dataFile)):
        os.makedirs(os.path.dirname(dataFile))

    np.save(dataFile, cbm)

    return 0


def codebookhistogram_matrix_load(dataDir, detector, descriptor, codebookmethod, codebooksize):
    "Loads codebook histograms matrix"

    dataFile = os.path.join(dataDir, 'codehistograms_matrices',
        detector + '+' + descriptor + codebookmethod + '+' + str(codebooksize))

    if not os.path.exists(dataFile):
        return None

    cbm = np.load(dataFile)

    return cbm


def main(argv):
    """Main function for running extract features script"""

    # Parse command line parameters
    parser = argparse.ArgumentParser(description='This script extracts visual features from given images')
    parser.add_argument('-i', '--imageDir', help="Path to the images", required=True)
    parser.add_argument('-dd', '--dataDir', help='Data directory where data will be stored', required=True)
    parser.add_argument('-il', '--imageList', default=None, help='Input image list file')
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")
    parser.add_argument('-k', '--codebooksize', type=int, default=1000, help='Size of the codebook')
    parser.add_argument('-cm', '--codebookmethod', default="MiniBatchKMeans", help="Codebook generation method")
    parser.add_argument('-c', '--clusters',type=int, default=0, help='Number of image clusters')
    parser.add_argument('-ln', '--LNormalization', default=2, help="Codebook histogram normalization (0=none, 1=L1, 2=L2)")
    parser.add_argument('-sx', '--SOMX', type=int, default=50, help='Number of horizontal units SOM')
    parser.add_argument('-sy', '--SOMY', type=int, default=30, help='Number of vertical units SOM')
    parser.add_argument('--debug', type=int, default=0, help='Debugging level')

    args = parser.parse_args()

    # Get imageSet object, which takes actual processing of features etc
    imageSet = ImageCollection(imgDir=args.imageDir,
                                dataDir=args.dataDir,
                                ipdetector=args.detector,
                                lfdescriptor=args.descriptor,
                                codebookmethod=args.codebookmethod,
                                codebooksize=args.codebooksize)

    # If user gave as an image list file, we must update the imageSet
    if args.imageList is not None:
        imageSet.read_imagelist(args.imageList)

    # Number of images
    N = len(imageSet.imageNames)

    # Try to load codebook histogram matrix
    cbm = codebookhistogram_matrix_load(args.dataDir, args.detector,
                                        args.descriptor, args.codebookmethod,
                                        args.codebooksize)

    # Read codebook histograms
    cbm = np.zeros((N, args.codebooksize))
    for imgid in range(0, N):
        sys.stdout.write("Loading codebook histograms.. %d/%d\r" %
                            (imgid, N))
        cbhistogram = imageSet.codebookhistograms_load(imageSet.imageNames[imgid])
        cbm[imgid, :] = cbhistogram
    print("\n\t * Done!")

    # Save because people typically hate to wait :)
    codebookhistogram_matrix_save(cbm, imageSet.imageNames, args.dataDir,
                                    args.detector, args.descriptor,
                                    args.codebookmethod, args.codebooksize)

    # Normalize codebook histograms (each sample independently)
    cbm = sklearn.preprocessing.normalize(cbm, 'l'+str(args.LNormalization),
                                            axis=1)

    ## Visualize image set using 2D SOM
    # Train SOM
    isom = SOM('imagemap', cbm, norm_method='none', mapsize = (int(args.SOMY), int(args.SOMX)))
    #TODO: Mapshape is not implemented yet in sompy
    #isom.set_topology(mapsize = (int(args.SOMY), int(args.SOMX)),
    #                    mapshape = 'toroid')
    isom.train()
    # Map images to xy on the map
    xy = isom.ind_to_xy(isom.bmu[0])
    # Sort images based on their distance to the map to find the most suitable
    # images for visualization
    d = isom.bmu[1]
    idx = np.argsort(d)
    imgmap = np.zeros(isom.mapsize)-1
    for i in idx:
        if imgmap[xy[i,0],xy[i,1]] == -1:
            imgmap[xy[i,0],xy[i,1]] = i

    # Generate SOM image
    I = np.zeros((isom.mapsize[0]*PICRES, isom.mapsize[1]*PICRES, 3), dtype=np.uint8)
    for sy in range(0, isom.mapsize[0]):
        offset_y = sy * PICRES
        for sx in range(0, isom.mapsize[1]):
            offset_x = sx * PICRES
            #
            imgid = int(imgmap[sy,sx])
            # If the cell is set to something
            if imgid > -1:
                subimage = pylab.imread(os.path.join(args.imageDir,
                                                imageSet.imageNames[imgid]))
                subimage = sm.imresize(subimage, (PICRES, PICRES))
                I[offset_y:offset_y+PICRES,offset_x:offset_x+PICRES,:] = subimage
    pylab.imshow(I)
    pylab.pause(0.1)
    resultFile = os.path.join(args.dataDir,
                                'results',
                                '2d_som_' +
                                str(args.SOMY) + 'x' + str(args.SOMY) +
                                time.strftime('_time_%Y-%m-%d_%H%M') + '.jpg')
    if not os.path.exists(os.path.dirname(resultFile)):
        os.makedirs(os.path.dirname(resultFile))
    sm.imsave(resultFile, I)
    print(("Resultfile saved in: %s" % resultFile))
    raw_input("Press [enter] to quit")


if __name__ == '__main__':
    main(sys.argv[1:])