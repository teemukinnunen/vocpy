#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import glob
import cv2
import sklearn.preprocessing as prepocessing
import scipy.spatial as spatial
import pylab

# Append vocpy scripts
sys.path.append('../')
from localfeatures import *

def codehistogram_load(imageFile, dataDir, detector, descriptor, codebookmethod, codebooksize):
    "Saves codebook histogram into the dataDir"

    dataFile = os.path.join(dataDir,
                            'codehistograms',
                            detector + '+' + descriptor,
                            codebookmethod + '+' + str(codebooksize),
                            imageFile + '.npy')

    # Create a directory for the codebook if needed
    if not os.path.exists(dataFile):
        print("Codebook histogramfile %s is missing! x/" % dataFile)
        sys.exit(-1)

    codehistogram = np.load(dataFile)
    return codehistogram

def main(argv):
    """Main function for matching images"""

    # Parse command line parameters
    parser = argparse.ArgumentParser(description='This script extracts visual features from given images')
    parser.add_argument('-i', '--imageDir', help="Path to the images", required=True)
    parser.add_argument('-dd', '--dataDir', help='Data directory where data will be stored', required=True)
    parser.add_argument('-il', '--imageList', default=None, help='Input image list file')
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")
    parser.add_argument('-k', '--codebooksize', default=1000, help='Size of the codebook')
    parser.add_argument('-cm', '--codebookmethod', default="MiniBatchKMeans", help="Codebook generation method")
    parser.add_argument('-c', '--clusters', default=0, help='Number of image clusters')
    parser.add_argument('-ln', '--LNormalization', default=2, help="Codebook histogram normalization (0=none, 1=L1, 2=L2)")
    parser.add_argument('--debug', default=0, help="Debug mode (plots images)")

    args = parser.parse_args()

    # Get a list of images
    if args.imageList is None:
        # Get a list of images from the given directory
        imgList = glob.glob(args.imageDir + '*/*.jpg')
    else:
        # Read the given imageList
        imgList = open(args.imageList).readlines()

    # Number of images
    N = len(imgList)

    # Load codebook histograms
    cbm = np.zeros((N, args.codebooksize))
    for imgid in range(0, N):
        sys.stdout.write("Loading codebook histograms.. %d/%d\r" %
                            (imgid, len(imgList)))
        imageFile = imgList[imgid]
        imageFilePath = os.path.join(imageFile[len(args.imageDir)+1:])
        cbhistogram = codehistogram_load(imageFilePath, args.dataDir,
                                        args.detector, args.descriptor,
                                        args.codebookmethod, args.codebooksize)
        cbm[imgid, :] = cbhistogram
    print("\n\t * Done!")

    # Find similar images based on codebook histograms
    cbm = prepocessing.normalize(cbm, 'l2')
    D = spatial.distance_matrix(cbm, cbm)
    idx = np.argsort(D, axis=1)

    if args.debug > 0:
        h = pylab.figure()

    for i in range(0, N):
        if args.debug > 0:
            I = pylab.imread(imgList[i])
            h.add_subplot(5,5,1)
            pylab.imshow(I)
            pylab.axis('off')
        for j in range(1, 25):
            sys.stdout.write("%s (d=%1.3f) " % (imgList[idx[i,j]][len(args.imageDir)+1:], D[i,idx[i,j]]))
            if args.debug > 0:
                h.add_subplot(5,5,j+1)
                I = pylab.imread(imgList[idx[i,j]])
                pylab.imshow(I)
                pylab.axis('off')
        sys.stdout.write('\n')
        if args.debug > 0:
            #pylab.draw_now()
            pylab.pause(0.01)
            pylab.savefig('/home/tekinnun/Pictures/work/visualisations/flickr_temporal_views/results/1/'+imgList[idx[i,j]][len(args.imageDir)+1:],bbox_inches='tight')
    # Match local features using spatial verification methods
    #[H, matchingFeatures, descDistances] = matchimages(args.image1, args.image2,
    #                                            args.detector, args.descriptor,
    #                                            args.debug>0)

# MAIN
if __name__ == '__main__':
    main(sys.argv[1:])
