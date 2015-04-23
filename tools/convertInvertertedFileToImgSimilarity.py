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

def read_and_process_invertedfile(filepath, N):
    f = open(filepath, 'r')

    S = np.zeros((N, N), np.int16)
    line = f.readline()
    count = 0
    while line:
        count = count + 1
        sys.stdout.write('\rProcessing code %06d' % count)
        idx = line.split(' ')
        for i in range(0,len(idx)-1):
            for j in range(i,len(idx)-1):
                S[idx[i],idx[j]] = S[idx[i],idx[j]] + 1
        line = f.readline()
    f.close()

    return S

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

    filepath = os.path.join(args.dataDir,
                            'invertedfiles',
                            args.detector + '+' +
                            args.descriptor +
                            args.codebookmethod + '+' +
                            str(args.codebooksize) + '.txt')
    if os.path.exists(filepath):
        S = read_and_process_invertedfile(filepath, N)
    else:
        print("Invertedfile: %s does not exist!" % filepath)
        sys.exit(-1)

    # Store similarity matrix
    simfile = os.path.join(args.dataDir,
                            'similaritymatrices',
                            args.detector + '+' +
                            args.descriptor +
                            args.codebookmethod + '+' +
                            str(args.codebooksize) + '.npy')

    if not os.path.exists(os.path.dirname(simfile)):
        os.makedirs(os.path.dirname(simfile))
    np.save(simfile, S)

if __name__ == '__main__':
    main(sys.argv[1:])