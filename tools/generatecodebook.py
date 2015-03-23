#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import glob
import random

# Append vocpy scripts
sys.path.append('..' + os.path.sep)
from datasets import *

def localfeatures_load(imageFile, dataDir, detector, descriptor):
    """Loads local feature file from the data directory"""

    # Construct a filename and path for the feature file
    dataFile = os.path.join(dataDir,
                            'localfeatures',
                            detector + '+' + descriptor,
                            imageFile)

    #TODO: Make sure that the file exists

    # Store the file using numpy
    ips = np.load(dataFile + '.ip.npy')
    lfs = np.load(dataFile + '.lf.npy')

    return [ips, lfs]


def codebook_save(cbs, dataDir, detector, descriptor, codebookmethod, codebooksize):
    """Saves codebook in the data directory"""
    # Construct a filename and path for the feature file
    dataFile = os.path.join(dataDir,
                            'codebooks',
                            detector + '+' + descriptor,
                            codebookmethod + '+' + str(codebooksize) + '.npy')

    # Create a directory for the codebook if needed
    if not os.path.exists(os.path.dirname(dataFile)):
        os.makedirs(os.path.dirname(dataFile))

    np.save(dataFile, cbs)

def main(argv):
    """Main function for running extract features script"""

    # Parse command line parameters
    parser = argparse.ArgumentParser(description='This script extracts visual features from given images')
    parser.add_argument('-i', '--imageDir', help="Path to the images", required=True)
    parser.add_argument('-dd', '--dataDir', help='Data directory where data will be stored', required=True)
    parser.add_argument('-il', '--imageList', default=None, help='Input image list file')
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")
    parser.add_argument('-k', '--codebooksize', default=1000, help='Size of the codebook')
    parser.add_argument('-cm', '--codebookmethod', default="MiniBatchKMeans", help="Codebook generation method")
    parser.add_argument('-N', '--nimages', default=-1, help="Number of images to be used for codebook generation (less images saves memory)")
    parser.add_argument('--debug', default=0, help="Debug level")

    args = parser.parse_args()

    print(args)

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

    # Choose random population for the codebook generation if we need to save
    # some memory
    N = len(imageSet.imageNames)
    if N > int(args.nimages) and int(args.nimages) > 0:
        imageSet.imageNames = random.sample(imageSet.imageNames, int(args.nimages))
        N = len(imageSet.imageNames)

    imageSet.codebook_generate()

if __name__ == '__main__':
    main(sys.argv[1:])
