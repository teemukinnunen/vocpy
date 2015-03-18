#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import glob
import random

# Append vocpy scripts
sys.path.append('../')
from localfeatures import LocalFeatures as lf
from codebook import Codebook


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
    parser.add_argument('-N', '--nimages', default=np.Inf, help="Number of images to be used for codebook generation (less images saves memory)")

    args = parser.parse_args()

    print(args)

    # Init codebook object
    codebook = Codebook(detector=args.detector,
                        descriptor=args.descriptor,
                        codebooksize=args.codebooksize,
                        codebookmethod=args.codebookmethod)

    # Get a list of images
    if args.imageList is None:
        # Get a list of images from the given directory
        imgList = glob.glob(args.imageDir + '*/*.jpg')
    else:
        # Read the given imageList
        imgList = open(args.imageList).read()

    # If the number of images to be used for codebook generation is smaller
    # than the number of images available, then choose randomly a subset
    if int(args.nimages) < len(imgList):
        random.shuffle(imgList)
        imgList = random.sample(imgList, int(args.nimages))

    print(len(imgList))

    # Read local features
    features = []
    for imgid in range(0, len(imgList)):
        sys.stdout.write("Loading local features.. %d/%d\r" % (imgid, len(imgList)))
        imageFile = imgList[imgid]
        imageFilePath = os.path.join(imageFile[len(args.imageDir):])
        [ips, lfs] = localfeatures_load(imageFilePath,
                                        args.dataDir,
                                        args.detector,
                                        args.descriptor)
        features.append(lfs)
    print("\n\t* DONE!")
    features = np.vstack(features)

    print features.shape

    print("Generating a codebook (ip: %s; lf: %s; method: %s; size: %d" %
        (args.detector, args.descriptor, args.codebookmethod, args.codebooksize))
    cbs = codebook.generate(features)
    codebook_save(cbs, args.dataDir, args.detector, args.descriptor, args.codebookmethod, args.codebooksize)

if __name__ == '__main__':
    main(sys.argv[1:])
