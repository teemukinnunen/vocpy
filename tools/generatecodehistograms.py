#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import glob

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


def codebook_load(dataDir, detector, descriptor, codebookmethod, codebooksize):
    """Saves codebook in the data directory"""
    # Construct a filename and path for the feature file
    dataFile = os.path.join(dataDir,
                            'codebooks',
                            detector + '+' + descriptor,
                            codebookmethod + '+' + str(codebooksize) + '.npy')

    # Create a directory for the codebook if needed
    if not os.path.exists(dataFile):
        print("Codebook-%s missing" % dataFile)
        sys.exit(-1)

    cbs = np.load(dataFile)

    return cbs

def codehistogram_save(codehistogram, imageFile, dataDir, detector, descriptor, codebookmethod, codebooksize):
    "Saves codebook histogram into the dataDir"

    dataFile = os.path.join(dataDir,
                            'codehistograms',
                            detector + '+' + descriptor,
                            codebookmethod + '+' + str(codebooksize),
                            imageFile + '.npy')

    # Create a directory for the codebook if needed
    if not os.path.exists(os.path.dirname(dataFile)):
        os.makedirs(os.path.dirname(dataFile))

    np.save(dataFile, codehistogram)

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

    # Load codebook
    cbs = codebook_load(args.dataDir, args.detector, args.descriptor, args.codebookmethod, args.codebooksize)
    codebook.codebook = cbs

    # Compute and save codebook histograms
    for imgid in range(0, len(imgList)):
        sys.stdout.write("Generating codebook histograms.. %d/%d\r" % (imgid, len(imgList)))
        imageFile = imgList[imgid]
        imageFilePath = os.path.join(imageFile[len(args.imageDir):])
        # Read local features
        [ips, lfs] = localfeatures_load(imageFilePath,
                                        args.dataDir,
                                        args.detector,
                                        args.descriptor)
        # Compute codebook histogram
        codehistogram = codebook.compute_histogram(lfs)

        codehistogram_save(codehistogram,
                            imageFilePath,
                            args.dataDir,
                            args.detector,
                            args.descriptor,
                            args.codebookmethod,
                            args.codebooksize)

    print("\n\t* DONE!")

if __name__ == '__main__':
    main(sys.argv[1:])
