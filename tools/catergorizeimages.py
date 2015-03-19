#!/usr/bin/env python2

# -*- coding: utf-8 -*-

NUMTHREADS = 4

import os
import sys
import argparse
import numpy as np
import glob
import sklearn.preprocessing
from sklearn.cluster import KMeans

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
    parser.add_argument('-k', '--codebooksize', default=1000, help='Size of the codebook')
    parser.add_argument('-cm', '--codebookmethod', default="MiniBatchKMeans", help="Codebook generation method")
    parser.add_argument('-c', '--clusters', default=0, help='Number of image clusters')
    parser.add_argument('-ln', '--LNormalization', default=2, help="Codebook histogram normalization (0=none, 1=L1, 2=L2)")

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

    # Try to load codebook histogram matrix
    cbm = codebookhistogram_matrix_load(args.dataDir, args.detector,
                                        args.descriptor, args.codebookmethod,
                                        args.codebooksize)

    # Read codebook histograms
    cbm = np.zeros((N, args.codebooksize))
    for imgid in range(0, N):
        sys.stdout.write("Loading codebook histograms.. %d/%d\r" %
                            (imgid, len(imgList)))
        imageFile = imgList[imgid]
        imageFilePath = os.path.join(imageFile[len(args.imageDir):])
        cbhistogram = codehistogram_load(imageFilePath, args.dataDir,
                                        args.detector, args.descriptor,
                                        args.codebookmethod, args.codebooksize)
        cbm[imgid, :] = cbhistogram
    print("\n\t * Done!")
    # Save because people typically hate to wait :)
    codebookhistogram_matrix_save(cbm, imgList, args.dataDir,
                                    args.detector, args.descriptor,
                                    args.codebookmethod, args.codebooksize)

    # Normalize codebook histograms (each sample independently)
    cbm = sklearn.preprocessing.normalize(cbm, 'l'+str(args.LNormalization),
                                            axis=1)

    # Cluster images
    if args.clusters is 0:
        n_imageClusters = int(np.ceil(N / 10))
    else:
        n_imageClusters = int(args.clusters)

    print("Clustering images into %d clusters" % n_imageClusters)
    km = KMeans(n_clusters=n_imageClusters, n_jobs=NUMTHREADS)
    km.fit(cbm)
    print("Cluster labels for the images")
    for imgid in range(0, N):
        print("%s \t %d" % (imgList[imgid], km.labels_[imgid]))


if __name__ == '__main__':
    main(sys.argv[1:])