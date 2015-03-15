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

def localfeatures_kp2array(ips):
    """Converts keypoint vector to array matrix"""
    A = np.zeros((len(ips), 7))

    for i in range(0, len(ips)):
        A[i,:2] = ips[i].pt
        A[i,2] = ips[i].angle
        A[i,3] = ips[i].size
        A[i,4] = ips[i].octave
        A[i,5] = ips[i].response
        A[i,6] = ips[i].class_id

    return A

def localfeatures_save(ips, lfs, imageFile, dataDir, detector, descriptor):
    """Saves local feature file in the data directory"""

    # Construct a filename and path for the feature file
    dataFile = os.path.join(dataDir,
                            'localfeatures',
                            detector + '+' + descriptor,
                            imageFile)

    print("datafile: " + dataFile)

    # Convert OpenCV keypoints to a matrix
    ips = localfeatures_kp2array(ips)

    # Make sure that the output directory exists
    dataDir = os.path.dirname(dataFile)
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    # Store the file using numpy
    np.save(dataFile+'.ip.npy', ips)
    np.save(dataFile+'.lf.npy', lfs)


def main(argv):
    """Main function for running extract features script"""

    # Parse command line parameters
    parser = argparse.ArgumentParser(description='This script extracts visual features from given images')
    parser.add_argument('-i', '--image', help="Path to the image", required=False)
    parser.add_argument('-id', '--imageDir', default=None, help='Input image directory')
    parser.add_argument('-dd', '--dataDir', default=None, help='Data directory where data will be stored')
    parser.add_argument('-il', '--imageList', default=None, help='Input image list file')
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")

    args = parser.parse_args()

    # If user wants to extracts features from many images
    if args.imageDir is not None:
        # dataDir must be given with imageDir
        if not args.dataDir:
            parser.error("--dataDir must be also given with --imageDir")
        # Get a list of images
        if args.imageList is None:
            # Get a list of images from the given directory
            imgList = glob.glob(args.imageDir + '*/*.jpg')
        else:
            # Read the given imageList
            imgList = open(args.imageList).read()
        print("Image Dir given: " + args.imageDir)
        # Start extracting features and saving them in the
        for imgid in range(0, len(imgList)):
            imageFile = imgList[imgid]
            print("Processing %s (%d/%d)" % (imageFile, imgid, len(imgList)))
            imageFilePath = os.path.join(imageFile[len(args.imageDir):])
            [ips, lfs] = lf.extractfeatures(imageFile,
                                        detector=args.detector,
                                        descriptor=args.descriptor)
            localfeatures_save(ips, lfs, imageFilePath, args.dataDir, args.detector, args.descriptor)

    # If user wants to extract features from one specific image
    elif args.image is not None:
        print("Image given: " + args.image)
    else:
        parser.error("Image OR image directory must be given!")


if __name__ == '__main__':
    main(sys.argv[1:])
