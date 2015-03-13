#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

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

    imageFilename = os.path.basename(imageFile)

    # Construct a filename and path for the feature file
    dataFile = os.path.join(dataDir,
                            'localfeatures',
                            detector + '+' + descriptor,
                            imageFilename)

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
    parser.add_argument('-i', '--image', help="Path to the image", required=True)
    parser.add_argument('-d', '--dataDir', default="data", help='Data directory where data will be stored')
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")

    args = parser.parse_args()

    #print("Command line arguments:")
    #print("dataDir: " + args.dataDir)
    #print(args)

    #TODO: Implement some logic to realise if a list of images is given
    imageFile = args.image
    print(("Extracting features from " + imageFile))
    [ips, lfs] = lf.extractfeatures(imageFile,
                                detector=args.detector,
                                descriptor=args.descriptor)
    # Store results in a data file
    localfeatures_save(ips,
                        lfs,
                        imageFile,
                        args.dataDir,
                        args.detector,
                        args.descriptor)

if __name__ == '__main__':
    main(sys.argv[1:])
