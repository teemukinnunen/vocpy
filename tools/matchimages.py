#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import glob
import cv2

# Append vocpy scripts
sys.path.append('../')
from localfeatures import *

def matchimages(imgPath1, imgPath2, detector, descriptor, debug=False):
    "Tries to match two images based on local features"

    print(debug)

    [H, matchingFeatures, descDistances] = match_images_spatially(imgPath1,
                                            imgPath2,
                                            detectorName=detector,
                                            descriptorName=descriptor,
                                            boolDebugPlot=debug)
    return [H, matchingFeatures, descDistances]


def main(argv):
    """Main function for matching images"""

    # Parse command line parameters
    parser = argparse.ArgumentParser(description='This script extracts visual features from given images')
    parser.add_argument('-i1', '--image1', help="Path to the image or directory", required=True)
    parser.add_argument('-i2', '--image2', help="Path to the image or directory", required=True)
    parser.add_argument('-dd', '--dataDir', help='Data directory where data will be stored', required=False)
    parser.add_argument('-il', '--imageList', default=None, help='Input image list file')
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")
    parser.add_argument('--debug', default=0, help="Debug mode (plots images)")

    args = parser.parse_args()

    [H, matchingFeatures, descDistances] = matchimages(args.image1, args.image2,
                                                args.detector, args.descriptor,
                                                args.debug>0)
    print("Transformation matrix H: ")
    print(H)
    print("Matching local features: ")
    print(matchingFeatures)
    print("Local feature description distances: ")
    print(descDistances)

# MAIN
if __name__ == '__main__':
    main(sys.argv[1:])
