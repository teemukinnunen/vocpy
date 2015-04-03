#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse

# Append vocpy scripts
sys.path.append('..' + os.path.sep)
from datasets import *

def main(argv):
    """Main function for running extract features script"""

    # Parse command line parameters
    parser = argparse.ArgumentParser(description='This script extracts visual features from given images')
    parser.add_argument('-i', '--imageDir', help="Path to the image or directory", required=True)
    parser.add_argument('-dd', '--dataDir', help='Data directory where data will be stored', required=True)
    parser.add_argument('-il', '--imageList', default=None, help='Input image list file')
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")
    parser.add_argument('-o', '--outputfile', default=None, help="Write all local features to a single file")
    parser.add_argument('--debug', type=int, default=0, help="Debug level")
    args = parser.parse_args()

    # Get a list of images
    imageSet = ImageCollection(imgDir=args.imageDir,
                                dataDir=args.dataDir,
                                ipdetector=args.detector,
                                lfdescriptor=args.descriptor)

    # If user gave as an image list file, we must update the imageSet
    if args.imageList is not None:
        imageSet.read_imagelist(args.imageList)

    # Finally, extract local features from given images
    if args.outputfile == None:
        imageSet.localfeatures_extract(debuglevel=args.debug)
    else:
        imageSet.localfeatures_extract_to_bin(args.outputfile, args.debug)

if __name__ == '__main__':
    main(sys.argv[1:])
