#!/usr/bin/env python2

# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

# Append vocpy scripts
sys.path.append('..' + os.path.sep)
from datasets import *

def main(argv):
    """Main function for running extract features script"""

    # Parse command line parameters
    parser = argparse.ArgumentParser(description='This script extracts visual features from given images')
    parser.add_argument('-i', '--imageDir', help="Path to the images", required=False)
    parser.add_argument('-dd', '--dataDir', help='Data directory where data will be stored', required=True)
    parser.add_argument('-il', '--imageList', default=None, help='Input image list file', required=False)
    parser.add_argument('-ip', '--detector', default="HARRIS", help='Local feature detector (HARRIS, SIFT, ...)')
    parser.add_argument('-lf', '--descriptor', default="SIFT", help="Local feature descriptor")
    parser.add_argument('-k', '--codebooksize', type=int, default=1000, help='Size of the codebook')
    parser.add_argument('-cm', '--codebookmethod', default="MiniBatchKMeans", help="Codebook generation method", required=True)
    parser.add_argument('-f', '--file', help="Codebook file path to be stored in dataDir", required=True)
    parser.add_argument('--debug', type=int, default=0, help="Debug level")

    args = parser.parse_args()

    print(args)

    # Get imageSet object, which takes actual processing of features etc
    imageSet = ImageCollection(imgDir=args.imageDir,
                                dataDir=args.dataDir,
                                ipdetector=args.detector,
                                lfdescriptor=args.descriptor,
                                codebookmethod=args.codebookmethod,
                                codebooksize=int(args.codebooksize))

    if os.path.exists(args.file):
        codebook = np.loadtxt(args.file)
        codebook = codebook[:,1:]
        imageSet.codebooksize = codebook.shape[0]
    else:
        print("Codebook file does not exist: %s" % args.file)
        sys.exit(-1)

    # Gen file path for codebookfile
    codebookfilepath = imageSet.gen_codebookfilepath()

    # Save codebook in a correct place in numpy format
    if not os.path.exists(os.path.dirname(codebookfilepath)):
        os.makedirs(os.path.dirname(codebookfilepath))
    np.save(codebookfilepath, codebook)

if __name__ == '__main__':
    main(sys.argv[1:])
