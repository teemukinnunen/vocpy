#!/usr/bin/env python2

# -*- coding: utf-8 -*-

NUMTHREADS = 4
PICRES = 50

import os
import sys
import argparse
import numpy as np
import cPickle

# Append vocpy scripts
sys.path.append('..' + os.path.sep)
from datasets import *

# Import SOMPY
#TODO: fix path if necessary ;)
sys.path.append(os.path.join('..', '..', 'SOMPY'))
from SOMPY import SOM

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

    # Init inverted file
    invertedfile = [[] for i in range(args.codebooksize)]

    # Build inverted file based on codebook histograms
    for imgid in range(0, N):
        #sys.stdout.write("\rGenerating inverted index.. %d/%d" %
        #                    (imgid, N))
        cbhistogram = imageSet.codebookhistograms_load(imageSet.imageNames[imgid])
        # Add link from each
        idx = np.argwhere(cbhistogram > 0)
        print(idx)
        for codeid in idx:
            #print("\nAdding image %05d to code %06d" % (imgid, codeid[0]))
            invertedfile[codeid[0]].append(imgid)
    print("\t * Done!")

    # Save codes
    invertedfilepath = os.path.join(args.dataDir,
                                    'invertedfiles',
                                    args.detector + '+' + args.descriptor +
                                    args.codebookmethod + '+' +
                                    str(args.codebooksize) +'.txt')

    if not os.path.exists(os.path.dirname(invertedfilepath)):
        os.makedirs(os.path.dirname(invertedfilepath))

    # Save inverted file using cPickle
    f = open(invertedfilepath, 'wb')
    cPickle.dump(invertedfile, f)
    f.close()

    #for codeid in range(0,args.codebooksize):
    #    print("Codeid: %06d len: %d" % (codeid, len(invertedfile[codeid])))
    #    for lid in invertedfile[codeid]:
    #        f.write("%d " % lid)
    #    f.write("\n")

    #print(invertedfile)

if __name__ == '__main__':
    main(sys.argv[1:])