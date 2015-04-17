# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#
# Dataset hanndling such as ImageCollection and their Annotations
#
#------------------------------------------------------------------------------

import os
import sys
import numpy

from localfeatures import LocalFeatures
from codebook import Codebook
from codebook import CodebookHistograms

class ImageCollection:
    """Image collection holds a collection of images + offers some
    image analysis tools for the collection."""
    imgDir = ''
    dataDir = ''

    ipdetector = 'FAST'
    lfdescriptor = 'FREAK'
    codebookmethod = 'MiniBatchKMeans'
    codebooksize = 1000
    histnormalisation = 'L2'

    imageNames = []

    def __init__(self,
                imgDir='',
                dataDir='',
                ipdetector='FAST',
                lfdescriptor='FREAK',
                codebookmethod='MiniBatchKMeans',
                codebooksize=1000,
                histnormalisation='L2'):
        """ImageCollection holds information about the a collection of images +
        help user to make visual analysis for the collection. It stores all the
        temporary data into dataDir.

        Parameters:
            imgdir                        - directory for the images
            dataDir                       - directory for temp data
            ipdetector                    - local feature detector
            lfdescriptor                  - local feature descriptor
            codebookmethod                - codebook generation method
            codebooksize                  - codebook size
            histnormalisation             - codebook histogram normalisation

        Class functions
            imgdir_read()                 - reads imgDir
            localfeatures_extract()       - extract features of images
            localfeatures_read()          - reads local features and returns it
            codebook_generate()           - generates a visual codebook
            codebook_load()               - loads the generated codebook
            codebook_save()               - saves the generated codebook
            codebookhistograms_generate() - generates codebook histograms
            codebookhistogreams_load()    - loads a list of codebook histograms
            codebookhistograms_save()     - saves codebook histograms

        Class member variables
            imageNames                    - list of image names :)"""

        self.imgDir = imgDir
        self.dataDir = dataDir
        # fix data dir if necessary
        if not self.imgDir.endswith(os.path.sep):
            self.imgDir = self.imgDir + os.path.sep

        self.ipdetector = ipdetector
        self.lfdescriptor = lfdescriptor
        self.codebookmethod = codebookmethod
        self.codebooksize = codebooksize
        self.histnormalisation = histnormalisation

        self.imageNames = []

        if len(self.imgDir) > 0:
            self.imgdir_read()

    # -------------------------------------------------------------------------
    # Image directory functions
    # -------------------------------------------------------------------------
    def imgdir_read(self):
        """Get a list of image files in the image folder"""

        # Go through all the directories and images within them and add them
        # to the list
        for root, subFolders, files in os.walk(self.imgDir):
            for file in files:
                if file.endswith('jpg'):
                    imgFullpath = os.path.join(root, file)
                    self.imageNames.append(imgFullpath[len(self.imgDir):])


    def read_imagelist(self, imageListFile):
        "Read a text file containing a list of images"
        if os.path.exists(imageListFile):
            # Read the given imageList
            #imgList = open(imageListFile, 'r').read().split('\n')
            imgList = open(imageListFile).read().splitlines()
            self.imageNames = imgList
        else:
            print("Image list file %s does not exist!" % imageListFile)

    # -------------------------------------------------------------------------
    # Local feature functions wrapped to the imagecollection class
    # -------------------------------------------------------------------------
    def gen_featurefile_path(self, imgFile):
        filepath = os.path.join(self.dataDir,
                                'localfeatures',
                                self.ipdetector + '+' + self.lfdescriptor,
                                imgFile)
        return filepath

    def localfeatures_extract(self, debuglevel=0):
        """Extract local features from a given imgDir"""
        imgIdx = 0
        nImgs = len(self.imageNames)

        for imgFile in self.imageNames:
            if debuglevel > 0:
                sys.stdout.write("Processing image %d of %d images.\r" % (imgIdx+1, nImgs))

            localfeaturefile = self.gen_featurefile_path(imgFile)

            # If the feature file already exists skip it
            if os.path.exists(localfeaturefile + '.key.npy') == False:
                lf = LocalFeatures(self.ipdetector,
                                    self.lfdescriptor)

                imgPath = os.path.join(self.imgDir,imgFile)
                if os.path.exists(imgPath):
                    # Extract local features from the image
                    try:
                        [f, d] = lf.extract(imgPath)
                    except:
                        print(("Could not extract features from: %s" % imgPath))
                        f = []
                        d = numpy.zeros((0,128))

                    localfeaturefiledir = os.path.dirname(localfeaturefile)

                    if os.path.exists(localfeaturefiledir) == False:
                        os.makedirs(localfeaturefiledir)

                    # Convert keypoints objects to matrix
                    f = lf.keypoints2framematrix(f)

                    numpy.save(localfeaturefile + '.desc', d)
                    numpy.save(localfeaturefile + '.key', f)

                else:
                    if debuglevel > 0:
                        sys.stdout.write("\n")
                    print("Imagefile %s does not exist!" % imgPath)

            imgIdx += 1
        if debuglevel > 0:
            print("\n\t * DONE!")

    def localfeatures_extract_to_bin(self,  outputfile=None, debuglevel=0):
        "Extract or load local features and save them into a single binary file"

        nImgs = len(self.imageNames)
        imgIdx = 0

        if outputfile == None:
            outputfile = os.path.join(self.dataDir,
                                        'localfeatures',
                                        'all_features_' + self.ipdetector +
                                        '+' + self.lfdescriptor + '.bin')

        # Get localfeature object for extracting local features
        lf = LocalFeatures(self.ipdetector, self.lfdescriptor)

        # Open outputfile for writing
        if not os.path.exists(os.path.dirname(outputfile)):
            os.makedirs(os.path.dirname(outputfile))
        of = open(outputfile, 'wb')

        # Write something already.. we need to fix these in the end
        nFeaturesTotal = numpy.uint32(0)
        nFeaturesTotal.tofile(of)
        nDims = numpy.uint32(0)
        nDims.tofile(of)

        for imgFile in self.imageNames:
            imgIdx = imgIdx + 1
            imgPath = os.path.join(self.imgDir, imgFile)
            if debuglevel > 0:
                sys.stdout.write("Processing image %d of %d images.\r" % (imgIdx, nImgs))

            localfeaturefile = self.gen_featurefile_path(imgFile)

            d = numpy.zeros((0,128), dtype=numpy.float32)

            # If the feature file already exists, read it
            if os.path.exists(localfeaturefile + '.desc.npy'):
                d = numpy.load(localfeaturefile + '.desc.npy')
            # If the feature file does not exist, extract it
            else:
                # Extract local features from the image
                try:
                    [f, d] = lf.extract(imgPath)
                except:
                    print(("Could not extract features from: %s" % imgPath))

            nFeatures = d.shape[0]
            nDims = numpy.uint32(d.shape[1])
            nFeaturesTotal = numpy.uint32(nFeaturesTotal + nFeatures)
            data = numpy.array(d, dtype=numpy.float32)
            data.tofile(of)

        # Update the number of features and feature dims
        of.seek(0,0)
        nFeaturesTotal.tofile(of)
        nDims.tofile(of)
        of.close()

        print("%d %dD features saved in %s" % (nFeaturesTotal, nDims, outputfile))

    def localfeatures_read(self, debuglevel=0):
        """Load local features which are already extracted and return them """
        # Initialize feature matrix
        features = numpy.zeros((0,0))
        # Read local features from each image in the ImageCollection
        count = 0
        for imageName in self.imageNames:
            count = count + 1
            # Print some progress information
            if debuglevel == 1:
                sys.stdout.write("Reading feature %d/%d\r" % (count, len(self.imageNames)))
            if debuglevel > 2:
                print("Feature matrix size: %d x %d" % (features.shape[0],
                                                        features.shape[1]))
            try:
                # Define name for the local feature file
                localfeaturefile = self.gen_featurefile_path(imageName)

                # Make sure that the local feature descriptor file exist
                if os.path.exists(localfeaturefile + '.desc.npy'):
                    # Read local feature descriptor
                    imgFeatures = numpy.load(localfeaturefile + '.desc.npy')
                    # If the feature matrix is not empty then add loaded features
                    # on the top of the matrix
                    if features.shape[0] > 0:
                        features = numpy.vstack((imgFeatures, features))
                    else:
                        # If the feature matrix is empty then set it to
                        # loaded image features
                        features = imgFeatures
                        features = numpy.vstack((imgFeatures, features))
                else:
                    print(("Local feature file does not exist (%s)" % (localfeaturefile + '.desc.npy')))
            except ValueError:
                print(("Problems with image: " + imageName))

        return features

    # -------------------------------------------------------------------------
    # Codebook generation function wrapped inside the collection class
    # -------------------------------------------------------------------------
    def codebook_generate(self, debuglevel=0):
        """Generates a codebook"""
        if debuglevel > 0:
            print("Loading local features for codebook generation")
        features = self.localfeatures_read(debuglevel=debuglevel)
        if debuglevel > 0:
            print("Local features loaded. Feature matrix is %d x %d" %
                                                        (features.shape[0],
                                                        features.shape[1]))
        codebook = Codebook(codebookmethod=self.codebookmethod,
                            codebooksize=self.codebooksize)
        if debuglevel > 0:
            print("Generating codebook")
        codebook.generate(features)
        if debuglevel > 0:
            print("Codebook generation done.")
        self.codebook_save(codebook)

        return codebook

    def gen_codebookfilepath(self, optionalFilename=""):
        "Generates file path for the codebook file"
        if len(optionalFilename) == 0:
            codebookfile = os.path.join(self.dataDir,
                                         'codebooks',
                                         self.ipdetector + '+' + self.lfdescriptor,
                                         self.codebookmethod + '+' + str(self.codebooksize) + '.npy')
        else:
            codebookfile = self.dataDir + '/codebooks/' + optionalFilename

        return codebookfile

    def codebook_save(self, codebook, optionalFilename='', debuglevel=0):
        """Saves the generated cocebook"""

        # Define filename for the codebook
        codebookfile = self.gen_codebookfilepath(optionalFilename)

        codebookpath = os.path.dirname(codebookfile)

        if os.path.exists(codebookpath) == False:
            os.makedirs(codebookpath)

        numpy.save(codebookfile, codebook.codebook)

    def codebook_load(self, optionalFilename='', debuglevel=0):
        """Loads the generated codebook"""

        # Define filename for the codebook
        codebookfile = self.gen_codebookfilepath(optionalFilename)

        if os.path.exists(codebookfile) == True:
            codebook = numpy.load(codebookfile)
            return codebook
        else:
            print "Could not load the damn codebook"
            return False

    # -------------------------------------------------------------------------
    # Codebook histogram generation functions
    # -------------------------------------------------------------------------
    def codebookhistograms_generate(self, codebook=[], debuglevel=0):
        """Generates codebook histograms for the image collection imgs"""

        # Load codebook
        if numpy.size(codebook) == 0:
            codebook = self.codebook_load()

        # Initialize codebook histogram matrix
        N = len(self.imageNames)
        k = codebook.shape[0]
        codebookhistograms = numpy.zeros((N, k))

        # Init codeebookhistogram object for computing histograms
        Codebookhist = CodebookHistograms()

        count = 0
        for imageFile in self.imageNames:
            count = count + 1

            if debuglevel > 0:
                sys.stdout.write('Generating codebook histograms (%d/%d)\r' % (count, len(self.imageNames)))

            # Try to load codebookhistogram
            codebookhist = self.codebookhistograms_load(imageFile)

            # If codebookhistograms is not being loaded, we need to compute one
            if codebookhist is None:

                localfeaturefile = self.gen_featurefile_path(imageFile)
                desc = numpy.load(localfeaturefile + '.desc.npy')

                # Compute codebookhistogram
                codebookhist = Codebookhist.generate(codebook, desc)

                # Store histogram
                self.codebookhistograms_save(codebookhist, imageFile)

            # Stack codebookhistograms into a matrix
            if codebookhistograms.size == 0:
                codebookhistograms[0, :] = codebookhist
            else:
                try:
                    codebookhistograms[count-1, :] = codebookhist
                except:
                    print("Couldnt concatenate codebook histogram to feature matrix")
                    print(codebookhistograms.shape)
                    print(codebookhist.shape)
                    if codebookhist.size == 0:
                        print("Codebookhistogram is empty (%s)" % imageFile)
                        print("Using codebookhistogram filled with zeroes")
                        codebookhist = numpy.zeros((1, k))
                        try:
                            codebookhistograms[count-1, :] = codebookhist
                        except:
                            print("Did not work out as planned... dyiing...")
                            sys.exit(-1)
                    else:
                        try:
                            codebookhistograms[count-1, :] = codebookhist.transpose
                        except:
                            print("Still failing.. dying..")
                            sys.exit(-1)

        return codebookhistograms

    def gen_codebookhistogram_filepath(self, imgFile):
        "Generates filepath for the codebookhistogram file"

        codebookhistfilepath = os.path.join(self.dataDir,
                                            'codebookhistograms',
                                            self.ipdetector + '+' + \
                                            self.lfdescriptor,
                                            self.codebookmethod + \
                                            '+' + str(self.codebooksize),
                                            imgFile + '.npy')

        return codebookhistfilepath

    def codebookhistograms_save(self, featurehist, imgFile, debuglevel=0):
        """Saves codebook histograms"""
        codebookhistfile = self.gen_codebookhistogram_filepath(imgFile)

        codebookhistpath = os.path.dirname(codebookhistfile)

        if os.path.exists(codebookhistpath) == False:
            os.makedirs(codebookhistpath)
        numpy.save(codebookhistfile, featurehist)

    def codebookhistograms_load(self, imgFile, debuglevel=0):
        """Loads codebook histograms"""

        codebookhistfile = self.gen_codebookhistogram_filepath(imgFile)

        f = numpy.zeros((1, self.codebooksize))

        if os.path.exists(codebookhistfile) == False:
            return None
        else:
            f = numpy.load(codebookhistfile)
            return f


class ImageAnnotations:
    """ImageAnnotations class helps user to use image directories as ground
    truth classes for the images"""
    class_ids = []
    class_names = []

    def __init__(self, imgList):
        """Initialise image annotations object"""
        self.class_ids = []
        self.class_names = []
        self.imgList2annotations(imgList)

    def get_class_id(self, classname):
        """"""
        for i in range(0, len(self.class_names)):
            if self.class_names[i] == classname:
                return i+1
        return -1

    def get_images_for_class(self, imgList, classname):
        selectedimages = []
        cid = self.get_class_id(classname)
        if cid < 0:
            return selectedimages

        for i in range(0, len(self.class_ids)):
            if self.class_ids[i] == cid:
                selectedimages.append(imgList[i])
        return selectedimages

    def get_class_ids_vector(self):
        """Returns class ids as a vector"""
        return self.class_ids

    def get_training_test_set(self, imgList, classes, imgsTrain=30, imgsTest=20):
        """Returns training and testings sets for supervised learning purposes"""
        trainImages = []
        testImages = []
        trainGT = []
        testGT = []

        for cname in classes:
            cid = self.get_class_id(cname)
            imgs = self.get_images_for_class(imgList,cname)
            N = len(imgs)
            rp = numpy.random.permutation(N)
            for i in range(0,imgsTrain):
                # Check that we have enough samples for training
                if i > N:
                    break

                trainImages.append(imgs[rp[i]])
                trainGT.append(cid)

            for i in range(imgsTrain,imgsTrain+imgsTest):
                # Check that we have enough samples for training
                if i < N:
                    #break
                    testImages.append(imgs[rp[i]])
                    testGT.append(cid)

        trainGT = numpy.array(trainGT)
        testGT = numpy.array(testGT)

        return (trainImages,testImages,trainGT,testGT)

    def del_class(self, classname):
        cid = self.get_class_id(classname)
        # Del all ids relating to the class
        self.class_ids.remove(cid)
        self.class_names.remove(classname)

    def imgList2annotations(self, imgList):

        for imgFile in imgList:

            cid, self.class_names = self.imgList2annotations_add(
                os.path.dirname(imgFile),
                self.class_names)

            self.class_ids.append(cid)

        return (self.class_ids, self.class_names)

    def imgList2annotations_add(self, dirname, annotations):
        cnt = 0
        founded = False
        for annotation in annotations:
            cnt = cnt + 1
            if annotation == dirname:
                founded = True
                cid = cnt
                return (cid, annotations)

        if founded == False:
            annotations.append(dirname)
            cid = cnt + 1
            return (cid, annotations)
