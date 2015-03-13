# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#
# Dataset hanndling such as ImageCollection and their Annotations
#
#------------------------------------------------------------------------------

import os
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

    # -------------------------------------------------------------------------
    # Local feature functions wrapped to the imagecollection class
    # -------------------------------------------------------------------------
    def localfeatures_extract(self,debuglevel=0):
        """Extract local features from a given imgDir"""
        imgIdx = 0
        nImgs = len(self.imageNames)
        for imgFile in self.imageNames:
            if debuglevel > 0:
                print("Processing image %d of %d images." % (imgIdx, nImgs))

            localfeaturefile = self.dataDir + '/localfeatures/' + imgFile + \
                '.' + self.ipdetector + '.' + self.lfdescriptor

            # If the feature file already exists skip it
            if os.path.exists(localfeaturefile + '.key.npy') == False:
                lf = LocalFeatures(self.ipdetector,
                                    self.lfdescriptor)

                [f, d] = LocalFeatures.extractfeatures(self.imgDir + imgFile,
                                                self.ipdetector,
                                                self.lfdescriptor)

                localfeaturefiledir = os.path.dirname(localfeaturefile)

                if os.path.exists(localfeaturefiledir) == False:
                    os.makedirs(localfeaturefiledir)

                # Convert keypoints objects to matrix
                f = lf.keypoints2framematrix(f)

                numpy.save(localfeaturefile + '.desc', d)
                numpy.save(localfeaturefile + '.key', f)
            imgIdx += 1

    def localfeatures_read(self, debuglevel=0):
        """Load local features which are already extracted and return them """
        #TODO: Does this function make any sense? It kinda seems to have no logic
        features = numpy.zeros((0,0))
        for imageName in self.imageNames:
            try:
                #lf = LocalFeatures(imageName, self.ipdetector, self.lfdescriptor)
                localfeaturefile = self.dataDir + '/localfeatures/' + imageName + \
                '.' + self.ipdetector + '.' + self.lfdescriptor

                if not os.path.isfile(localfeaturefile + '.desc.npy'):
                    print(("Local feature file does not exist (%s)" % (localfeaturefile + '.desc.npy')))

                if features == None or features.size == 0:
                    #features = lf.load_descriptors(self.dataDir)
                    features = features = numpy.load(localfeaturefile + '.desc.npy')
                else:
                    features = numpy.vstack((features,
                                            numpy.load(localfeaturefile + '.desc.npy')))
            except ValueError:
                print(("Problems with image: " + imageName))

        return features

    def localfeatures_load_matlab(self, detector='hesaff', descriptor='gloh'):
        """Reads local features stored in a matlab .mat format"""
        for imageFile in self.imageNames:
            lf = LocalFeatures(imageFile)
            lf.load_featurespace(self.dataDir, detector, descriptor)
            self.localfeatures.append(lf)

    # -------------------------------------------------------------------------
    # Codebook generation function wrapped inside the collection class
    # -------------------------------------------kmeans = sc.KMeans(10)------------------------------
    def codebook_generate(self, debuglevel=0):
        """Generates a codebook"""
        features = self.localfeatures_read()
        codebook = Codebook(codebookmethod=self.codebookmethod,
                            codebooksize=self.codebooksize)

        codebook.generate(features)

        self.codebook_save(codebook)

        return codebook

    def codebook_save(self, codebook, optionalFilename='', debuglevel=0):
        """Saves the generated cocebook"""

        # Define filename for the codebook
        if len(optionalFilename) == 0:
            codebookfile = self.dataDir + '/codebooks/' + self.ipdetector + '.' + \
                self.lfdescriptor + '.' + self.codebookmethod + '.' + \
                str(self.codebooksize) + '.npy'
        else:
            codebookfile = self.dataDir + '/codebooks/' + optionalFilename

        codebookpath = os.path.dirname(codebookfile)

        if os.path.exists(codebookpath) == False:
            os.makedirs(codebookpath)

        numpy.save(codebookfile, codebook.codebook)

    def codebook_load(self, optionalFilename='', debuglevel=0):
        """Loads the generated codebook"""
        # Define filename for the codebook
        if len(optionalFilename) == 0:
            codebookfile = self.dataDir + '/codebooks/' + self.ipdetector + '.' + \
                self.lfdescriptor + '.' + self.codebookmethod + '.' + \
                str(self.codebooksize) + '.npy'
        else:
            codebookfile = self.dataDir + '/codebooks/' + optionalFilename

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

        codebookhistograms = numpy.zeros((0, 0))

        for imageFile in self.imageNames:
            # Try to load codebookhistogram
            [codebookhist, isLoaded] = self.codebookhistograms_load(imageFile)

            # If codebookhistograms is not being loaded, we need to compute one
            if isLoaded == False:
                #Lf = LocalFeatures(imageFile,
                #                   self.ipdetector,
                #                   self.lfdescriptor)
                #desc = Lf.load_descriptors(self.dataDir)
                localfeaturefile = self.dataDir + '/localfeatures/' + imageFile + \
                '.' + self.ipdetector + '.' + self.lfdescriptor
                desc = numpy.load(localfeaturefile + '.desc.npy')

                # Compute codebookhistogram
                Codebookhist = CodebookHistograms()
                codebookhist = Codebookhist.generate(codebook, desc)

                # Store histogram
                self.codebookhistograms_save(codebookhist, imageFile)

            # Stack codebookhistograms into a matrix
            if codebookhistograms.size == 0:
                codebookhistograms = codebookhist
            else:
                try:
                    codebookhistograms = numpy.vstack((codebookhistograms,
                                                    codebookhist))
                except:
                    print "Couldnt concatenate codebook histogram to feature matrix"
                    print codebookhistograms.shape
                    print codebookhist.shape
                    print codebook.codebooksize
                    if codebookhist.size == 0:
                        print "Codebookhistogram is zero size!! Which does not make any sense."
                        print "Changing codebookhistogram to 1 x CB size filled with zeros"
                        codebookhist = numpy.zeros((1, codebookhistograms.shape[1]))
                        try:
                            codebookhistograms = numpy.vstack((codebookhistograms,
                                                    codebookhist))
                        except:
                            print "Did not work out as planned... dyiing..."
                            1/0
                    else:
                        try:
                            codebookhistograms = numpy.vstack((codebookhistograms,
                                                            codebookhist.transpose))
                        except:
                            print "Still failing.. dying.."
                            1/0

        return codebookhistograms

    def codebookhistograms_save(self, featurehist, imgFile, debuglevel=0):
        """Saves codebook histograms"""
        codebookhistfile = self.dataDir + \
                            '/codebookhistograms/' + \
                            self.ipdetector + '/' + \
                            self.lfdescriptor + '/' + \
                            self.codebookmethod + \
                            '/' + str(self.codebooksize) + \
                            '/' + imgFile + '.npy'

        codebookhistpath = os.path.dirname(codebookhistfile)

        if os.path.exists(codebookhistpath) == False:
            os.makedirs(codebookhistpath)
        numpy.save(codebookhistfile, featurehist)

    def codebookhistograms_load(self, imgFile, debuglevel=0):
        """Loads codebook histograms"""
        codebookhistfile = self.dataDir + \
                            '/codebookhistograms/' + \
                            self.ipdetector + '/' + \
                            self.lfdescriptor + '/' + \
                            self.codebookmethod + \
                            '/' + str(self.codebooksize) + \
                            '/' + imgFile + '.npy'

        f = numpy.zeros((1, self.codebooksize))

        if os.path.exists(codebookhistfile) == False:
            return (f, False)
        else:
            f = numpy.load(codebookhistfile)
            return (f, True)


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