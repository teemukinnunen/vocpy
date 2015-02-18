# -*- coding: utf-8 -*-

# OS stuff
import os
# Numerical python
import numpy
# For matching local features
import scipy.cluster.vq
# OpenCV library
import cv2

class LocalFeatures:
    """Local features class helps user to extract local features using
    various state-of-the-art local feature detectors implemented in OpenCV"""

    # Inputs
    detectorName = 'FAST'
    descriptorName = 'FREAK'
    boolRootDesc = False

    # Detector and descriptor objects
    detector = 0
    descriptor = 0

    # Outputs
    #frames = []
    #descriptors = []

    def __init__(self, detector='FAST', descriptor='FREAK'):
        """Initialise Localfeature object"""

        # Set important class variables
        self.detectorName = detector
        self.descriptorName = descriptor

        descriptor = descriptor.upper()

        if descriptor == 'ROOTSIFT':
            self.boolRootDesc = True
            self.descriptorName = 'SIFT'

        # Create detector and descriptor objects
        self.detector = cv2.FeatureDetector_create(self.detectorName)
        self.descriptor = cv2.DescriptorExtractor_create(self.descriptorName)

    def extract(self, image, max_img_size=600):
        """Extract local features from a given image"""

        if not type(image) == str:
            I = image
        else:
            # Read image
            I = cv2.imread(image)

        I = LocalFeatures.imscaledown(self, I, max_img_size)

        # Detect features / interest points
        # TODO: Fix mask
        f = self.detector.detect(I) #, mask)

        # And describe features
        [f, d] = self.descriptor.compute(I, f)

        # Compute root of the descriptor if root is set true
        if self.boolRootDesc is True:
            d = numpy.sqrt(d)

        return (f, d)

    @staticmethod
    def extractfeatures(image, detector='FAST', descriptor='FREAK',
                        max_img_size=600, mask=None):
        """Extracts local features from a given image
         - Inputs:
            image           - image file or image as a matrix
            detector        - local feature detector from OpenCV (FAST,Dense,
                                Orb,Harris,...)
            descriptor      - local feature descriptor from OpenCV (FREAK, SURF,
                                SIFT, BRIEF, ...)
            max_img_size    - image is going to be scaled down to fit in this
                                maximum width and height of the image
            mask            - Mask for selecting local features
         - Outputs:
            f            - feature frames i.e. keypoints/interest points
            d            - local feature descriptors"""

        """TODO: Maybe I should change from the generic feature detector method
        into detector based methods to gain more control over detector
        parameters.
        """
        if not type(image) == str:
            I = image
        else:
            # Read image
            I = cv2.imread(image)

        # Set mask for the images
        if mask == None:
            mask = numpy.ones((I.shape[0], I.shape[1]),
                            dtype=numpy.uint8)

        boolRootDesc = False

        descriptor = descriptor.upper()

        if descriptor == 'ROOTSIFT':
            boolRootDesc = True
            descriptor = 'SIFT'

        # Create detector and descriptor objects
        detector = cv2.FeatureDetector_create(detector)
        descriptor = cv2.DescriptorExtractor_create(descriptor)

        # Resize image1
        I = LocalFeatures.imscaledown(I, max_img_size)

        # Detect features / interest points
        # TODO: Fix mask
        f = detector.detect(I) #, mask)

        # And describe features
        [f, d] = descriptor.compute(I, f)

        # Compute root of the descriptor if root is set true
        if boolRootDesc is True:
            d = numpy.sqrt(d)

        return (f, d)

    def imscaledown(self, I, max_img_size=numpy.inf):
        """Resizes an input image to smaller image if necessary"""
        # Resize image1
        if max_img_size < numpy.inf:
            h, w, c = I.shape
            ws = float(w) / max_img_size
            hs = float(h) / max_img_size
            s = max(ws, hs)
            I = cv2.resize(I, (int(w / s), int(h / s)))

        return I

    def match_descriptors(self, d1, d2):
        """Matches local features and returns matches"""

        [hits, d] = scipy.cluster.vq.vq(d1, d2)

        return (hits, d)

    def keypoints2framematrix(self, keypoints):
        """Converts OpenCV keypoint structure into a numpy array"""
        f = []
        for k in keypoints:
            f.append((k.pt[0], k.pt[1], k.angle, k.size, k.octave, k.response))
        return numpy.array(f)


    def save_descriptors(self, dataDir, imgName, descriptors):
        """Save descriptors"""
        dataFile = dataDir + '/localfeatures/' + self.detectorName + '/' + \
            self.descriptorName + '/' + imgName

        dataPath = os.path.dirname(dataFile)

        if os.path.exists(dataPath) == False:
            os.makedirs(dataPath)

        if os.path.exists(dataFile + '.desc.npy') == False:
            numpy.save(dataFile + '.desc.npy', descriptors)

    def load_descriptors(self, dataDir, imgName):
        """load descriptors (should be moved to imagecollection class)"""
        dataFile = dataDir + '/localfeatures/' + self.detectorName + '/' + \
            self.descriptorName + '/' + imgName

        if os.path.exists(dataFile + '.desc.npy') == True:
            descsriptors = numpy.load(dataFile + '.desc.npy')
            return descsriptors

    def save_frames(self, dataDir, imgName, frames):
        """save frames (should be moved to imagecollection class)"""
        dataFile = dataDir + '/localfeatures/' + self.detectorName + '/' + \
            self.descriptorName + '/' + imgName

        dataPath = os.path.dirname(dataFile)

        if os.path.exists(dataPath) == False:
            os.makedirs(dataPath)

        if os.path.exists(dataFile + '.frame.npy') == False:
            numpy.save(dataFile + '.frame.npy',
                self.keypoints2framematrix(frames))

    def load(self, dataDir, imgName, detector='', descriptor=''):
        """Loads local feature stored in npy (numpy) format"""

        if len(detector) == 0:
            detector = self.detectorName
        if len(descriptor) == 0:
            descriptor = self.descriptorName

        # Initialize variables for frames (interest points) and descriptors
        frames = []
        descriptors = []

        localfeaturefile = dataDir + '/localfeatures/' + self.detectorName + '/' + \
            self.descriptorName + '/' + imgName

        if os.path.exists(localfeaturefile + '.desc.npy'):
            descriptors = numpy.load(localfeaturefile + '.desc.npy')

        if os.path.exists(localfeaturefile + '.key.npy'):
            frames = numpy.load(localfeaturefile + '.key.npy')

        return [frames, descriptors]

    # Save features in numpy-format
    def save(self, dataDir, imgName, frames, descriptors):
        """Saves detected frames and descriptors of local features"""
        self.save_descriptors(dataDir, imgName, descriptors)
        self.save_frames(dataDir, imgName, descriptors)
