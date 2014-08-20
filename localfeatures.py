# -*- coding: utf-8 -*-

# OS stuff
import os
# Numerical python
import numpy
# OpenCV library
import cv2

class LocalFeatures:
    """Local features class helps user to extract local features using
    various state-of-the-art local feature detectors implemented in OpenCV"""

    # Inputs
    detector = 'FAST'
    descriptor = 'FREAK'
    imgName = ''
    # Outputs
    frames = []
    descriptors = []

    def __init__(self, imgName, detector='FAST', descriptor='FREAK'):
        """Initialise Localfeature object"""
        self.imgName = imgName
        self.detector = detector
        self.descriptor = descriptor

    @staticmethod
    def extractfeatures(image, detector='FAST', descriptor='FREAK',
                        max_img_size=600, mask=None):
        """Extracts local features from a given image
         - Inputs:
            image           - image file or image as a matrix
            detector        - local feature detector from OpenCV (FAST,Dense,Orb,Harris,...)
            descriptor      - local feature descriptor
            max_img_size    - image is going to be scaled down to fit in this maximum width and height of the image
            mask            - Mask for selecting local features
         - Outputs:
            f            - feature frames i.e. keypoints/interest points
            d            - local feature descriptors"""

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
        if max_img_size < numpy.inf:
            h, w, c = I.shape
            ws = float(w) / max_img_size
            hs = float(h) / max_img_size
            s = max(ws, hs)
            I = cv2.resize(I, (int(w / s), int(h / s)))

        # Detect features / interest points
        # TODO: Fix mask
        f = detector.detect(I) #, mask)

        # And describe features
        [f, d] = descriptor.compute(I, f)

        # Compute root of the descriptor if root is set true
        if boolRootDesc is True:
            d = numpy.sqrt(d)

        return (f, d)


    def extract_from_imgDir(self, imgDir):
        """Extracts local features of a LocalFeatures object from
        a given imgDir"""
        [f,d] = LocalFeatures.extractfeatures(imgDir + '/' + self.imgName,
                                              self.detector,
                                              self.descriptor)

        # Store features and image name
        self.frames = f
        self.descriptors = d

        return (f, d)

    def save_descriptors(self, dataDir):
        'Save descriptors'
        dataFile = dataDir + '/localfeatures/' + self.detector + '/' + \
            self.descriptor + '/' + self.imgName

        dataPath = os.path.dirname(dataFile)

        if os.path.exists(dataPath) == False:
            os.makedirs(dataPath)


        if os.path.exists(dataFile + '.desc.npy') == False:
            numpy.save(dataFile + '.desc.npy', self.descriptors)

    def load_descriptors(self, dataDir):
        'load descriptors (should be moved to imagecollection class)'
        dataFile = dataDir + '/localfeatures/' + self.detector + '/' + \
            self.descriptor + '/' + self.imgName

        if os.path.exists(dataFile + '.desc.npy') == True:
            descs = numpy.load(dataFile + '.desc.npy', self.descriptors)
            self.descriptors = descs
            return descs

    def save_frames(self, dataDir):
        'save frames (should be moved to imagecollection class)'
        dataFile = dataDir + '/localfeatures/' + self.detector + '/' + \
            self.descriptor + '/' + self.imgName

        dataPath = os.path.dirname(dataFile)

        if os.path.exists(dataPath) == False:
            os.makedirs(dataPath)

        if os.path.exists(dataFile + '.frame.npy') == False:
            numpy.save(dataFile + '.frame.npy',
                self.keypoints2framematrix(self.frames))

    def load_featurespace(self, dataDir, detector='hesaff', descriptor='gloh'):
        """Reads and loads local features stored in featurespace format"""
        filename = dataDir + '/localfeatures/' + '/' + self.imgName + '.' + \
                   detector + '.' + descriptor + '.desc'

        if os.path.exists(filename):
            fp = file(filename)
            # Size of the descriptor
            try:
                d = int(fp.readline())
                # Number of descriptors
                N = int(fp.readline())

                # Init temp frames and temp descriptors
                ftmp = numpy.zeros((1, 5), dtype=numpy.float)
                dtmp = numpy.zeros((1, d), dtype=numpy.int)

                # Loop through all the descriptors
                for i in range(0, N):
                    line = fp.readline()
                    vals = line.split(' ')
                    #ftmp = []
                    #dtmp = []
                    # Parse strs to floats
                    for j in range(0, 5):
                        ftmp[0,j] = (float(vals[j]))
                    # Parse strs to ints
                    for j in range(5, 5 + d):
                        dtmp[0, j - 5] = (int(vals[j]))

                    if i > 0:
                        self.frames = numpy.vstack((self.frames, ftmp))
                        self.descriptors = numpy.vstack((self.descriptors, dtmp))
                    else:
                        self.frames = ftmp
                        self.descriptors = dtmp

            except Exception, e:
                print e
                self.frames = []
                self.descriptors = []

        else:
            print "could not open the file: " + filename + "!!1!"

    # This function read numpy-stored features
    def load(self, dataDir, detector='FAST', descriptor='FREAK'):
        """Loads local feature stored in npy (numpy) format"""

        if len(detector) == 0:
            detector = self.detector
        if len(descriptor) == 0:
            descriptor = self.descriptor

        f = []
        d = []

        localfeaturefile = dataDir + '/localfeatures/' + self.detector + '/' + \
            self.descriptor + '/' + self.imgName

        if os.path.exists(localfeaturefile + '.desc.npy'):
            d = numpy.load(localfeaturefile + '.desc.npy')

        if os.path.exists(localfeaturefile + '.key.npy'):
            f = numpy.load(localfeaturefile + '.key.npy')

        self.frames = f
        self.descriptors = d

    # Save features in numpy-format
    def save(self, dataDir):
        """Saves detected frames and descriptors of local features"""
        self.save_descriptors(dataDir)
        self.save_frames(dataDir)

    # Save frames in numpy-format
    def get_frames(self):
        """Returns detected frames (interest points)"""
        return self.frames

    # Save descriptors in numpy-format
    def get_descriptors(self):
        """Returns descriptions of the local features"""
        return self.descriptors

    # Match local features based on their descriptors
    def match_descriptors(self, d2, d1=[]):
        """Matches local features and returns matches"""
        if len(d1) == 0:
            d1 = self.descriptors
        [hits, d] = scipy.cluster.vq.vq(d1, d2)
        return (hits, d)


    def keypoints2framematrix(self, keypoints):
        """Converts OpenCV keypoint structure into a numpy array"""
        f = []
        for k in keypoints:
            f.append((k.pt[0], k.pt[1], k.angle, k.size, k.octave, k.response))
        return numpy.array(f)