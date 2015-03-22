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

        I = LocalFeatures.imscaledown(I, max_img_size)

        # Detect features / interest points
        f = self.detector.detect(I)

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

    @staticmethod
    def imscaledown(I, max_img_size=numpy.inf):
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

def match_images_spatially(image1, image2, detectorName='ORB',
    descriptorName='BRIEF', bestMatches=5, validationThreshold=10,
    mask1=None, mask2=None, boolDebugPlot=False):
    """ Matches local feature spatially

        Inputs:
            image1                Filename (str) or Image (cv2 array)
            image2                Filename (str) or Image (cv2 array)
            detectorName          Detector (def. 'ORB')
            descriptorName        Descriptor (def. 'BRIEF')
            bestMatches           Number of best descriptor matches (def. 5)
            validationThreshold   Maximal distance to accept match (def. 10)
            mask1                 Mask for detecting features (def. None)
            mask2                 Mask for detecting features (def. None)
            boolDebugPlot         Show debug plot in the end (def. False)
        Outputs:
            H                     Estimated Homography between the pair of imgs
            matchingFeatures      Indexes for matching features
            descDistances         Descriptor distances of matching features
    """
    # Set default values before doing anything
    H = numpy.zeros((3, 3))
    matchingFeatures = []
    descDistances = numpy.inf

    if isinstance(image1, str):
        image1 = cv2.imread(image1)
    if isinstance(image2, str):
        image2 = cv2.imread(image2)

    # Make sure that the descriptor name is written in uppercase
    descriptorName = descriptorName.upper()

    # Extract local features
    [ip1, lf1] = LocalFeatures.extractfeatures(image1,
                                                    detector=detectorName,
                                                    descriptor=descriptorName,
                                                    max_img_size=numpy.inf,
                                                    mask=mask1)
    [ip2, lf2] = LocalFeatures.extractfeatures(image2,
                                                    detector=detectorName,
                                                    descriptor=descriptorName,
                                                    max_img_size=numpy.inf,
                                                    mask=mask2)

    # Set local feature mathing method
    if descriptorName.endswith('SIFT') or descriptorName == 'SURF':
        binaryMatcher = False
    else:
        binaryMatcher = True

    if binaryMatcher:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, False)

    try:
        # Match local feature descriptors
        matches = matcher.knnMatch(lf1, lf2, bestMatches)
    except:
        print('Couldnt find any maching features. WTF!')
        return [H, matchingFeatures, descDistances]

    # Get max and min distances
    min_dist = 100
    max_dist = 0
    for i in range(0, len(matches)):
        if matches[i][0].distance > max_dist:
            max_dist = matches[i][0].distance
        if matches[i][0].distance < min_dist:
            min_dist = matches[i][0].distance

    # If min distance is close to zero, we need to do this hack
    #if min_dist <= max_dist / 0.001: # THIS IS TOO STRICT IN 99% cases
    if min_dist <= max_dist / 6:
        min_dist = max_dist / 6

    # Set threshold for accepting matches
    matchingThreshold = min_dist * 3

    # Filter out bad matches
    good_matches = []
    for i in range(0, len(matches)):
        #for j in range(0, bestMatches):
        for j in range(0, len(matches[i])):
            if matches[i][j].distance <= matchingThreshold:
                good_matches.append(matches[i][j])

    # Generate a list of 2d points for spatial matching
    scene1 = numpy.zeros((len(good_matches), 2))
    scene2 = numpy.zeros((len(good_matches), 2))

    for i in range(0, len(good_matches)):
        scene1[i, :] = ip1[good_matches[i].queryIdx].pt
        scene2[i, :] = ip2[good_matches[i].trainIdx].pt

    try:
        # Estimate homography transformation between the images using RANSAC
        [H, mask] = cv2.findHomography(scene1, scene2, cv2.RANSAC)
    except:
        print('Problem occured during initial RANSAC step!')
        return [H, matchingFeatures, descDistances]

    # Fix scene1 points to a correct format for the cv2.perspectiveTransform
    scene1 = numpy.array(scene1, dtype='float32')
    [n, m] = numpy.shape(scene1)
    scene1 = numpy.reshape(scene1, (1, n, m))

    # Transform scene1 points to scene2
    scene1t2 = cv2.perspectiveTransform(scene1, H)
    scene1t2 = numpy.reshape(scene1t2, (n, m))
    scene1t2 = numpy.array(scene1t2, dtype='float32')

    # Match transformed points
    matchingFeatures = []
    descDistances = []
    for i in range(0, len(scene1t2)):
        # Compute spatial distance between the original point in scene2 and
        # transformed point from scene1
        dist = scipy.spatial.distance.euclidean(scene2[i, :], scene1t2[i, :])
        if dist < validationThreshold:
            # Set matching features
            matchingFeatures.append((good_matches[i].queryIdx,
                                     good_matches[i].trainIdx))
            descDistances.append(good_matches[i].distance)


    #if boolDebugPlot:
    if boolDebugPlot:
        plot_matching_features(image1, image2, ip1, ip2, matchingFeatures)

    return [H, matchingFeatures, descDistances]

def plot_matching_features(image1, image2, ip1, ip2, matches, figid=101):
    """Plots matching features given as list of interest points ip1 and ip2"""

    import pylab

    [h1, w1, c1] = numpy.shape(image1)
    [h2, w2, c2] = numpy.shape(image2)
    I = numpy.zeros((max(h1, h2), w1 + w2))

    # convert to graylevel by taking mean
    I[0:h1, 0:w1] = numpy.reshape(numpy.mean(image1, 2), (h1, w1))
    I[0:h2, w1:w1 + w2] = numpy.reshape(numpy.mean(image2, 2), (h2, w2))

    # Set up figure
    fh = pylab.figure(figid)
    fh.clf()

    # Dsiplay graylevel imgs
    pylab.imshow(I)
    pylab.gray()
    pylab.axis('image')
    pylab.axis('tight')
    pylab.axis('off')

    # Plot matches
    #for i in range(0,len(matches)):
    for match in matches:
        [x1, y1] = ip1[match[0]].pt
        [x2, y2] = ip2[match[1]].pt
        x2 = x2 + w1
        pylab.plot((x1, x2), (y1, y2), 'ro-')

    pylab.show()
