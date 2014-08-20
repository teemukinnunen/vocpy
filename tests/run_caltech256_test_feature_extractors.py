#!/usr/bin/python2

# -*- coding: utf-8 -*-

import os
import sys
from sklearn import svm

# Add path to
sys.path.append('../')

# Import Codebook and CodebookHistograms
from codebook import *
# Import ImageCollection and ImageAnnotations
from datasets import *
# Import LocalFeatures
from localfeatures import *
#
import clustering

# TODO: Set these paths accordingly
imageDir = os.environ['IMAGESETDIR']+'/256_ObjectCategories/'
dataDir = os.environ.get('DATADIR', '/tmp/'+os.environ['USER']+'/data') + '/c256'

nTrain = 50
nTest = 20
nClasses = 10
codebooksize_ = 500

ipdetectors = ["FAST", "STAR", "SIFT", "SURF", "ORB", "BRISK", "MSER", "GFTT", "HARRIS", "Dense", "SimpleBlob"]
lfdescriptors = ["SIFT", "SURF", "BRIEF", "BRISK", "ORB", "FREAK"]

testId = int(sys.argv[1])
ipId = numpy.mod(testId,len(ipdetectors))
lfId = int(numpy.floor(testId/len(ipdetectors)))

ipdetector = ipdetectors[ipId]
lfdescriptor = lfdescriptors[lfId]
codebookmethod_ = 'MiniBatchKMeans'

# Get a list of images
imageSet = ImageCollection(imageDir)
# Get annotations based on image sub directories
imageAnnotations = ImageAnnotations(imageSet.imageNames)
# A list of object classes
classes = imageAnnotations.class_names[0:nClasses]

[trainImages,testImages,trainGT,testGT] = imageAnnotations.get_training_test_set(imageSet.imageNames,classes,nTrain,nTest)

trainingSet = ImageCollection(imageDir,
                                dataDir,
                                ipdetector=ipdetector,
                                lfdescriptor=lfdescriptor,
                                codebookmethod=codebookmethod_,
                                codebooksize=codebooksize_)
trainingSet.imageNames = trainImages

testingSet = ImageCollection(imageDir,
                            dataDir,
                            ipdetector=ipdetector,
                            lfdescriptor=lfdescriptor,
                            codebookmethod=codebookmethod_,
                            codebooksize=codebooksize_)
testingSet.imageNames = testImages

# Extract local features
print("Extracting local features..")
trainingSet.localfeatures_extract(debuglevel=0)
testingSet.localfeatures_extract(debuglevel=0)
localfeatures = trainingSet.localfeatures_read()

resultFile = 'c256_feature_detector_test_' + ipdetector + '_' + lfdescriptor + '.txt'
print(resultFile)

f = open(resultFile,'a')

print("Computing new codebook with: %d codes" % codebooksize_)
codebook = Codebook(detector=ipdetector,descriptor=lfdescriptor,codebooksize=codebooksize_, codebookmethod=codebookmethod_)
codebook = codebook.generate(localfeatures)

#
print("Computing codebook histograms..")
trainingFeatures = trainingSet.codebookhistograms_generate(codebook.codebook)
trainingFeatures = CodebookHistograms.normalise(trainingFeatures,2)
testingFeatures = testingSet.codebookhistograms_generate(codebook.codebook)
testingFeatures = CodebookHistograms.normalise(testingFeatures,2)

#
print("Training a SVM classifier...")
svc = svm.SVC(C=1000, kernel='linear')
svc.fit(trainingFeatures,trainGT)
print("Testing the SVM classifier...")
p = svc.predict(testingFeatures)

f.write('%d \t %d \t %d \t %1.5f\n' % (testId, len(numpy.unique(trainGT)), codebooksize_, float((p==testGT).sum()) / float(len(p))) )

#
print trainingFeatures.shape
print("Processing done.. evaluation..")
print "Accuracy is: " + str(float((p==testGT).sum()) / float(len(p))) + \
      " with " + str(len(numpy.unique(trainGT))) + " classes."

f.close()