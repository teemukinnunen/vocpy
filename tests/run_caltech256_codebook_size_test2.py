#!/usr/bin/python2

# -*- coding: utf-8 -*-

import os
import sys
from sklearn import svm
import scipy.spatial as ss
import numpy as np

# Add path to
sys.path.append('../')

# Import Codebook and CodebookHistograms
from codebook import *
# Import ImageCollection and ImageAnnotations
from datasets import *
# Import LocalFeatures
from localfeatures import *

# TODO: Set these paths accordingly
imageDir = os.environ['IMAGESETDIR']+'/256_ObjectCategories/'
dataDir = os.environ.get('DATADIR', '/tmp/'+os.environ['USER']+'/data') + '/c256'
resultDir = os.environ.get('RESULTDIR')

# Get test id as input parameter
testId = int(sys.argv[1])

datasetname = 'c256'
nTrain = 50
nTest = 20
nClasses = 10
codebooksizes = [100, 200, 500, 1000, 2000, 5000]
codebooksize_ = codebooksizes[testId]

ipdetector = "PyramidHARRIS"
lfdescriptor = "SIFT"
codebookmethod_ = 'MiniBatchKMeans'
hnormalisation = 2

# Make result dir
resultDir = os.path.realpath(resultDir + '/' +
                            datasetname + '_' + str(nClasses) + '/' +
                            ipdetector + '_' + lfdescriptor + '/' +
                            codebookmethod_ + '_' + str(codebooksize_) + '_L' + str(hnormalisation))

if not os.path.exists(resultDir):
    os.makedirs(resultDir)

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

# Store image names in the result dir for debugging purposes
f = open(resultDir + '/trainingimages.txt','w')
for imagename in trainingSet.imageNames:
    f.write('%s\n' % imagename)
f.close()

np.save(resultDir + '/training_classes.npy', trainGT)

f = open(resultDir + '/testingimages.txt','w')
for imagename in testingSet.imageNames:
    f.write('%s\n' % imagename)
f.close()

np.save(resultDir + '/training_classes.npy', testGT)

# Extract local features
print("Extracting local features..")
trainingSet.localfeatures_extract(debuglevel=0)
testingSet.localfeatures_extract(debuglevel=0)
localfeatures = trainingSet.localfeatures_read()

resultFile = os.path.realpath(resultDir + '/classification_perf.txt')
print(resultFile)

f = open(resultFile,'a')

print("Computing new codebook with: %d codes" % codebooksize_)
codebook = Codebook(detector=ipdetector,descriptor=lfdescriptor,codebooksize=codebooksize_, codebookmethod=codebookmethod_)
codebook = codebook.generate(localfeatures)

#
print("Computing codebook histograms..")
trainingFeatures = trainingSet.codebookhistograms_generate(codebook)
trainingFeatures = CodebookHistograms.normalise(trainingFeatures,2)
testingFeatures = testingSet.codebookhistograms_generate(codebook)
testingFeatures = CodebookHistograms.normalise(testingFeatures,2)

# save training and testing features
np.save(resultDir + '/trainingFeatures.npy', trainingFeatures)
np.save(resultDir + '/testingFeatures.npy', testingFeatures)

# Compute distance matrix between each training sample
D = ss.distance_matrix(trainingFeatures, trainingFeatures)
np.save(resultDir + '/feature_histogram_distances.npy', D)

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