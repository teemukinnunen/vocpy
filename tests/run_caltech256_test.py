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

# TODO: Set these paths accordingly
imageDir = os.environ['IMAGESETDIR']+'/256_ObjectCategories/'
dataDir = os.environ.get('DATADIR', '/tmp/'+os.environ['USER']+'/data') + '/c256'

nTrain = 30
nTest = 10


codebooksize_ = 1000
codebookmethod_ = 'MiniBatchKMeans'
ipdetector='HARRIS'
lfdescriptor='SIFT'


# Get a list of images
imageSet = ImageCollection(imageDir)
# Get annotations based on image sub directories
imageAnnotations = ImageAnnotations(imageSet.imageNames)
# A list of object classes
classes = imageAnnotations.class_names[0:5]

[trainImages,testImages,trainGT,testGT] = imageAnnotations.get_training_test_set(imageSet.imageNames,classes,nTrain,nTest)

trainingSet = ImageCollection(imageDir, dataDir, ipdetector=ipdetector,lfdescriptor=lfdescriptor,codebookmethod='som')
trainingSet.imageNames = trainImages

testingSet = ImageCollection(imageDir, dataDir, ipdetector=ipdetector,lfdescriptor=lfdescriptor,codebookmethod='som')
testingSet.imageNames = testImages

# Extract local features
print("Extracting local features..")
trainingSet.localfeatures_extract(debuglevel=0)
testingSet.localfeatures_extract(debuglevel=0)


# Load previously generated codebook
print("Computing new codebook with: %d codes" % codebooksize_)
localfeatures = trainingSet.localfeatures_read()
codebook = Codebook(detector=ipdetector,descriptor=lfdescriptor,codebooksize=codebooksize_, codebookmethod=codebookmethod_)
codebook = codebook.generate(localfeatures)

#
print("Computing codebook histograms..")
trainingFeatures = trainingSet.codebookhistograms_generate(codebook)
#trainingFeatures = CodebookHistograms.normalise(trainingFeatures,2)
testingFeatures = testingSet.codebookhistograms_generate(codebook)
#testingFeatures = CodebookHistograms.normalise(testingFeatures,2)

#
print("Training a SVM classifier...")
svc = svm.SVC(C=1000, kernel='linear')
svc.fit(trainingFeatures,trainGT)
print("Testing the SVM classifier...")
p = svc.predict(testingFeatures)

#
print("Processing done.. evaluation..")

print "Accuracy is: " + str(float((p==testGT).sum()) / float(len(p))) + \
      " with " + str(len(numpy.unique(trainGT))) + " classes."

from sklearn.naive_bayes import MultinomialNB as bayes

clf = bayes()
clf.fit(trainingFeatures, trainGT)
bayes(alpha=1.0, class_prior=None, fit_prior=True)
p2 = clf.predict(testingFeatures)

print(("Accuracy is: " + str(float((p2==testGT).sum()) / float(len(p2))) + \
      " with " + str(len(numpy.unique(trainGT))) + " classes."))