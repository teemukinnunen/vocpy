#==============================================================================
# Script for running a test for automatic image news value assesment.
#
# It uses various features such as:
# --bof histograms
# --color histograms
#
# For learning it uses Support Vector Regression (SVR) with either RBF, poly or
# linear kernel.
#
# Author: teemu . kinnunen -at- aalto . fi
#
#==============================================================================

# Import some generic things
import sys
import os
# OpenCV
import cv2
# Numerical python
import numpy as np
# For matplotlib for plotting
import pylab
# Skicit-learn for machine learning things
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

# Add path to necessary custom libraries
sys.path.append('../')
# Color histograms
import lowlevelfeatures as llf
# File input/output
import fileio as fio
# Import Codebook and CodebookHistograms
from codebook import *
# Import ImageCollection and ImageAnnotations
from datasets import *
# Import LocalFeatures
from localfeatures import *

# TODO: Set paths accordingly
imageDir = '/home/tekinnun/wrk/projects/newsvalue/images/'
dataDir = '/home/tekinnun/wrk/projects/newsvalue/data/'

imagelist = fio.read_imagelist_file(imageDir + 'imagelist.txt')

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [0.1, 1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
codebooksize_ = 0
codebookmethod_ = 'MiniBatchKMeans'
ipdetector_='PyramidHARRIS'
lfdescriptor_='SIFT'
#------------------------------------------------------------------------------
# Divide images into training and testing sets
#------------------------------------------------------------------------------

# Traditional 80/20% split
testIdx = range(0,len(imagelist),5)
trainIdx = np.setxor1d(range(0,90),testIdx)
results = []


# Generate train image list
trainImageList = []
trainingSet = ImageCollection(imgDir=imageDir,
                                dataDir=dataDir,
                                ipdetector=ipdetector_,
                                lfdescriptor=lfdescriptor_,
                                codebooksize=codebooksize_,
                                codebookmethod=codebookmethod_)
for id in trainIdx:
    trainImageList.append(imagelist[id])
trainingSet.imageNames = trainImageList

# Generate test image list
testImageList = []
testingSet = ImageCollection(imgDir=imageDir,
                                dataDir=dataDir,
                                ipdetector=ipdetector_,
                                lfdescriptor=lfdescriptor_,
                                codebooksize=codebooksize_,
                                codebookmethod=codebookmethod_)
for id in testIdx:
    testImageList.append(imagelist[id])
testingSet.imageNames = testImageList

D = fio.read_news_value_ground_truth_csv(dataDir + '/ground_truth/et_uutisarvo_arviot.csv')
gtc = np.mean(D[:,5:95], 0)

#------------------------------------------------------------------------------
# Extract features
#------------------------------------------------------------------------------

## BOF FEATURES
trainingSet.localfeatures_extract(debuglevel=0)
testingSet.localfeatures_extract(debuglevel=0)

## BUILD CODEBOOKS
for codebooksize_ in [10, 20, 50, 100, 200, 500, 1000, 2000]:
    # Update codebooksize
    trainingSet.codebooksize = codebooksize_
    testingSet.codebooksize = codebooksize_

    print("Computing new codebook with: %d codes" % codebooksize_)
    codebook = Codebook(detector=ipdetector_,
                        descriptor=lfdescriptor_,
                        codebooksize=codebooksize_,
                        codebookmethod=codebookmethod_)
    localfeatures = trainingSet.localfeatures_read()

    codebook = codebook.generate(localfeatures)

    # Gen feature histograms
    trainingFeatures = trainingSet.codebookhistograms_generate(codebook)
    trainingFeatures = CodebookHistograms.normalise(trainingFeatures,2)
    testingFeatures = testingSet.codebookhistograms_generate(codebook)
    testingFeatures = CodebookHistograms.normalise(testingFeatures,2)

    #------------------------------------------------------------------------------
    # Learn models
    #------------------------------------------------------------------------------
    trainGT = gtc[trainIdx]
    testGT = gtc[testIdx]


    clf = GridSearchCV(svm.SVR(C=1), tuned_parameters, cv=5)

    clf.fit(trainingFeatures, trainGT)
    p = clf.predict(testingFeatures)

    #print("predicted values:")
    #print(p)
    #print("grount truth")
    #print(testGT)
    #print("Errors")
    #print(np.abs(p-testGT))

    #------------------------------------------------------------------------------
    # Evaluate models
    #------------------------------------------------------------------------------

    perf_r2 =  metrics.r2_score(testGT, p)
    perf_mae = metrics.mean_absolute_error(testGT, p)
    perf_mse = metrics.mean_squared_error(testGT, p)

    results.append([codebooksize_, perf_r2, perf_mae, perf_mse])

    print(("Codebooksize = %4d \t R2 = %1.4f Mean absolute err=%1.4f MSE=%1.4f" % (codebooksize_, perf_r2, perf_mae, perf_mse)))

    #------------------------------------------------------------------------------
    # Report results
    #------------------------------------------------------------------------------
    if False:
        pylab.figure()
        pylab.hold('on')
        pylab.plot(clf.predict(trainingFeatures), trainGT,'bo')
        pylab.plot(p, testGT,'rx')
        pylab.axis([1,10,1,10])
        pylab.plot([1,10],[1,10],'k-')
        pylab.grid('on')
        pylab.show()

print()
print(results)
