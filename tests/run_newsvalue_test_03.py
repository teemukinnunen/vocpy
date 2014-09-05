# -*- coding: utf-8 -*-
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

def read_attention_ground_truth_feature(dataDir, imagename, d=100):
    featdir = dataDir + 'ground_truth/attentionmaps'

    I = cv2.imread(featdir + '/' + trainingSet.imageNames[0] + '.png')

    # It is actually a grey-level image, and thus, we can use first dimension
    I = I[:,:,0]
    dx = int(np.ceil(np.sqrt(d)))
    I = cv2.resize(I,(dx,dx))

    # Make it a vector
    I.resize(1,I.size)

    return I

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


## Ground truth news value
D = fio.read_news_value_ground_truth_csv(dataDir + '/ground_truth/et_uutisarvo_arviot.csv')
gtc = np.mean(D[:,5:95], 0)

trainGT = gtc[trainIdx]
testGT = gtc[testIdx]


#------------------------------------------------------------------------------
# Extract features
#------------------------------------------------------------------------------

## READ FEATURES
for ndims in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:

    trainingFeatures = []
    for imagename in trainingSet.imageNames:
        h = read_attention_ground_truth_feature(dataDir, imagename, ndims)
        trainingFeatures.append(h)
    trainingFeatures = np.vstack(trainingFeatures)

    testingFeatures = []
    for imagename in testingSet.imageNames:
        h = read_attention_ground_truth_feature(dataDir, imagename, ndims)
        testingFeatures.append(h)
    testingFeatures = np.vstack(testingFeatures)

    #------------------------------------------------------------------------------
    # Learn models
    #------------------------------------------------------------------------------

    clf = GridSearchCV(svm.SVR(C=1), tuned_parameters, cv=5)

    print(trainingFeatures.shape)

    clf.fit(trainingFeatures, trainGT)
    p = clf.predict(testingFeatures)

    if False:
        clf2 = svm.SVR(C=clf.best_params_['C'],
                        gamma=clf.best_params_['gamma'],
                        kernel=clf.best_params_['kernel'])
        clf2.fit(trainingFeatures, trainGT)
        p2 = clf2.predict(testingFeatures)
        print(p-p2)

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

    print(("Ndims = %4d \t R2 = %1.4f Mean absolute err=%1.4f MSE=%1.4f" % (ndims, perf_r2, perf_mae, perf_mse)))

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
