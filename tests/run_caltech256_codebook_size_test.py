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
nclasses = 50

ipdetector='PyramidHARRIS'
lfdescriptor='SIFT'


# Get a list of images
imageSet = ImageCollection(imageDir)
# Get annotations based on image sub directories
imageAnnotations = ImageAnnotations(imageSet.imageNames)
# A list of object classes
classes = imageAnnotations.class_names[0:nclasses]

f = open('c256_somoclus_codebooksize_test.txt','a')

for trial in range(1,3):
    [trainImages,testImages,trainGT,testGT] = imageAnnotations.get_training_test_set(imageSet.imageNames,classes,nTrain,nTest)

    trainingSet = ImageCollection(imageDir, dataDir, ipdetector=ipdetector,lfdescriptor=lfdescriptor,codebookmethod='som')
    trainingSet.imageNames = trainImages

    testingSet = ImageCollection(imageDir, dataDir, ipdetector=ipdetector,lfdescriptor=lfdescriptor,codebookmethod='som')
    testingSet.imageNames = testImages

    # Extract local features
    print("Extracting local features..")
    trainingSet.localfeatures_extract(debuglevel=0)
    testingSet.localfeatures_extract(debuglevel=0)
    localfeatures = trainingSet.localfeatures_read()


    for codebooksize_  in [100,200,500,1000,2000,5000,10000]:
        trainingSet.codebooksize = codebooksize_
        testingSet.codebooksize = codebooksize_

        print("Computing new codebook with: %d codes" % codebooksize_)
        codebook = Codebook(detector=ipdetector,descriptor=lfdescriptor,codebooksize=codebooksize_, codebookmethod='som')
        [codebook_, bmus, U] = clustering.som(localfeatures,nSomX=codebooksize_,nSomY=1)
        codebook_ = codebook_.reshape(codebooksize_,128)

        #
        #print("Computing codebook histograms..")
        trainingFeatures = trainingSet.codebookhistograms_generate(codebook_)
        trainingFeatures = CodebookHistograms.normalise(trainingFeatures,2)
        testingFeatures = testingSet.codebookhistograms_generate(codebook_)
        testingFeatures = CodebookHistograms.normalise(testingFeatures,2)

        #
        #print("Training a SVM classifier...")
        svc = svm.SVC(C=1000, kernel='linear')
        svc.fit(trainingFeatures,trainGT)
        #print("Testing the SVM classifier...")
        p = svc.predict(testingFeatures)

        f.write('%d \t %d \t %d \t %1.5f\n' % (trial, len(numpy.unique(trainGT)), codebooksize_, float((p==testGT).sum()) / float(len(p))) )

        #
        print trainingFeatures.shape
        #print("Processing done.. evaluation..")
        print "Accuracy is: " + str(float((p==testGT).sum()) / float(len(p))) + \
              " with " + str(len(numpy.unique(trainGT))) + " classes."

f.close()