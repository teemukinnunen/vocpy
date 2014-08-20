# -*- coding: utf-8 -*-

import numpy
import scipy.spatial as ss

class KNN:
    trainingSamples = []
    trainingClasses = []

    def __init__(self, trainingSamples_, trainingClasses_):
        # Learning
        self.trainingSamples = trainingSamples_
        self.trainingClasses = trainingClasses_

    def predictClasses(self, testSamples_):
        # Compute distance between test samples and training samples
        D = ss.distance_matrix(testSamples_, self.trainingSamples)

        # Get the closest match
        idx = numpy.argmin(D,1)

        # Get class ids for each closest match
        C = self.trainingClasses[idx]

        return C

def compute_distance_matrix(F1, F2):
    D = ss.distance_matrix(F1, F2)
    return D

def compute_k_best_correct_ratios(D, gtC):
    "Computes a ratio of best matches for each image"
    # Make sure that we dont have zero distance to training sample itself
    D = D + numpy.eye(len(gtC))*D.max()+1

    idx = numpy.argsort(D,axis=1)
    # ""Predicted classes""
    P = gtC[idx]
    # Initialize array for computing ratios
    r = numpy.empty([len(gtC),len(gtC)],numpy.float)
    # Number of best matches (used for dividision)
    K = numpy.array(range(1,len(gtC)+1),dtype=numpy.float)
    for i in range(0,len(gtC)):
            r[i, :] = numpy.array(numpy.cumsum(P[i,:]==gtC[i]),dtype=numpy.float) / K

    return r

def compute_miss_streak(r, gtC):
    "Computes how many misses is made"

    misstreak = numpy.zeros((len(gtC),1), dtype=numpy.int)

    for i in range(0,len(gtC)):
        m = numpy.argwhere(r[i,:]>0)
        misstreak[i] = m[0]

    return misstreak

def eliminate_poor_training_samples(D, gtC, max_miss_streak = 3):
    """Eliminates bad training samples ie. samples which are confusing training
    a classifier. Inputs are the distance matrix (computed from training features
    and ground truth classes for trainign samples/features."""

    r = compute_k_best_correct_ratios(D, gtC)
    misstreak = compute_miss_streak(r, gtC)

    while numpy.max(misstreak) > max_miss_streak:
        # Get id for the worst sample
        bad_sample_id = numpy.argmax(misstreak)
        print "The worst sample is: " + str(bad_sample_id) + " with ms: " + str(misstreak[bad_sample_id])
        # Delete rotten sample from distance matrix
        D = numpy.delete(D,bad_sample_id,0)
        D = numpy.delete(D,bad_sample_id,1)
        # Delete ground truth class info
        gtC = numpy.delete(gtC,bad_sample_id,0)
        # Repeat elimination if needed
        r = compute_k_best_correct_ratios(D, gtC)
        misstreak = compute_miss_streak(r, gtC)

    bad_sample_id = numpy.argmax(misstreak)
    print "The worst sample is: " + str(bad_sample_id) + " with ms: " + str(misstreak[bad_sample_id])
    print numpy.max(misstreak)

    return [D, gtC]


def find_sub_categories(D, trainGT):
    "Finds groups of training images which matches k-NN best correctly"

    D = D + numpy.eye(D.shape[0])+numpy.max(D)

    idx = numpy.argsort(D,0)

    matches = []

    for i in range(0,D.shape[0]):
        Fidx = numpy.argwhere(trainGT[idx[i,:]] != trainGT[i])
        good_matches = idx[i,0:Fidx[0]]
        matches.append( good_matches )

def compute_accuracy(pC, gtC):
    "Computer accuracy for predicted Classes and ground truth Classes"

    # True positives
    TP = sum(pC==gtC)
    # Number of test samples
    N = len(pC)
    # Accuracy
    acc = float(TP)/float(N)

    return acc