# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#
# Codebook and codebook histogram compuation
#
#------------------------------------------------------------------------------

import numpy
import cv2
import os
import sys
import scipy
import scipy.cluster.vq
from sklearn.cluster import KMeans

class Codebook:
    detector = ''
    descriptor = ''
    codebooksize = 10000
    codebookmethod = 'som'
    codebook = []
    codebookname = ''

    def __init__(self, detector='hesaff', descriptor='gloh', codebooksize=1000,
            codebookmethod='som'):
        self.detector = detector
        self.descriptor = descriptor
        self.codebooksize = codebooksize
        self.codebookmethod = codebookmethod
        self.codebookname = self.detector + '_'  + \
                            self.descriptor + '_' + \
                            self.codebookmethod + '_' + \
                            str(self.codebooksize)

    def load_text_somoclu(self, filename):
        f = open(filename,'r')

        line = f.readline()
        [N,M] = line[1:].split(' ')
        N = int(N)
        M = int(M)
        line = f.readline()
        d = int(line[1:])

        self.codebook = numpy.empty([N,M,d], dtype=float)

        for i in range(0,N):
            for j in range(0,M):

                line = f.readline()
                vals = line.split(' ')

                # Verify that we have enough numbers
                if len(vals)-1 is not d:
                    print('d=%d ? len(vals)=%d' % (d, len(vals)))

                for k in range(0, d):
                     self.codebook[i,j,k] = float(vals[k])


        # Reshape codebook to N x d
        self.codebook = numpy.reshape(self.codebook, (N*M, d))

    def load_matlab(self, dataDir):

        filename = dataDir + '/codebooks/' + self.detector + '_' + \
            self.descriptor + '_' + self.codebookmethod + '_' + \
            str(self.codebooksize) + '.mat'

        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            self.codebook = data['codebook']
        else:
            print("Could not load the damn codebook!")

    def compute_histogram(self, descriptors):
        # Check if the list of descriptors is empty we can return zero hist

        if len(descriptors) == 0:
            return numpy.zeros((self.codebooksize, 0))

        [hits, d] = scipy.cluster.vq.vq(descriptors, self.codebook)
        [y, x] = numpy.histogram(hits, bins=range(0,self.codebooksize + 1))

        return y

    def generate(self, features):
        """Codebook generation function which takes a list of features as input
        and returns codebook"""

        #TODO: Add other codebook generation methods

        if self.codebookmethod == "MiniBatchKMeans":
            # Import MiniBatchKMeans from sklearn package and hope that it is
            # available :
            from sklearn.cluster import MiniBatchKMeans
            # Set parameters
            mbk = MiniBatchKMeans(init='k-means++', n_clusters=self.codebooksize,
                batch_size=self.codebooksize * 3, n_init=3, max_iter=50,
                max_no_improvement=3, verbose=0, compute_labels=False)
            # Cluster data points
            mbk.fit(features)
            self.codebook = mbk.cluster_centers_
        elif self.codebookmethod == "KMeans":
            import scipy.cluster.vq
            [codebook, label] = scipy.cluster.vq.kmeans2(features,
                                                        self.codebooksize,
                                                        iter=100,
                                                        minit='points',
                                                        missing='warn')
            self.codebook = codebook
        else:
            print("Unknown codebook method: %s" % self.codebookmethod)
            sys.exit(-1)

        return self.codebook

    def compute_histogram(self, codebook, features):
        [N, d] = codebook.shape
        if features.size <= 1:
            return numpy.zeros((N, 0))

        [hits, d] = scipy.cluster.vq.vq(features, codebook)
        [y, x] = numpy.histogram(hits, bins=range(0, N + 1))
        return y

    @staticmethod
    def hkmeans(data, branching=100, depth=10, max_data_points=100000):
        """Hiearchical k-means"""


        #print numpy.shape(data)
        # If there is too many data points, choose max_data_points randomly
        [N,d] = numpy.shape(data)
        rp = range(0,N)
        if N > max_data_points:
            rp = numpy.random.permutation(N)
            rp = rp[0:max_data_points]

        # If the number
        if N < branching:
            clusters = data
            subclusters = []
            clusterNode = Codebook.ClusterNode(clusters,subclusters)
            return clusterNode

        # Cluster data points
        kmeans = KMeans(branching)
        kmeans.fit(data[rp,:])
        C = kmeans.predict(data)

        clusters = kmeans.cluster_centers_

        # Cluster data points if we are not yet in the bottom
        subclusters = clusters
        if depth > 1:
            newdepth = depth-1
            for c in range(0,numpy.max(C)):
                idx = numpy.argwhere(C==c)
                Nc = len(idx)
                if len(idx) > branching:
                    s = Codebook.hkmeans(numpy.reshape(data[idx,:],(Nc,d)),
                                                    branching=branching,
                                                    depth=newdepth,
                                                    max_data_points=max_data_points)
                    subclusters = numpy.vstack((subclusters,s))
                if len(idx) <= branching:
                    #subclusters[c,:,:] = data[idx,:]
                    subclusters = numpy.vstack((subclusters,data[idx,:]))

        return subclusters


    class ClusterData:
            """ClusterData holds information about datapoints and clusters"""
            clusters = []

            def __init__(self,clusters):
                self.clusters = clusters

    class ClusterNode:
            """Holds information about the hierarchy of the clusters"""
            subclusters = []
            clusters = []

            def __init__(self,clusters,subclusters):
                self.clusters = clusters
                self.subclusters = subclusters

###############################################################################
# Codebook histograms
###############################################################################


class CodebookHistograms:
    """This class provides functions to generate codebook histograms
    and to normalise them etc"""

    def generate(self, codebook, features):
        [N, d] = codebook.shape
        if features.size <= 1:
            return numpy.zeros((0, N))

        [hits, d] = scipy.cluster.vq.vq(features, codebook)
        [y, x] = numpy.histogram(hits, bins=range(0, N + 1))
        return y

    @staticmethod
    def normalise(codebookhistograms,n=2):
        codebookhistograms = codebookhistograms.astype(float)
        [N,d] = codebookhistograms.shape
        for i in range(0,N):
            normf = numpy.linalg.norm(codebookhistograms[i,:],ord=n)
            if normf != 0:
                codebookhistograms[i,:] = codebookhistograms[i,:] / normf
        return codebookhistograms