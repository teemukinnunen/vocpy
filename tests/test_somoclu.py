#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import somoclu
import numpy as np

#data = np.loadtxt('rgbs.txt')
data = np.random.rand(1000,3)
print(data)
data = np.float32(data)
nSomX = 50
nSomY = 50
nVectors = data.shape[0]
nDimensions = data.shape[1]
data1D = np.ndarray.flatten(data)
nEpoch = 10
radius0 = 0
radiusN = 0
radiusCooling = "linear"
scale0 = 0
scaleN = 0.01
scaleCooling = "linear"
kernelType = 0
mapType = "planar"
snapshots = 0
initialCodebookFilename = ''
codebook_size = nSomY * nSomX * nDimensions
codebook = np.zeros(codebook_size, dtype=np.float32)
globalBmus_size = int(nVectors * int(np.ceil(nVectors/nVectors))*2)
globalBmus = np.zeros(globalBmus_size, dtype=np.intc)
uMatrix_size = nSomX * nSomY
uMatrix = np.zeros(uMatrix_size, dtype=np.float32)
somoclu.trainWrapper(data1D, nEpoch, nSomX, nSomY,
                     nDimensions, nVectors,
                     radius0, radiusN,
                     radiusCooling, scale0, scaleN,
                     scaleCooling, snapshots,
                     kernelType, mapType,
                     initialCodebookFilename,
                     codebook, globalBmus, uMatrix)
print codebook
print globalBmus
print uMatrix
