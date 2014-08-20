import numpy as np

def som(data, nSomX = 50, nSomY = 50, nEpoch = 10, radius0 = 0, radiusN = 0,
        radiusCooling = "linear", scale0 = 0, scaleN = 0.01, scaleCooling = "linear",
        kernelType = 0, mapType = "planar", initialCodebookFilename = ''):
    """This is a wrapper for somoclu wrapper.
    USAGE:
        [codebook, bmus, U] = clustering.som(localfeatures,nSomX=1000,nSomY=1)
    """


    # Import somoclu here since it might not work on all devices
    # (it needs to be downloaded and compiled seperately)
    import somoclu

    nVectors = data.shape[0]
    nDimensions = data.shape[1]
    data1D = np.ndarray.flatten(data)

    snapshots = 0

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
    #print codebook
    #print globalBmus
    #print uMatrix
    return [codebook, globalBmus, uMatrix]
