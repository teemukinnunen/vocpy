# vocpy - Simple Visual Object Categorisation tool for PYthon

## Dependencies
* numpy - http://www.numpy.org/
* scipy - http://www.scipy.org/
* opencv (cv2 module for python is needed) - http://opencv.org/
* sklearn - http://scikit-learn.org/stable/


## Tools
Tools-directory consists of usefull tools for extracting features and mathcing images.

### Local feature extraction
usage: extractfeatures.py [-h] -i IMAGEDIR -dd DATADIR [-il IMAGELIST]
                          [-ip DETECTOR] [-lf DESCRIPTOR]
            
Extracts local features from a given IMAGEDIR using DETECTOR to detect features and DESCRIPTOR to describe the detected local features.

### Visual codebook
usage: generatecodebook.py [-h] -i IMAGEDIR -dd DATADIR [-il IMAGELIST]
                           [-ip DETECTOR] [-lf DESCRIPTOR] [-k CODEBOOKSIZE]
                           [-cm CODEBOOKMETHOD]
                           
Generates a visual codebook from local features which are extracted from images in IMAGEDIR using DETECTOR and DESCRIPTOR. Codebook is generates using CODEBOOKMETHOD and the size of the codebook can be given as CODEBOOKSIZE.

### Visual codebook histograms
usage: generatecodehistograms.py [-h] -i IMAGEDIR -dd DATADIR [-il IMAGELIST]
                                 [-ip DETECTOR] [-lf DESCRIPTOR]
                                 [-k CODEBOOKSIZE] [-cm CODEBOOKMETHOD]

Generatates codebook hisgrams comparing local features extracted from images with previously generated codebook.

### Image matching
usage: matchimages.py [-h] -i1 IMAGE1 -i2 IMAGE2 [-dd DATADIR] [-il IMAGELIST]
                      [-ip DETECTOR] [-lf DESCRIPTOR] [--debug DEBUG]

Matches two images (IMAGE1 and IMAGE2) using spatially matching local features.

