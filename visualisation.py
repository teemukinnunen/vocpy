# -*- coding: utf-8 -*-

import numpy as np
import pylab
import cv2

def plot_results_lf_test(m, cbsize, perf):
    ""
    cbsizes = np.unique(cbsize)

    methods = np.unique(m)

    mu= np.zeros((len(methods), len(cbsizes)))
    sd = np.zeros((len(methods), len(cbsizes)))


    for mi in range(0, len(methods)):
        for ci in range(0, len(cbsizes)):
            idx1 = np.argwhere(m==methods[mi])
            idx1 = idx1[:,0]
            idx2 = np.argwhere(cbsize[idx1]==cbsizes[ci])
            mu[mi,ci] = np.mean(perf[idx1[idx2]])
            sd[mi, ci] = np.std(perf[idx1[idx2]])

    for ci in range(0, len(cbsizes)):
        pylab.hold()
        pylab.errorbar(range(0,len(methods)), mu[:,ci], sd[:,ci])

def plot_faces(faces, img): #, eye_cascade):

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
