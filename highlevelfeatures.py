# -*- coding: utf-8 -*-
import glob
import cv2

#cascadefiles = glob.glob('/usr/share/opencv/haarcascades/*.xml')
cascadefiles = ['haarcascade_frontalface_alt.xml',
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_frontalface_alt_tree.xml',
                'haarcascade_frontalface_default.xml',
                'haarcascade_fullbody.xml',
                'haarcascade_profileface.xml',
                'haarcascade_upperbody.xml']

def detect_person_parts(I):

    detectedObjects = []

    for cascadefile in cascadefiles:
        try:
            cascadedetector = cv2.CascadeClassifier(cascadefile)
            if I.ndim == 3:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

            objects = cascadedetector.detectMultiScale(I, 1.3, 5)
            detectedObjects.append(objects)

        except:
            print(("Couldnt use cascade detector %s" % cascadefile))

    return detectedObjects