# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#
# File reading and writing
#
#------------------------------------------------------------------------------

import os
import sys
import numpy as np
import logging


def read_ascii_matrix_somclu(filename, sep=' '):
    f = open(filename)

    line = f.readline()
    [N,M] = line[1:].split(' ')
    N = int(N)
    M = int(M)
    line = f.readline()
    d = int(line[1:])

    A = np.empty([N,M,d], dtype=float)

    for i in range(0,N):
        for j in range(0,M):

            line = f.readline()
            vals = line.split(sep)

            # Verify that we have enough numbers
            if len(vals)-1 is not d:
                print('d=%d ? len(vals)=%d' % (d, len(vals)))

            for k in range(0, d):
                A[i,j,k] = float(vals[k])

    return A

def read_test_results_lf_test(resultfile):
    "[m, nclasses, dbsize, perf, methods, cbsizes] = read_test_results_lf_test(resultfile)"
    data = np.loadtxt(resultfile)
    m = np.array(data[:,0], dtype=int)
    nclasses = np.array(data[:,1], dtype=int)
    cbsize = np.array(data[:,2], dtype=int)
    perf = np.array(data[:,3], dtype=float)

    methods = np.unique(m)
    cbsizes = np.unique(cbsize)

    return [m, nclasses, cbsize, perf, methods, cbsizes]

def read_configs_json(filepath):
    "Reads a config file formatted in JSON format"
    import json
    if not os.path.exists(filepath):
        print("Config file %s does not exist!" % filepath)
        sys.exit(-1)

    f = open(filepath, 'r')
    configs = json.load(f)
    f.close()
    return configs

class TestLog:
    filename = ''

    def __init__(self, filename_):
        self.filename = filename_

        filedir = os.path.dirname(self.filename)
        if not os.path.exist(filedir):
            os.makedirs(filedir)

        logging.basicConfig(filename=filename_,
                            level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            filemode='w')

    def close(self, ):
        self.filep.close()

    def write(self, message):
        self.filep.write(message + '\n')

def read_imagelist_file(imagelistfile):
    f = open(imagelistfile, 'r')
    imagelist = f.readlines()

    # Clean up the image file paths
    for i in range(0, len(imagelist)):
        # Clean up the './' from the beginning...
        [tmp, imgpath] = imagelist[i].split('./')
        # Clean up the '\n' from the end
        [imgpath, tmp] = imgpath.split('\n')
        # Update
        imagelist[i] = imgpath

    return imagelist

def read_news_value_ground_truth_csv(filename):
    f = open(filename, 'r')

    rows = f.readlines()
    N = len(rows)
    M = len(rows[0].split(';'))


    D = np.zeros((N,M))

    for i in range(0,N):
        fields = rows[i].split(';')
        for j in range(0, M):
            if len(fields[j])>0:
                D[i,j] = int(fields[j])
            else:
                D[i,j] = -1
    return D

def save_matrix_bin(filename, data):
    "Saves given matrix (data) into a binary file"

    rows = np.uint32(data.shape[0])
    cols = np.uint32(data.shape[1])

    if os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # Open the file for writing in binary mode
    f = open(filename, 'wb')

    # Write file
    rows.tofile(f)
    cols.tofile(f)
    data.tofile(f)

    # Close the file
    f.close()

def read_matrix_bin(filename):
    "Reads a matrix from a file and returns it"

    f = open(filename, 'rb')

    rows = np.fromfile(f, dtype=np.uint32, count=1)
    cols = np.fromfile(f, dtype=np.uint32, count=1)
    data = np.fromfile(f, dtype=np.float32)

    data = data.reshape((rows,cols))

    return data