#Written for CS545 Assignment 4 by Austen Hsiao, 985647212

import numpy as np
import pandas as pd
import os
import random
import math

random.seed()

# Opens data file and returns a numpy array of the data (assumes that original data is comma delimited)
def open_file(file):
    file = pd.read_csv(file, sep=',', header=None, engine='python').to_numpy()
    return file

# Generates a numpy array of 'k' example data to serve as the starting centers
def init_clusters(k, trainingSet):
    centers = []  
    lines = len(trainingSet)-1
    for i in range(k):
        clusterCenter = random.randint(0, lines)
        while list(trainingSet[clusterCenter]) in centers:
            clusterCenter = random.randint(0, lines)
        centers.append(list(trainingSet[clusterCenter]))
    return np.array(centers)

# Returns the euclidian distance between 2 points. Takes in 2 numpy arrays. 
# Note that the points include the class denotation on the end, so the summation is up until the last element
def distance(point1, point2):
    return math.sqrt(np.sum(np.square(np.subtract(point1[:-1], point2[:-1]))))

# Performs 'i' number of iterations
def clusterIterate(i, trainingSet, clusterCenter):
    print("TRAININGSET", trainingSet[0])
    print("CENTER", clusterCenter[0])
    print(distance(trainingSet[0], clusterCenter[0]))


if __name__ == '__main__':
    # Change this to point to the right working directory. # # # # # # # # # # # #
    os.chdir('C:\\Users\\Austen\\Desktop\\Sourcefiles\\Python\\Assignment4')
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    trainingData = open_file('optdigits\\optdigits.train')
    centers = init_clusters(10, trainingData)

    clusterIterate(0, trainingData, centers)
    