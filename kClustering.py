#Written for CS545 Assignment 4 by Austen Hsiao, 985647212

import numpy as np
import pandas as pd
import os
import random
import math
from PIL import Image

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

# distance returns the index of the closest center, given 1 data point (point1) and the list of centerpoints
def minDistanceIndex(point1, listOfCenters):
    # distanceList contains all of the distances between point1 and every center
    distanceList = [math.sqrt(np.sum(np.square(np.subtract(point1[:-1], center[:-1])))) for center in listOfCenters]

    # because we iterated through listOfCenters, we can get the closest center point by
    # finding the index of the minimum distance in the distance list
    minimum = min(distanceList)
    minIndex = distanceList.index(min(distanceList))

    # If we have multiple minimums (equal values), randomly choose one 
    if distanceList.count(minimum) > 1:
        minIndex = random.choice([i for i in range(len(distanceList)) if distanceList[i] == minimum]) 
    return minIndex

# returns distance between 2 points
def euclidDistance(point1, point2):
    return math.sqrt(np.sum(np.square(np.subtract(point1[:-1], point2[:-1]))))

# returns distance^2 between 2 points
def euclidDistanceSq(point1, point2):
    return np.sum(np.square(np.subtract(point1[:-1], point2[:-1])))

# returns the averageMSE given the clusters and datapoints associated with each cluster
def averageMSE(clusterCenters, clusterMembers):
    # clusterCenters contains all of the center points (for expt1, there are 10)
    # clusterMembers contains all of the data points associated with each clusterCenter. clusterMembers[0] will be a list of all points associated with clusterCenters[0]

    msePerCluster = []
    for cluster,center in zip(clusterMembers,clusterCenters):
        currentCenterMSE = 0
        for dataPoint in cluster:
            currentCenterMSE += euclidDistanceSq(dataPoint, center)
        msePerCluster.append(currentCenterMSE/len(cluster))
    return sum(msePerCluster)/len(msePerCluster)

def meanSquareSeparation(clusterCenters):
    total = 0
    count = (len(clusterCenters) * (len(clusterCenters-1))/2.0)

    for a in clusterCenters:
        for b in clusterCenters:
            if (a == b).all():
                continue
            total += euclidDistanceSq(a, b)
    total = total / 2.0 # Since we have duplicates eg. every center is paired twice (AB vs BA)
    return total/count

def entropyForCluster(singleClusterMembers):
    total = len(singleClusterMembers)
    counts = {}
    for point in singleClusterMembers:
        if point[-1] in counts.keys():
            counts[point[-1]] += 1
        else:
            counts[point[-1]] = 1
    runningTotal = 0
    for i in counts.keys():
        x = (counts.get(i)/total)
        runningTotal += x * math.log2(x)
    return -1*runningTotal

def meanEntropy(clusterMembers):
    meanEntropy = 0
    m = sum([len(i) for i in clusterMembers])
    for cluster in clusterMembers:
        mi = len(cluster)
        entropy = entropyForCluster(cluster)
        meanEntropy += (mi/m)*entropy
    return meanEntropy

def accuracy(clusterMembers):
    hits = 0
    total = 0
    for singleCluster in clusterMembers:
        counts = {}
        total += len(singleCluster)
        for point in singleCluster:
            if point[-1] in counts.keys():
                counts[point[-1]] += 1
            else:
                counts[point[-1]] = 1
        classLabel = max(counts, key=counts.get)
        hits += counts[classLabel]
    return hits/total

def confusionMatrix(clusterMembers):
    classLabelsByCluster = [] #this array will contain the class labels for the corresponding cluster. eg. classLabelsByCluster[0] = 2 means that the class for cluster 0 is 2
    for singleCluster in clusterMembers:
        counts = {}
        for point in singleCluster:
            if point[-1] in counts.keys():
                counts[point[-1]] += 1
            else:
                counts[point[-1]] = 1
        classLabelsByCluster.append(max(counts, key=counts.get))
    print("Cluster class labels:", classLabelsByCluster)
    confusionMatrix = np.zeros([10,10], dtype=int) # This is hardcoded because there are 10 digits

    clusterNumber = 0
    for singleCluster in clusterMembers:
        for point in singleCluster:
            column = classLabelsByCluster[clusterNumber]
            row = point[-1]
            confusionMatrix[row][column] += 1
        clusterNumber += 1

    print("Confusion Matrix:")
    for i in confusionMatrix:
        print(i)

def generateGreyscaleImages(clusterCenters):
    scale = 255/16.0
    currIndex = 0
    for cluster in clusterCenters:
        center = cluster[:-1]
        center.resize((8,8))
        (Image.fromarray(np.uint8(center*scale), 'L')).save(str(currIndex) + ".png")
        currIndex += 1


# Performs iterations with the trainingSet and an initial np array of clusterCenters until the previous centers and the new centers are the same
def clusterIterate(trainingSet, clusterCenters):
    clusterList = [[]] * len(clusterCenters)

    for data in trainingSet:
        minIndex = minDistanceIndex(data, clusterCenters)
        if clusterList[minIndex] == []:
            clusterList[minIndex] = [data]
        else:
            clusterList[minIndex].append(np.array(data))

    # Calculate new cluster centers and update
    updatedClusterCenters = np.array([np.mean(i, axis=0) for i in clusterList])

    # If the updatedClusters are equal to the original clusters, we're finished looping
    compare = (updatedClusterCenters == clusterCenters)
    if compare.all():
        print("AverageMSE:", averageMSE(clusterCenters, clusterList))
        print("MSS:", meanSquareSeparation(clusterCenters))
        print("MeanEntropy:", meanEntropy(clusterList))
        print("Accuracy:", accuracy(clusterList))
        confusionMatrix(clusterList)
        generateGreyscaleImages(clusterCenters)
        return
    else:
        # Otherwise, keep iterating
        return clusterIterate(trainingSet, updatedClusterCenters)

if __name__ == '__main__':
    # Change this to point to the right working directory. # # # # # # # # # # # #
    os.chdir('C:\\Users\\Austen\\Desktop\\Sourcefiles\\Python\\Assignment4')
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # This also assumes that we're running the py script up one level from the optdigits files
    trainingData = open_file('optdigits\\optdigits.train')
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    centers = init_clusters(10, trainingData)
    clusterIterate(trainingData, centers)