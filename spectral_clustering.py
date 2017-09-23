#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.datasets.samples_generator import make_blobs

cutoff = 0.985
k = 3

def find_distance_matrix(data):
    euclid_distance = []
    for i in data:
        distance = []
        for j in data:
            distance.append(np.linalg.norm(i - j) * np.linalg.norm(i - j))
        distance = np.array(distance)
        euclid_distance.append(distance)
    euclid_distance = np.array(euclid_distance)
    return euclid_distance

def inverse_squareform(matrix):
    inv_sqfrm = []
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix[i])):
            inv_sqfrm.append(matrix[i][j])
    inv_sqfrm = np.array(inv_sqfrm)
    return inv_sqfrm

def rbfkernel(gamma, distance):
    return np.exp(-gamma * distance)

def KMeans_implementation(data):
    alpha = np.zeros(shape = (len(data), k))

    random.seed()
    indexes = random.sample(range(0, len(data)), k)
    means = data[indexes]

    for i in range(15):
        new_alpha = np.zeros(shape = (len(data),k))
        for i in range(len(data)):
            min_value = 123213123123123
            min_index = -1
            for j in range(k):
                distance = np.linalg.norm(data[i] - means[j])**2
                if distance < min_value:
                    min_value = distance
                    min_index = j
            new_alpha[i][min_index] = 1
        alpha = new_alpha[:]
        class_frequency = pd.DataFrame(alpha)    
        cluster_count = [class_frequency[i].value_counts()[1] for i in range(k)]
        for j in range(k):
            means[j] = 0
            for i in range(len(data)):
                means[j] += (alpha[i][j] * data[i])
            means[j] = means[j] / float(cluster_count[j])

    labels = []
    for i in range(len(alpha)):
        for j in range(k):
            if alpha[i][j] == 1:
                labels.append(j)
    return labels

def main():
    # Generate Clusters
    # df=pd.read_csv('./segmentation.data.modified')
    centers = [[1, 1], [-1, -1], [1, -1]]
    # X, _ = make_blobs(n_samples = 90, centers = centers, cluster_std = 0.5)
    # df = pd.DataFrame(X)
    X = datasets.make_moons(n_samples=100, noise=.05)
    df = pd.DataFrame(X[0]) ## convert to DF
    data = np.array(df)

    # Visualize the data
    f = plt.figure(1)
    plt.scatter(df[0],df[1])
    f.show()

    # Making kernel matrix
    distance = find_distance_matrix(data)
    gamma = 1/(2*np.var(inverse_squareform(distance)))
    kernel = rbfkernel(gamma, distance)

    # Calculating weight matrix, and using cutoff to indicate far away points
    W = kernel[:]
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            if kernel[i][j] < cutoff:
                W[i][j] = 0

    # Making degree matrix
    D = np.zeros(shape = (len(W), len(W)))
    for i in range(len(W)):
        D[i][i] = np.sum(W[i])           

    # Calculate Laplacian Matrix
    L = D - W

    # Find eigen vectors of Laplacian Matrix
    eigen_values, eigen_vectors = np.linalg.eig(L)
    indexes = eigen_values.argsort()[::1]
    second_vector = eigen_vectors[:, indexes[1:k]]

    # Use K-means to find labels
    label = KMeans_implementation(second_vector)
    print label

    g = plt.figure(2)
    plt.scatter([data[i][0] for i in range(len(data)) if label[i] == 1], [data[i][1] for i in range(len(data)) if label[i] == 1], color='red', alpha=0.5)
    plt.scatter([data[i][0] for i in range(len(data)) if label[i] == 0], [data[i][1] for i in range(len(data)) if label[i] == 0], color='blue', alpha=0.5)
    plt.scatter([data[i][0] for i in range(len(data)) if label[i] == 2], [data[i][1] for i in range(len(data)) if label[i] == 2], color='green', alpha=0.5)
    g.show()

    raw_input()

if __name__ == '__main__':
    main()