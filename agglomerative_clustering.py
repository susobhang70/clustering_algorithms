#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

k = 3

class cluster_node:
	def __init__(self, vec, id, left=None, right=None, distance=0.0, node_vector = None):
		self.leftnode = left
		self.rightnode = right
		self.vec = vec
		self.id = id
		self.distance = distance
		if node_vector is None:
			self.node_vector = [self.id]
		else:
			self.node_vector = node_vector[:]

def euclidean_distance(vec1, vec2):
	return np.sqrt(sum((vec1 - vec2) ** 2))

def min_distance(clust1, clust2, distances):
	d = 12123123123123
	for i in clust1.node_vector:
		for j in clust2.node_vector:
			try:
				distance = distances[(i,j)]
			except:
				try:
					distance = distances[(j,i)]
				except:
					distance = euclidean_distance(clust1.vec, clust2.vec)
			if distance < d:
				d = distance
	return d

def agglomerative_clustering(data, distance):
	# cluster the rows of the data matrix
	distances = {}
	currentclustid = -1

	# cluster nodes are initially just the individual rows
	nodes = [cluster_node(np.array(data[i]), id=i) for i in range(len(data))]

	while len(nodes) > k:
		lowestpair = (0,1)
		closest = euclidean_distance(nodes[0].vec,nodes[1].vec)
	
		# loop through every pair looking for the smallest distance
		for i in range(len(nodes)):
			for j in range(i+1,len(nodes)):
				# distances is the cache of distance calculations
				if (nodes[i].id,nodes[j].id) not in distances:
					if distance == "min":
						distances[(nodes[i].id,nodes[j].id)] = min_distance(nodes[i], nodes[j], distances)
					else:
						distances[(nodes[i].id,nodes[j].id)] = euclidean_distance(nodes[i].vec,nodes[j].vec)
		
				d = distances[(nodes[i].id,nodes[j].id)]
		
				if d < closest:
					closest = d
					lowestpair = (i,j)
		
		# calculate the average of the two nodes
		len0 = len(nodes[lowestpair[0]].node_vector)
		len1 = len(nodes[lowestpair[1]].node_vector)
		mean_vector = [(len0*nodes[lowestpair[0]].vec[i] + len1*nodes[lowestpair[1]].vec[i])/(len0 + len1) \
						for i in range(len(nodes[0].vec))]
		
		# create the new cluster node
		new_node = cluster_node(np.array(mean_vector), currentclustid, left = nodes[lowestpair[0]], right = nodes[lowestpair[1]], \
			distance = closest, node_vector = nodes[lowestpair[0]].node_vector + nodes[lowestpair[1]].node_vector)
		
		# cluster ids that weren't in the original set are negative
		currentclustid -= 1
		del nodes[lowestpair[1]]
		del nodes[lowestpair[0]]
		nodes.append(new_node)

	return nodes

def main():
	# Generate data
	# df = pd.read_csv('./segmentation.data.modified')
	centers = [[1, 1], [-1, -1], [1, -1]]
	X, _ = make_blobs(n_samples = 90, centers = centers, cluster_std = 0.5)
	df = pd.DataFrame(X) ## convert to DF

	# Visualize the data
	f = plt.figure(1)
	plt.scatter(df[0],df[1])
	f.show()
	colorset = ['red', 'green', 'blue', 'yellow', 'brown', 'orange', 'black']

	data = np.array(df)

	# Average criterion agglomerative clustering
	cluster = agglomerative_clustering(data, "avg")
	# plt.scatter(cluster.leftnode.vec[0], cluster.leftnode.vec[1], color = 'yellow')
	# plt.scatter(cluster.rightnode.leftnode.vec[0], cluster.rightnode.leftnode.vec[1], color = 'red')
	# plt.scatter(cluster.rightnode.rightnode.vec[0], cluster.rightnode.rightnode.vec[1], color = 'green')
	j = 0
	m = plt.figure(2)
	for i in cluster:
		plt.scatter(data[i.node_vector].T[0], data[i.node_vector].T[1], color = colorset[j])
		j += 1
	m.show()

	# Min criterion agglomerative clustering
	g = plt.figure(3)
	cluster = agglomerative_clustering(data, "min")
	# plt.scatter(cluster.leftnode.vec[0], cluster.leftnode.vec[1], color = 'yellow')
	# plt.scatter(cluster.rightnode.leftnode.vec[0], cluster.rightnode.leftnode.vec[1], color = 'red')
	# plt.scatter(cluster.rightnode.rightnode.vec[0], cluster.rightnode.rightnode.vec[1], color = 'green')
	j = 0
	for i in cluster:
		plt.scatter(data[i.node_vector].T[0], data[i.node_vector].T[1], color = colorset[j])
		j += 1
	g.show()
	raw_input()

if __name__ == '__main__':
	main()