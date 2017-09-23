# Clustering

## Run
- `python agglomerative_clustering.py`  
- `python kernel_kmeans.py`  
- `python spectral_clustering.py`  

Note that ideally, one performs clustering on real world datasets. This was the objective here too. And hence, a dataset was downloaded - the <a href="http://archive.ics.uci.edu/ml/datasets/image+segmentation"> `UCI ML Image Segmentation Data Set` <a href=""></a>. It has a few features and hence can be plotted and visualized. But the problem was that there were no clear cut clusters based on the class labels. Another problem was that there was a large number of class labels. Hence for understanding clustering algorithms, I decided to create `blobs` using scikit-learn and then cluster them - each blob as a cluster - ideally.

### **Agglomerative Clustering** 

  - This implements two cluster merging criterion:  
  	* `min distance` criterion - Merge the two clusters which have the minimum distance between two points - one from each cluster.

  	* `avg distance` criterion - Merge two clusters based on the average distance of all the points in each cluster.

  - Reference used <http://www.janeriksolem.net/2009/04/hierarchical-clustering-in-python.html>

### **Kernel KMeans Clustering**

  - Two kernels have been used here: `polynomial kernel` and `RBF kernel`. You can make the change in line 54 and 55, depending on what you want to use 

  - Reference used <http://www.cs.ucsb.edu/~veronika/MAE/Global_Kernel_K-Means.pdf>

### **Spectral Clustering** 

  - Reference used <http://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf>  
