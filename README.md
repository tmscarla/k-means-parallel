# K-means parallel
A parallel implementation of the unsupervised clustering algorithm K-means with **OpenMP** and **MPI**. The parallelization leverages on a shared memory multiprocessing programming (OpenMP) and a message-passing protocol that allows the communication among nodes (MPI).

<p align="center">
<img width="350" src="https://github.com/tmscarla/k-means-parallel/blob/master/img/kmeans.gif"/>
</p>

## Requirements
The project requires:
- MPICH
- OpenMP

## Goal
Our goal is to maximize the performance, thus minimize the execution time of the algorithm. In this case our hyperparameters are:
- number of processors (MPI) 
- number of threads considered in the parallel region (OpenMP)
Best performance strongly depends on computer specifications. 

## Dataset 
The first line of the dataset must contain the values of the initial parameters, in the following order: 
<p align="center">
	<i>n. point dimension, n. clusters, max iterations</i>
</p>
Then, each line refers to point dimensions values.
You can create your own dataset where the dimension values of each point are randomly assigned. 

## Compile and run
To compile it you need to run the following command: 
~~~~
mpic++ -fopenmp -o main main.cpp Node.cpp DatasetBuilder.cpp
~~~~
Where the -fopenmp flag that includes the GCC's OpenMP implementation. Then, if you want to run the program:
~~~~
mpirun -np (number of processors) ./main
~~~~
If you have configured a cluster of machine in order to run it among several nodes, such as master node, client1 and client2, the master node must run:
~~~~
mpirun -np (no. processors) -host master,client1,client2 ./main
~~~~
Or you can directly call a specific hostname configuration file:
~~~~
mpirun -np (no.processors) -hostfile hostfile ./main
~~~~

## Workflow
### Initial Configuration
Suppose we have P processors, where the master node is defined by the Rank 0, and a dataset with N points represented in the space with a M-dimensional vectors. Then the master node: 
1. loads the dataset and scatters it among nodes, assigning to each of them N/P points. R Remaining points are assigned to the first R nodes. 
2. reads initial configuration parameters: no. clusters, no. dimensions, max iterations.
3. chooses K points as initial centroids and broadcast them to the other nodes.

### K-means 'loop'
4. In each node the following steps are executed: 
	- for each point in the local dataset find the membership among the K clusters. Each point must be in one and only one cluster. Distance calculation between point and centroids is performed in parallel using OpenMP
	- within each cluster, sum 'dimension-to-dimension' all the points that belong to that cluster. We obtain a local for each cluster, in each node. 

<p align="center">
<img width="80%" src="https://github.com/tmscarla/k-means-parallel/blob/master/img/MPI_Allreduce.png"/>
</p>

5. Once we get the local summations, with MPI_Allreduce operation, we can store the sum of the local summations (global sum) in each node. In the same way we can obtain the global number of points in each cluster and store that value in each nodes. So to recalculate a centroid, we can simply divide the global sum of that cluster over the number of points belong to it. Compute new centroids
6. Go to point 4 and repeat until termination

## Distance Metrics
Two distance metrics are implemented: 
1. Euclidean Distance
2. Cosine Similarity

## Termination
K-means converges when no more point changes its membership status. Since our dataset is distributed among nodes, we cannot know directly if no more changes occurs. 
To deal with this, we have defined a flag in each node that is set when no more changes in the local dataset happen. Those flags are collected by nodes to check if there are changes or not. Furthermore, in order to avoid unnecessary long computation, we have limited the number of iteration that can be executed. 


<p align="center">
<img width="90%" src="https://github.com/tmscarla/k-means-parallel/blob/master/img/summary.png"/>
</p>


