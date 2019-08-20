//
// Created by Claudio Russo Introito on 2019-04-28.
//

#ifndef KMEANSCLUSTERING_NODE_H
#define KMEANSCLUSTERING_NODE_H

#include <string>
#include <mpi.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "Point.h"

using namespace std;


class Node{

private:
    int rank;
    MPI_Comm comm;
    MPI_Datatype pointType;

    int total_values;
    int num_local_points;
    int K, max_iterations;
    int numPoints;  //Total number of points in the whole dataset

    int notChanged;   //Serve a stabilire se durante un run ci sono stati cambiamenti di appartenenza ad un cluster.
                        // Se non ci sono stati e it < max iterations, allora l'algoritmo ha raggiunto la configurazione ottima

    int* memCounter;        //Membership count

    vector<Point> dataset;
    vector<Point> localDataset;
    vector<Point> clusters;
    vector<Point> localSum;
    int numPointsPerNode;
    vector<int> memberships; //This vector has same length as localDataset: for each point in localDataset is
                            // associated the id of nearest cluster in the corresponding position in membership
    int* globalMembership;
    double total_time;

public:
    Node(int rank, MPI_Comm comm = MPI_COMM_WORLD);
    int getMaxIterations();
    void readDataset();
    void scatterDataset();
    void extractCluster();
    int getIdNearestCluster(Point p); //private
    int run(int it);
    void updateLocalSum();  //private
    void computeGlobalMembership();
    int getNumPoints();
    int* getGlobalMemberships();
    void printClusters();
    void writeClusterMembership(string filename);

    void getStatistics();
    void printStatistics();

};


#endif //KMEANSCLUSTERING_NODE_H
