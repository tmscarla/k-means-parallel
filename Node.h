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


    int* memCounter;        //Membership count

    vector<Punto> dataset;
    vector<Punto> localDataset;
    vector<Punto> clusters;
    vector<Punto> localSum;
    int numPointsPerNode;
    vector<int> memberships; //This vector has same length as localDataset: for each point in localDataset is
                            // associated the id of nearest cluster in the corresponding position in membership
    //TODO gestire anche la globalMembership
    int* globalMembership;

public:
    Node(int rank, MPI_Comm comm = MPI_COMM_WORLD);
    void readDataset();
    void scatterDataset();
    void extractCluster();
    int getIdNearestCluster(Punto p);
    //void distributedPointSum(vector<Punto> *in, vector<Punto> *inout, int* len, MPI_Datatype* dptr);
    bool run(int it);
    void updateLocalSum();
    double* serializePointValues(vector<Punto> v);
    void deserializePointValues(double* values);

    void computeGlobalMembership();
    int getNumPoints();
    int* getGlobalMemberships();
};


#endif //KMEANSCLUSTERING_NODE_H
