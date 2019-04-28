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

    vector<Punto> dataset;
    vector<Punto> localDataset;
    int numPointsPerNode;

public:
    Node(int rank, MPI_Comm comm = MPI_COMM_WORLD);
    void readDataset();
    void scatterDataset();
};


#endif //KMEANSCLUSTERING_NODE_H
