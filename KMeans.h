//
// Created by Claudio Russo Introito on 2019-03-02.
//

#ifndef KMEANSCLUSTERING_KMEANS_H
#define KMEANSCLUSTERING_KMEANS_H



#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include "Point.h"
#include "Cluster.h"

using namespace std;


class KMeans{

private:
    int K; // number of clusters
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters;

    // return ID of nearest center (uses euclidean distance)
    int getIDNearestCenter(Point point);

public:

    KMeans(int K, int total_points, int total_values, int max_iterations);
    void run(vector<Point> & points);
};

#endif //KMEANSCLUSTERING_KMEANS_H