//
// Created by Alessio Russo Introito on 2019-03-02.
//

#ifndef KMEANSCLUSTERING_POINT_H
#define KMEANSCLUSTERING_POINT_H

#define MAX_DIM 100

using namespace std;

struct Point {
    double values[MAX_DIM];
    int id;
    int size;
};

#endif //KMEANSCLUSTERING_POINT_H