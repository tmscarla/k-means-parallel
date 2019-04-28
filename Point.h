//
// Created by Alessio Russo Introito on 2019-03-02.
//

#ifndef KMEANSCLUSTERING_POINT_H
#define KMEANSCLUSTERING_POINT_H

#define MAX_DIM 25

using namespace std;

struct Punto {
    double values[MAX_DIM];
    int id;
};

#endif //KMEANSCLUSTERING_POINT_H