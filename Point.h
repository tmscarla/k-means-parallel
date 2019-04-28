//
// Created by Alessio Russo Introito on 2019-03-02.
//

#ifndef KMEANSCLUSTERING_POINT_H
#define KMEANSCLUSTERING_POINT_H

#define MAX_DIM 25

using namespace std;

struct Punto {
    double values[MAX_DIM];
    int size;

    /*TODO Teoricamente size non serve a niente; potremmo trasformarlo in un identificatore del punto, in modo
     * da associare un valore del punto ad un cluster*/
};

#endif //KMEANSCLUSTERING_POINT_H