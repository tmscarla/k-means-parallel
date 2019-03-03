//
// Created by Claudio Russo Introito on 2019-03-02.
//

#ifndef KMEANSCLUSTERING_CLUSTER_H
#define KMEANSCLUSTERING_CLUSTER_H



#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
//#include <time.h>
#include <algorithm>
#include "Point.h"

using namespace std;

class Cluster
{
private:
    int id_cluster;
    int id_centroid;
    vector<double> central_values;
    vector<Point> points;

public:
    Cluster(int id_cluster, Point point){   //L'intero cluster è identificato da id_cluster e il centroids è point

        this->id_cluster = id_cluster;


        int total_values = point.getTotalValues();

        for(int i = 0; i < total_values; i++)
            central_values.push_back(point.getValue(i));

        points.push_back(point);    //add centroid to cluster
    }

    void addPoint(Point point){ //add points to the cluster
        points.push_back(point);
    }

    int getIdCentroid(){
        return id_centroid;
    }

    bool removePoint(int id_point){  //remove point from the cluster

        int total_points = points.size();

        for(int i = 0; i < total_points; i++)
        {
            if(points[i].getID() == id_point)
            {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    double getCentralValue(int index){
        return central_values[index];
    }

    void setCentralValue(int index, double value){
        central_values[index] = value;
    }

    Point getPoint(int index){
        return points[index];
    }

    int getTotalPoints(){
        return points.size();
    }

    int getID(){
        return id_cluster;
    }
};

#endif //KMEANSCLUSTERING_CLUSTER_H