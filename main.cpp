#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include "Point.h"
#include "Cluster.h"
#include "KMeans.h"

int main() {
    srand (time(NULL));

    int total_points, total_values, K, max_iterations, has_name;

    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    vector<Point> points;
    string point_name;

    for(int i = 0; i < total_points; i++)
    {
        vector<double> values;

        for(int j = 0; j < total_values; j++)
        {
            double value;
            cin >> value;
            values.push_back(value);
        }

        if(has_name)
        {
            cin >> point_name;
            Point p(i, values, point_name);
            points.push_back(p);
        }
        else
        {
            Point p(i, values);
            points.push_back(p);
        }
    }

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);

    return 0;

}