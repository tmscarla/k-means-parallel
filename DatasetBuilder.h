//
// Created by Alessio Russo Introito on 2019-08-26.
//

#ifndef K_MEANS_PARALLEL_DATASETBUILDER_H
#define K_MEANS_PARALLEL_DATASETBUILDER_H

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

class DatasetBuilder {

public:
    DatasetBuilder(int numPoints, int pointDimension, int numClusters, int maxIteration, string filename);
    ~DatasetBuilder();
    void createDataset();

private:
    int numPoints, pointDimension, numClusters, maxIteration;
    string filename;
};


#endif //K_MEANS_PARALLEL_DATASETBUILDER_H
