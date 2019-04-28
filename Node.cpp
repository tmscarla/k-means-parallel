//
// Created by Claudio Russo Introito on 2019-04-28.
//

#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stddef.h>
#include <fstream>
#include <sstream>
#include "Node.h"

Node::Node(int rank, MPI_Comm comm) : rank(rank) , comm(comm) {

    //Create vector<Point> Datatype in order to be able to send and receive element of struct Punto
    int blocksize[] = {MAX_DIM, 1};
    MPI_Aint displ[] = {0, offsetof(Punto, size)};
    MPI_Datatype blockType[] = {MPI_DOUBLE, MPI_INT};

    MPI_Type_create_struct(2, blocksize, displ, blockType, &pointType);
    MPI_Type_commit(&pointType);

}

void Node::readDataset() {
    if (rank == 0) {

        // READ DATASET
        ifstream infile("twitter_points_20 copiaridotta.csv");
        string line;
        cout << "Leggo il file" << endl;

        /*Potremmo mettere un id di un punto in modo che i cluster mantengano gli id dei punti. Che poi in realtà
         * è come se ce l'avessimo di già, è la variabile num*/

        int count = 0;
        int num = 0;
        while(getline(infile, line, '\n')){
            if (count == 0) {
                cout << "la prima riga è " << line << endl;
                stringstream ss(line);
                getline(ss, line, ';');
                total_values = stoi(line);
                cout << "Total values is: " << total_values << endl;
                //Aggiungere qui gli ulteriori campi della prima riga (num points, num clusters, iterations) con
                // lo stesso format di total_values: quindi scrivere getline(ss, line, ';');     total_values = stoi(line);
                getline(ss, line, '\n');
                max_iterations = stoi(line);
                cout << "Max iteration is: " << max_iterations << endl;
                count++;
                //dataset.resize(total_values);
            } else {
                Punto point;
                point.size = total_values;
                //getline(infile, line);
                int i = 0;
                stringstream ss(line);
                while(getline(ss, line, ';')){
                    //cout << "Put value " << line << " in the array num " << num  << endl;
                    point.values[i] = stod(line);
                    //cout << "Il valore aggiunto è: " << point.values[i] << "\n" << endl;
                    i++;
                }
                num++;
                dataset.push_back(point);

            }
        }

        infile.close();

        //cout << "First element has values : " << dataset[2].values[0] << ". Last one is " << dataset[2].values[19] << endl;

    }
}


void Node::scatterDataset() {
    int numNodes;
    MPI_Comm_size(comm, &numNodes);

    int pointsPerNode[numNodes];
    int datasetDisp[numNodes];

    if(rank == 0) {
        int numPoints = dataset.size(); //forse è di tipo unsigned long
        cout << "Total points: " << numPoints << endl;

        int partial = numPoints/numNodes;
        fill_n(pointsPerNode, numNodes, partial);

        /* Assing remainder R of the division to first R node*/
        if((numPoints % numNodes) != 0){
            int r = numPoints % numNodes;

            for(int i = 0; i < r; i ++){
                pointsPerNode[i] += 1;
            }
        }

        //Vector contains strides (https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node72.html) so, we need to
        // know precisely where starting to divide the several part of the vector<Punto>


        int sum = 0;
        for(int i = 0; i < numNodes; i++){
            if(i == 0){
                datasetDisp[i] = 0;
            }else{
                sum += pointsPerNode[i-1];
                datasetDisp[i] = sum;
            }
        }
    }

    MPI_Scatter(pointsPerNode, 1, MPI_INT, &num_local_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Node " << rank << " has num of points equal to " << num_local_points << "\n" << endl;

    localDataset.resize(num_local_points);

    MPI_Scatterv(dataset.data(), pointsPerNode, datasetDisp, pointType, localDataset.data(), num_local_points, pointType, 0, MPI_COMM_WORLD);


    if(rank == 1){
        cout << "First element of Node 1 has values : " << localDataset[0].values[0] << ". Last one is " << localDataset[0].values[19] << endl;
    }
}


