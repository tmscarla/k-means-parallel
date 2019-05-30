#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include "Point.h"
#include "Node.h"
#include <stddef.h>
#include <mpi.h>
#include <fstream>
#include <sstream>


#define MAX_DIM 100

using namespace std;

/*Dobbiamo per forza creare una classe o una struct Point per fare in modo che il punto venga salvato in un
 * vettore. Non si può creare un vettore di arrays di double*/


int main(int argc, char *argv[]) {
    srand(time(NULL));

    int numNodes, rank;
    const int tag = 13;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);

    cout << "I'm rank: " << rank << endl;

    int total_values;
    int total_points;
    int K, max_iterations;
    vector<Punto> dataset;

    Node node(rank, MPI_COMM_WORLD);

    //string path = "twitter_points_20 copiaridotta.csv";
    node.readDataset();
    node.scatterDataset();
    node.extractCluster();
    for (int it = 0; it < 10; it++) {
        cout << "Iteration " << it << " starts!" << endl;
        int notChanged = node.run(it);
        cout << "Iteration " << it << " ends!" << endl;

        if(rank == 0){
            cout << "Global not changed = " << notChanged << ". NumNodes = " << numNodes << endl;
        }

        if(notChanged == numNodes){
            cout << "No more changes, k-means terminates at iteration " << it << endl;
            break;
        }

    }

    cout << "Get the memberships!!! " << endl;
    node.computeGlobalMembership();
    if(rank == 0){
        int* gm;
        gm = node.getGlobalMemberships();
        int numPoints = node.getNumPoints();
        /*for(int i = 0; i < numPoints; i++){
            cout << "Point " << i << " belongs to cluster " << gm[i] << endl;
        }*/

        node.printClusters();
    }


    MPI_Finalize();
    //Un punto è un array di valori double. Leggere prima la dimensione di ciascun punto, poi creare i punti
    // come array con quella dimensione

}