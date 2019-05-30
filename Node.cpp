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
#include <math.h>
#include "Node.h"

Node::Node(int rank, MPI_Comm comm) : rank(rank), comm(comm), notChanged(1) {

    //Create vector<Point> Datatype in order to be able to send and receive element of struct Punto
    int blocksize[] = {MAX_DIM, 1, 1};
    MPI_Aint displ[] = {0, offsetof(Punto, id), offsetof(Punto, size)};
    MPI_Datatype blockType[] = {MPI_DOUBLE, MPI_INT, MPI_INT};

    MPI_Type_create_struct(3, blocksize, displ, blockType, &pointType);
    MPI_Type_commit(&pointType);

}

void Node::readDataset() {

    if (rank == 0) {

        // READ DATASET
        ifstream infile("data/iris_1.csv");
        string line;
        cout << "Reading file.." << endl;

        int count = 0;
        int num = 0;
        while (getline(infile, line, '\n')) {
            if (count == 0) {
                // cout << "la prima riga è " << line << endl;
                stringstream ss(line);
                getline(ss, line, ',');
                total_values = stoi(line);
                cout << "Total values is: " << total_values << endl;

                getline(ss, line, ',');
                K = stoi(line);
                cout << "Number of clusters K is: " << K << endl;

                //Adding here other values of the first row

                getline(ss, line, '\n');
                max_iterations = stoi(line);
                cout << "Max iteration is: " << max_iterations << endl;
                count++;
            } else {
                Punto point;
                point.id = num;
                point.size = total_values;
                int i = 0;
                stringstream ss(line);
                while (getline(ss, line, ',')) {
                    point.values[i] = stod(line);
                    i++;
                }
                num++;
                dataset.push_back(point);
            }
        }

        infile.close();

        cout << "Reading ended" << endl;
    }
}


void Node::scatterDataset() {
    /* Scatter dataset among nodes */

    int numNodes;
    MPI_Comm_size(comm, &numNodes);

    int pointsPerNode[numNodes];
    int datasetDisp[numNodes];

    if (rank == 0) {
        numPoints = dataset.size();
        cout << "Total points: " << numPoints << endl;

        int partial = numPoints / numNodes;
        fill_n(pointsPerNode, numNodes, partial);

        /* Assing remainder R of the division to first R node*/
        if ((numPoints % numNodes) != 0) {
            int r = numPoints % numNodes;

            for (int i = 0; i < r; i++) {
                pointsPerNode[i] += 1;
            }
        }

        //Vector contains strides (https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node72.html) so, we need to
        // know precisely where starting to divide the several part of the vector<Punto>

        int sum = 0;
        for (int i = 0; i < numNodes; i++) {
            if (i == 0) {
                datasetDisp[i] = 0;
            } else {
                sum += pointsPerNode[i - 1];
                datasetDisp[i] = sum;
            }
        }
    }

    MPI_Scatter(pointsPerNode, 1, MPI_INT, &num_local_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Node " << rank << " has num of points equal to " << num_local_points << "\n" << endl;

    localDataset.resize(num_local_points);

    MPI_Scatterv(dataset.data(), pointsPerNode, datasetDisp, pointType, localDataset.data(), num_local_points,
                 pointType, 0, MPI_COMM_WORLD);


    //Send the dimension of points to each node
    MPI_Bcast(&total_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

    memberships.resize(num_local_points);

    for (int i = 0; i < num_local_points; i++) {
        memberships[i] = -1;
    }

    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

}


void Node::extractCluster() {

    /* Initially to extract the clusters, we choose randomly K point of the dataset. This action is performed
     * by the Node 0, who sends them to other nodes in broadcast. Ids of clusters are the same of their initial centroid point  */

    if (rank == 0) {
        if (K >= dataset.size()) {
            cout << "ERROR: Number of cluster >= number of points " << endl;
            return;
        }

        vector<int> clusterIndices;
        vector<int> prohibitedIndices;

        for (int i = 0; i < K; i++) {
            while (true) {            //TODO fix double loop
                int randIndex = rand() % dataset.size();

                if (find(prohibitedIndices.begin(), prohibitedIndices.end(), randIndex) == prohibitedIndices.end()) {
                    prohibitedIndices.push_back(randIndex);
                    clusterIndices.push_back(randIndex);
                    break;
                }
            }
        }

        //Take points which refer to clusterIndices and send them in broadcast to all Nodes

        /* C++11 extension
        for(auto&& x: clusterIndices){      //packed range-based for loop  (https://www.quora.com/How-do-I-iterate-through-a-vector-using-for-loop-in-C++)
            clusters.push_back(dataset[x]);
        }
         */

        for(int i = 0; i < clusterIndices.size(); i++) {
            clusters.push_back(dataset[clusterIndices[i]]);
        }


        cout << "The id of point chosen for initial values of cluster are : " << endl;
        for (int i = 0; i < clusters.size(); i++) {
            cout << "Cluster referring to point with id: " << clusters[i].id << " with first value "
                 << clusters[i].values[0] << endl;
        }


    }

    //Send the number of clusters in broadcast
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    clusters.resize(K);

    //Send the clusters centroids values
    MPI_Bcast(clusters.data(), K, pointType, 0, MPI_COMM_WORLD);

}

int Node::getIdNearestCluster(Punto p) {
    double sum = 0.0;
    double min_dist;


    int idCluster = 0;  //is the position in the vector clusters, not the id of the point that represents the initial centroid


    //Initialize sum and min_dist
    for (int i = 0; i < total_values; i++) {
        sum += pow(clusters[0].values[i] - p.values[i], 2.0);
    }

    min_dist = sqrt(sum);

    //if(p.id == 54){
        //cout << "[0] Point " << p.id << " ha min_dist = " << min_dist << endl;
    //}

    //Compute the distance from others clusters
    for (int k = 1; k < K; k++) {

        double dist;
        sum = 0.0;

        for (int i = 0; i < total_values; i++) {
            sum += pow(clusters[k].values[i] - p.values[i], 2.0);
        }

        dist = sqrt(sum);

       // if(p.id == 54){
            //cout << "Point " << p.id << " ha min_dist = " << min_dist << endl;
            //cout << "Point " << p.id << " dista dal cluster in pos " << k << " di " << dist << endl;
        //}

        if (dist < min_dist) {
            //if(p.id == 54){
                //cout << "Point " << p.id << " ha dist = " << dist << " < " << min_dist << " = min_dist" << endl;
            //}
            min_dist = dist;
            idCluster = k;

        }
    }
    if(p.id == 54) {
        //cout << "The nearest cluster is in pos " << idCluster << endl;
    }
    return idCluster;
}


int Node::run(int it) {

    notChanged = 1;
    localSum.resize(K);

    int resMemCounter[K];

    if (it == 0) {
        // memCounter va inizializzato alla prima iterazione del ciclo. Successivamente va solo modificato in modo che se
        // un punto cambia cluster di appartenenza, counter del numero di punti nel cluster vecchio viene decrementato, quello
        // nuovo invece viene incrementato.
        memCounter = new int[K];
        for (int k = 0; k < K; k++) {
            memCounter[k] = 0;
        }
    }

    for (int i = 0; i < localDataset.size(); i++) {

        int old_mem = memberships[i];
        int new_mem = getIdNearestCluster(localDataset[i]);

        if(new_mem != old_mem){
            memberships[i] = new_mem;
            memCounter[new_mem] += 1;
            if(old_mem != -1){
                memCounter[old_mem] -= 1;
            }
            notChanged = 0;
        }

        /*
        if (rank == 0) {
            cout << "In Node " << rank << " point " << localDataset[i].id << " belongs to cluster at position "
                 << memberships[i] << ". The cluster id is " << clusters[memberships[i]].id << endl;
        }
         */
    }

    // Reset of resMemCounter at each iteration
    for (int k = 0; k < K; k++) {
        resMemCounter[k] = 0;
    }

    MPI_Allreduce(memCounter, resMemCounter, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  //In questo modo si ottiene il numero dei punti che appartengono a ciascun cluster
    /*
    for (int i = 0; i < K; i++) {
        cout << "Cluster at position " << i << " contains " << resMemCounter[i] << " points" << endl;
    }
    */
    updateLocalSum();

    /*To recalculate cluster centroids, we sum locally the points (values-to-values) which belong to a cluster.
     * The result will be a point with values equal to that sum. This point is sent (with AllReduce) to each
     * node by each node with AllReduce which computes the sum of each value-to-value among all sent points.
     */

    //Since AllReduce doesn't support operations with vector, we need to serialize the vector into an array (reduceArr)
    // and once AllReduce is done, we need to re-arrange the array obtained into a vector of Point

    double reduceArr[K * total_values];
    double results[K * total_values];


    for (int i = 0; i < K; i++) {
        for (int j = 0; j < total_values; j++) {
            reduceArr[i * total_values + j] = localSum[i].values[j];
        }
    }

    MPI_Allreduce(reduceArr, results, K * total_values, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    /*
    for (int i = 0; i < K; i++) {

        cout << "After MPI_Allreduce, In Node " << rank << " the localSum of cluster  " << clusters[i].id
             << " has values 0 equal to " << results[i * total_values] << "\n" << endl;

    }*/

    for (int k = 0; k < K; k++) {
        if(rank == 0) {
            cout << "Cluster in position " << k << " contains " << resMemCounter[k] << " points" << endl;
        }
        for (int i = 0; i < total_values; i++) {
            if(rank == 0) {
                //cout << results[k * total_values + i] << "-->";
            }
            if(resMemCounter[k] != 0) {
                results[k * total_values +
                        i] /= resMemCounter[k];
                clusters[k].values[i] = results[k * total_values + i];
            }else{
                results[k * total_values +
                        i] /= 1;
                clusters[k].values[i] = results[k * total_values + i];
            }
            if(rank == 0) {
                //cout << results[k * total_values + i] << endl;
            }
        }
        /*
        cout << "After the division, In Node " << rank << " the results of cluster in position " << k
             << " has values 0 equal to " << results[k * total_values] << ". The cluster at position " <<
             k << " with id " << clusters[k].id << " has first values = " << clusters[k].values[0] << endl;
             */
    }


    MPI_Barrier(MPI_COMM_WORLD);

    int globalNotChanged;

    /*To stop the iteration k-means before reaching the max_iterations, in all Node no cluster has to change its centroids
     * w.r.t. preceding iteration. In order to reach this goal, we set a variable [notChanged] to 1 if no point changes its
     * membership, 0 otherwise. Then with All_Reduce all Nodes know how many nodes [globalNotChanged] have their points unchanged and if
     * if that number is equal to the number of processes it means that all points have not changed their memberships*/
    MPI_Allreduce(&notChanged, &globalNotChanged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return globalNotChanged;
}


void Node::updateLocalSum() {
    //reset LocalSum at each iteration
    for(int k = 0; k < localDataset.size(); k++){
        for(int j = 0; j < total_values; j++){
            localSum[k].values[j] = 0;
        }
    }

    for (int i = 0; i < localDataset.size(); i++) {
        if(rank == 0){
            cout << "Point " << localDataset[i].id << " has values: " ;
        }
        for (int j = 0; j < total_values; j++) {
            if(rank == 0){
                cout << localDataset[i].values[j] << " , ";
            }
            localSum[memberships[i]].values[j] += localDataset[i].values[j];
        }
        if(rank == 0) {
            cout << "\nLocalSum at position " << memberships[i] << " : " ;
            for (int f = 0; f < total_values; f++) {
                cout << localSum[memberships[i]].values[f] << " , ";
            }
        }
        cout << "\n" << endl;
        //cout << "In Node " << rank << " point " << localDataset[i].id << " belongs to cluster at position " << memberships[i] << ". The cluster id is " << clusters[memberships[i]].id << endl;
    }
}

void Node::computeGlobalMembership() {


    globalMembership = new int[numPoints];
    int localMem[numPoints];
    int globalMember[numPoints];
    for (int i = 0; i < numPoints; i++) {
        globalMember[i] = 0;
        localMem[i] = 0;
    }

    for (int i = 0; i < num_local_points; i++) {
        int p_id = localDataset[i].id;
        int c_id = memberships[i];
        localMem[p_id] = c_id;

        //cout << "In Node " << rank << " point " << p_id << " belongs to cluster " << c_id << endl;
    }


    MPI_Reduce(&localMem, &globalMember, numPoints, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int j = 0; j < numPoints; j++) {
           // cout << "Point " << j << " belongs to cluster " << globalMember[j] << endl;
            globalMembership[j] = globalMember[j];
        }
    }


}

int *Node::getGlobalMemberships() {
    return globalMembership;
}

int Node::getNumPoints() {
    return numPoints;
}

void Node::printClusters() {
    int total = 0;
    for (int i = 0; i < K; i++) {
        cout << "Cluster " << i << " contains: " << endl;
        int count = 0;
        for (int j = 0; j < numPoints; j++) {
            if (i == globalMembership[j]) {
                cout << "Point " << dataset[j].id << endl;
                count++;
            }
        }
        //cout << "[printCluster] Cluster at position " << i << " contains " << count << " points" << endl;
        //total += count;
    }
    //cout << "Total number in cluster are: " << total << endl;
}


