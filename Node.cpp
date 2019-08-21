//
// Created by Claudio Russo Introito on 2019-04-28.
//

#include <mpi.h>
//#include "/usr/local/opt/libomp/include/omp.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stddef.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include "Node.h"

//t= omp_get_wtime();

Node::Node(int rank, MPI_Comm comm) : rank(rank), comm(comm), notChanged(1) {

    //Create vector<Point> Datatype in order to be able to send and receive element of struct Point
    int blocksize[] = {MAX_DIM, 1, 1};
    MPI_Aint displ[] = {0, offsetof(Point, id), offsetof(Point, size)};
    MPI_Datatype blockType[] = {MPI_DOUBLE, MPI_INT, MPI_INT};

    MPI_Type_create_struct(3, blocksize, displ, blockType, &pointType);
    MPI_Type_commit(&pointType);
    total_time = 0;
    lastIteration = 0;
}

double Node::squared_norm(Point p1, Point p2){
    // This operation compute || p1 - p2 ||^2
    double sum = 0.0;
    for(int j = 0; j < total_values; j++){
        sum += pow(p1.values[j] - p2.values[j], 2.0);
    }
    return sum;
}

void Node::setLastIteration(int lastIt) {
    lastIteration = lastIt;
}

void Node::readDataset() {

    if (rank == 0) {
        double start = MPI_Wtime();

        // READ DATASET
        string filename, point_dimension;
        cout << "Enter point dimension: 20, 50 or 100\n";
        getline(cin, point_dimension);
        filename =  "data/tweets_points_" + point_dimension + ".csv";

        ifstream infile(filename);
        string line;
        cout << "Reading file.." << endl;

        int count = 0;
        int num = 0;
        while (getline(infile, line, '\n')) {
            if (count == 0) {
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
                Point point;
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

        double end = MPI_Wtime();
        total_time += end - start;

    }
}


void Node::scatterDataset() {
    /* Scatter dataset among nodes */

    double start = MPI_Wtime();

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
        // know precisely where starting to divide the several part of the vector<Point>

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

    MPI_Scatter(pointsPerNode, 1, MPI_INT, &num_local_points, 1, MPI_INT, 0, comm);

    //cout << "Node " << rank << " has num of points equal to " << num_local_points << "\n" << endl;

    localDataset.resize(num_local_points);

    MPI_Scatterv(dataset.data(), pointsPerNode, datasetDisp, pointType, localDataset.data(), num_local_points,
                 pointType, 0, comm);


    //Send the dimension of points to each node
    MPI_Bcast(&total_values, 1, MPI_INT, 0, comm);

    memberships.resize(num_local_points);

    for (int i = 0; i < num_local_points; i++) {
        memberships[i] = -1;
    }

    MPI_Bcast(&numPoints, 1, MPI_INT, 0, comm);
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, comm);

    double end = MPI_Wtime();
    total_time += end - start;
}


void Node::extractCluster() {

    /* Initially to extract the clusters, we choose randomly K point of the dataset. This action is performed
     * by the Node 0, who sends them to other nodes in broadcast. Ids of clusters are the same of their initial centroid point  */

    if (rank == 0) {
        double start = MPI_Wtime();

        if (K >= dataset.size()) {
            cout << "ERROR: Number of cluster >= number of points " << endl;
            return;
        }

        vector<int> clusterIndices;
        vector<int> prohibitedIndices;

        for (int i = 0; i < K; i++) {
            while (true) {
                int randIndex = rand() % dataset.size();

                if (find(prohibitedIndices.begin(), prohibitedIndices.end(), randIndex) == prohibitedIndices.end()) {
                    prohibitedIndices.push_back(randIndex);
                    clusterIndices.push_back(randIndex);
                    break;
                }
            }
        }

        string string_choice;
        int choice;
        bool gameOn = true;

        while(gameOn){
            cout << "\nChoose initialization method" <<endl;
            cout << "1) Random \n";
            cout << "2) First k points\n";
            cout << "3) 13 points referring to 13 different topics\n";
            cout << "Enter your choice and press return: ";

            getline(cin, string_choice);
            choice = atoi(string_choice.c_str());

            switch (choice){
                case 1: {
                    for (int i = 0; i < clusterIndices.size(); i++) {
                        clusters.push_back(dataset[clusterIndices[i]]);
                    }
                    gameOn = false;
                    break;
                }
                case 2: {
                    for (int i = 0; i < clusterIndices.size(); i++) {
                        clusters.push_back(dataset[i]);
                    }
                    gameOn = false;
                    break;
                }

                case 3: {
                    // Fixed point referring to different topics
                    vector<int> allDiffTopicsPoint = {0, 1, 2, 4, 5, 6, 8, 9, 13, 17, 19, 7030, 4486};
                    for (int i = 0; i < allDiffTopicsPoint.size(); i++) {
                        clusters.push_back(dataset[allDiffTopicsPoint[i]]);
                    }
                    gameOn = false;
                    break;
                }

                default: {
                    cout << "Not a Valid Choice. \n";
                    cout << "Choose again.\n";
                    getline(cin, string_choice);
                    choice = atoi(string_choice.c_str());
                    break;
                }
            }
        }

        /*
        //Take points which refer to clusterIndices and send them in broadcast to all Nodes
        for(int i = 0; i < clusterIndices.size(); i++) {
            //clusters.push_back(dataset[clusterIndices[i]]);
            //clusters.push_back(dataset[i]); //TODO delete this line, Adding random selection (line above)
        }

        // Fixed point referring to different topics
        int allDiffTopicsPoint = [0, 1, 2, 4, 5, 6, 8, 9, 13, 17, 19, 7030, 4486];
        */

        /*
        cout << "The id of point chosen for initial values of cluster are : " << endl;
        for (int i = 0; i < clusters.size(); i++) {
            cout << "Cluster referring to point with id: " << clusters[i].id << " with first value "
                 << clusters[i].values[0] << endl;
        }*/

        double end = MPI_Wtime();
        total_time += end - start;
    }

    //Send the number of clusters in broadcast
    MPI_Bcast(&K, 1, MPI_INT, 0, comm);

    clusters.resize(K);

    //Send the clusters centroids values
    MPI_Bcast(clusters.data(), K, pointType, 0, comm);

}

int Node::getIdNearestCluster(Point p) {
    double sum = 0.0;
    double min_dist;


    int idCluster = 0;  //is the position in the vector clusters, not the id of the point that represents the initial centroid


    //Initialize sum and min_dist
    for (int i = 0; i < total_values; i++) {
        sum += pow(clusters[0].values[i] - p.values[i], 2.0);
    }

    min_dist = sqrt(sum);

    //Compute the distance from others clusters
    for (int k = 1; k < K; k++) {

        double dist;

        sum = squared_norm(clusters[k], p);

        dist = sqrt(sum);

        if (dist < min_dist) {
            min_dist = dist;
            idCluster = k;

        }
    }
    return idCluster;
}


int Node::run(int it) {
    double start = MPI_Wtime();

    double t_i, t_f;

    notChanged = 1;
    localSum.resize(K);

    int resMemCounter[K];

    // Reset of resMemCounter at each iteration
    fill_n(resMemCounter, K, 0);

    if (it == 0) {
        // memCounter va inizializzato alla prima iterazione del ciclo. Successivamente va solo modificato in modo che se
        // un punto cambia cluster di appartenenza, counter del numero di punti nel cluster vecchio viene decrementato, quello
        // nuovo invece viene incrementato.
        memCounter = new int[K] ();
    }

    #pragma omp parallel for shared(memCounter) num_threads(4)
    for (int i = 0; i < localDataset.size(); i++) {

        int old_mem = memberships[i];
        int new_mem = getIdNearestCluster(localDataset[i]);

        if(new_mem != old_mem){
            memberships[i] = new_mem;

            //critical section : memCounter is a vector and for each iteration we update at most two elements. For those element we
            // need to guaratee the atomicity of the operation, but that lock must not block the access of other processes to
            // other array elements.
            // using atomic pragma resolves our issue: https://stackoverflow.com/questions/17553282/how-to-lock-element-of-array-using-tbb-openmp
            #pragma omp atomic update
            memCounter[new_mem]++;
            if(old_mem != -1){
                #pragma omp atomic update
                memCounter[old_mem]--;
            }

            notChanged = 0;
        }
    }


    MPI_Allreduce(memCounter, resMemCounter, K, MPI_INT, MPI_SUM, comm);  // We obtain the number of points that belong to each cluster
    updateLocalSum();

    /*To recalculate cluster centroids, we sum locally the points (values-to-values) which belong to a cluster.
     * The result will be a point with values equal to that sum. This point is sent (with AllReduce) to each
     * node by each node with AllReduce which computes the sum of each value-to-value among all sent points.
     */

    //Since AllReduce doesn't support operations with vector, we need to serialize the vector into an array (reduceArr)
    // and once AllReduce is done, we need to re-arrange the array obtained into a vector of Point
    double reduceArr[K * total_values];
    double reduceResults[K * total_values];

    t_i = omp_get_wtime();
    #pragma omp parallel for num_threads(4)   //Questo forse rallenta
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < total_values; j++) {
            reduceArr[i * total_values + j] = localSum[i].values[j];
        }
    }

    t_f = omp_get_wtime();
    //cout << "OMP time to reduceArr : " << t_f - t_i << endl;



    MPI_Allreduce(reduceArr, reduceResults, K * total_values, MPI_DOUBLE, MPI_SUM,
                  comm);

    for (int k = 0; k < K; k++) {
        if(rank == 0) {
            //cout << "Cluster in position " << k << " contains " << resMemCounter[k] << " points" << endl;
        }
        for (int i = 0; i < total_values; i++) {
            if(rank == 0) {
                //cout << reduceResults[k * total_values + i] << "-->";
            }
            if(resMemCounter[k] != 0) {
                reduceResults[k * total_values +
                        i] /= resMemCounter[k];
                clusters[k].values[i] = reduceResults[k * total_values + i];
            }else{
                reduceResults[k * total_values +
                        i] /= 1;
                clusters[k].values[i] = reduceResults[k * total_values + i];
            }
        }
    }

    int globalNotChanged;

    /*To stop the iteration k-means before reaching the max_iterations, in all Node no cluster has to change its centroids
     * w.r.t. preceding iteration. In order to reach this goal, we set a variable [notChanged] to 1 if no point changes its
     * membership, 0 otherwise. Then with All_Reduce all Nodes know how many nodes [globalNotChanged] have their points unchanged and if
     * if that number is equal to the number of processes it means that all points have not changed their memberships*/
    MPI_Allreduce(&notChanged, &globalNotChanged, 1, MPI_INT, MPI_SUM, comm);

    double end = MPI_Wtime();
    total_time += end - start;

    return globalNotChanged;
}


void Node::updateLocalSum() {
    //reset LocalSum at each iteration
    for(int k = 0; k < K; k++){
        for(int j = 0; j < total_values; j++){
            localSum[k].values[j] = 0;
        }
    }

    //TODO reduction on localSum
    double ti = omp_get_wtime();
    for (int i = 0; i < localDataset.size(); i++) {
        for (int j = 0; j < total_values; j++) {
            localSum[memberships[i]].values[j] += localDataset[i].values[j];
        }
    }

    double tf = omp_get_wtime();
    //cout << "Update local sum time : " << tf - ti << endl;
}

void Node::computeGlobalMembership() {

    globalMembership = new int[numPoints];

    int localMem[numPoints];
    int globalMember[numPoints];

    fill_n(localMem, numPoints, 0);
    fill_n(globalMember, numPoints, 0);


    for (int i = 0; i < num_local_points; i++) {
        int p_id = localDataset[i].id;
        int c_id = memberships[i];
        localMem[p_id] = c_id;
    }


    MPI_Reduce(&localMem, &globalMember, numPoints, MPI_INT, MPI_SUM, 0, comm);

    if (rank == 0) {
        for (int j = 0; j < numPoints; j++) {
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
    }
}

int Node::getMaxIterations(){
    return max_iterations;
}

void Node::writeClusterMembership(string filename){
    ofstream myfile;
    myfile.open("data/" + filename + ".csv");
    myfile << "Point_id,Cluster_id" << "\n";
    for(int p = 0; p < numPoints; p++){
        myfile << dataset[p].id << "," << clusters[globalMembership[p]].id << "\n";
    }
    myfile.close();
}

vector<double> Node::SSW() {
    //Standard Deviation of the sum of squares within each cluster
    vector<double> ssws;
    double ssw = 0.0;

    for (int k = 0; k < K; k++) {

        double dist;
        double sum;
        for (int j = 0; j < numPoints; j++) {
            if (k == globalMembership[j]) {

                sum = squared_norm(dataset[j], clusters[k]);

                dist = sqrt(sum);
                ssw += dist;
            }
        }
        ssws.push_back(sqrt(sum));
    }
    return ssws;
}


double Node::SSB() {
    //Standard Deviation of the sum of squares between clusters and mean point values

    //Find the mean point values among all points
    double sum = 0.0;
    double ssb = 0.0;
    Point mean;
    fill_n(mean.values, total_values, 0);

    for(int i = 0; i < numPoints; i++){
        mean += dataset[i];
    }
    for(int j = 0; j < total_values; j++){
        mean.values[j] /= numPoints;
    }

    for(int k = 0; k < K; k++){
        sum += squared_norm(clusters[k], mean);

    }
    ssb = sqrt(sum);

    return ssb;
}

void Node::getStatistics() {

    double *executionTimes;
    int numNodes;
    MPI_Comm_size(comm, &numNodes);

    if(rank == 0) {
        executionTimes = new double[numNodes];
    }

    MPI_Gather(&total_time, 1 , MPI_DOUBLE, executionTimes, 1, MPI_DOUBLE, 0, comm);

    if(rank == 0){
        cout << "---------------------  Statistics  ------------------------- " << endl;

        cout << " - Iteration computed: " << lastIteration << endl;

        cout << "\n - Execution time: " << endl;
        for(int i = 0; i < numNodes; i++){
            cout << "Process " << i << " tooks " << executionTimes[i] << endl;
        }

        cout << "\n - Number of points in each cluster" << endl;
        for(int k = 0; k < K; k++){
            int count = 0;
            for (int j = 0; j < numPoints; j++) {
                if (k == globalMembership[j]) {
                    count++;
                }
            }
            cout << "Cluster " << k << " : " << count << endl;

        }

        cout << "\n - Variance: " << endl;
        cout << "    - SSW: " << endl;
        vector<double> ssws = SSW();
        for(int i = 0; i < ssws.size(); i++) {
            cout << "   Cluster " << i << " : " << ssws[i] << endl;
        }

        cout << "\n   - SSB:    ";
        double ssb = SSB();
        cout << ssb << endl;

    }
}

