//
// Created by HuanGege on 2020/6/21.
//

#ifndef NEURALBOY_NET_HELPER_H
#define NEURALBOY_NET_HELPER_H

#include "activations.h"
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;


class Layer {
public:
    MatrixXd weights;
    MatrixXd bias;
    VectorXd (*activation)(VectorXd &);
    MatrixXd (*derivativeActivation)(VectorXd &);

    Layer(int wRow, int wCol, int bRow = 0, string activationName = "sigmoid");
};


class NN {
protected:
    int inputLength;
    vector<Layer> nn;

public:
    NN(int inputLength): inputLength(inputLength) {}

    void add_layer(int outLen, bool useBias=false, string activationName="sigmoid");

    pair<vector<VectorXd>, vector<VectorXd>> forwardPropagation(VectorXd &x);

    vector<VectorXd> backPropagation(VectorXd &y, vector<VectorXd> &xNet, vector<VectorXd> &zNet);

    void updateWeights(vector<VectorXd> &errorWeights, vector<VectorXd> &xNet, double leaningRate);

    double getLossValue(vector<VectorXd> &xNet, VectorXd &y);

    double train(vector<VectorXd> xs, vector<VectorXd> ys, double learningRate);

    void printNet();
};


class SuperNN: public NN {
protected:

public:
    SuperNN(int inputLength): NN(inputLength) {}

    void updateWeights(vector<vector<VectorXd>> &batchErrors, vector<vector<VectorXd>> &batchXNet, double leaningRate);

    double train(vector<VectorXd> xs, vector<VectorXd> ys, double learningRate);
};

#endif //NEURALBOY_NET_HELPER_H
