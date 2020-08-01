//
// Created by HuanGege on 2020/5/5.
//

#include "activations.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

double tanhScalar(double &x) {
    double e2x = exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

double sigmoidScalar(double &x) {
    return 1.0 / (1.0 + exp(0.0 - x));
}

double reluScalar(double &x) {
    return x > 0.0 ? x : 0.0;
}

// tanh
VectorXd Activation::tanh(VectorXd &x) {
    VectorXd result = x;
    for (int i = 0; i < x.size(); i++) {
        result(i) = tanhScalar(x(i));
    }

    return result;
}

MatrixXd Activation::tanhDerivative(VectorXd &x) {
    MatrixXd result = MatrixXd::Zero(x.size(), x.size());

    for (int i = 0; i < x.size(); i++) {
        result(i, i) = tanhScalar(x(i));
        result(i, i) = 1 - result(i, i) * result(i, i);
    }

    return result;
}

// sigmoid
VectorXd Activation::sigmoid(VectorXd &x) {
    VectorXd result = x;
    for (int i = 0; i < x.size(); i++) {
        result(i) = sigmoidScalar(x(i));
    }

    return result;
}

MatrixXd Activation::sigmoidDerivative(VectorXd &x) {
    MatrixXd result = MatrixXd::Zero(x.size(), x.size());

    for (int i = 0; i < x.size(); i++) {
        result(i, i) = sigmoidScalar(x(i));
        result(i, i) = (1.0 - result(i, i)) * result(i, i);
    }

    return result;
}


// relu
VectorXd Activation::relu(VectorXd &x) {
    VectorXd result = x;
    for (int i = 0; i < x.size(); i++) {
        result(i) = reluScalar(x(i));
    }

    return result;
}

MatrixXd Activation::reluDerivative(VectorXd &x) {
    MatrixXd result = MatrixXd::Zero(x.size(), x.size());
    for (int i = 0; i < x.size(); i++) {
        result(i, i) = x(i) > 0.0 ? 1.0 : 0.0;
    }

    return result;
}

Activation::Activation(string &activationName) {
    if (activationName == "tanh") {
        activation = tanh;
        derivativeActivation = tanhDerivative;
    } else if (activationName == "relu") {
        activation = relu;
        derivativeActivation = reluDerivative;
    } else {
        // sigmoid by default
        activation = sigmoid;
        derivativeActivation = sigmoidDerivative;
    }
}

