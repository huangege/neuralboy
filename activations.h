//
// Created by HuanGege on 2020/6/21.
//

#ifndef NEURALBOY_ACTIVATIONS_H
#define NEURALBOY_ACTIVATIONS_H

#include <Eigen/Dense>
#include <string>

using namespace Eigen;

class Activation {
private:

public:
    // tanh
    static VectorXd tanh(VectorXd &x);
    static MatrixXd tanhDerivative(VectorXd &x);

    // sigmoid
    static VectorXd sigmoid(VectorXd &x);
    static MatrixXd sigmoidDerivative(VectorXd &x);

    // relu
    static VectorXd relu(VectorXd &x);
    static MatrixXd reluDerivative(VectorXd &x);


    VectorXd (*activation)(VectorXd &);
    MatrixXd (*derivativeActivation)(VectorXd &);

    Activation(std::string &activationName);
};

#endif //NEURALBOY_ACTIVATIONS_H
