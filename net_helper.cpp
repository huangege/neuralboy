//
// Created by HuanGege on 2020/5/5.
//

#include "activations.h"
#include "net_helper.h"
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;


// layer class
Layer::Layer(int wRow, int wCol, int bRow, string activationName) {
    weights = MatrixXd::Random(wRow, wCol);
    if (bRow != 0)
        bias = MatrixXd::Random(bRow, 1);
    Activation layerActivation(activationName);
    activation = layerActivation.activation;
    derivativeActivation = layerActivation.derivativeActivation;
}


// network class
void NN::add_layer(int outLen, bool useBias, string activationName) {
    int curInputLen = inputLength;
    if (!nn.empty()) curInputLen = nn.back().weights.rows();
    int biasLen = outLen;
    if (!useBias) biasLen = 0;
    Layer newLayer = Layer(outLen, curInputLen, biasLen, activationName);
    nn.push_back(newLayer);
}

pair<vector<VectorXd>, vector<VectorXd>> NN::forwardPropagation(VectorXd &x) {
    vector<VectorXd> xNet;
    vector<VectorXd> zNet;

    xNet.push_back(x);

    for (auto i: nn) {
        VectorXd zCur = i.weights * xNet.back();
        if (i.weights.rows() == i.bias.rows()) zCur += i.bias; // use bias
        zNet.push_back(zCur);

        VectorXd xCur = i.activation(zCur);
        xNet.push_back(xCur);
    }

    return {xNet, zNet};
}

vector<VectorXd> NN::backPropagation(VectorXd &y, vector<VectorXd> &xNet, vector<VectorXd> &zNet) {
    vector<VectorXd> errors;

    VectorXd outError = nn.back().derivativeActivation(zNet.back()) * (xNet.back() - y);
    errors.push_back(outError);

    for (int i = nn.size() - 2; i >= 0; i--) {
        VectorXd curError = nn[i].derivativeActivation(zNet[i]) *
                            nn[i + 1].weights.transpose() * errors.back();
        errors.push_back(curError);
    }

    return errors;
}

void NN::updateWeights(vector<VectorXd> &errorWeights, vector<VectorXd> &xNet, double leaningRate) {
    int totalLayers = nn.size();

    for (int i = 0; i < nn.size(); i++) {
        nn[i].weights = nn[i].weights - leaningRate * errorWeights[totalLayers - 1 - i] * xNet[i].transpose();
    }
}

double NN::getLossValue(vector<VectorXd> &xNet, VectorXd &y) {
    auto lossMat = xNet.back() - y;
    auto lossMatElementSquare = lossMat.array() * lossMat.array() / 2.0;

    return lossMatElementSquare.sum();
}

/**
 *
 * @param xs
 * @param ys
 * @param learningRate
 * @return
 */
double NN::train(vector<VectorXd> xs, vector<VectorXd> ys, double learningRate) {
    double curLossValue = 0.0;
    for (int i = 0; i < xs.size(); i++) {
        auto xNetAndzNet = forwardPropagation(xs[i]);
        auto errors = backPropagation(ys[i], xNetAndzNet.first, xNetAndzNet.second);
        curLossValue = getLossValue(xNetAndzNet.first, ys[i]);

        updateWeights(errors, xNetAndzNet.first, learningRate);
    }

    return curLossValue;
}

void NN::printNet() {
    string flagString = "=======================";
    cout << "Your net is like:" << endl;
    cout << "Layer input: " << inputLength << endl;
    for (int i = 0; i < nn.size(); i++) {
        cout << "Layer " << i << ": " << nn[i].weights.rows() << endl;
    }
    cout << flagString << endl;
}


// below is space for SuperNN
/**
 * Update weights and bias using mini-batching GD.
 * @param batchErrors
 * @param batchXNet
 * @param leaningRate
 */
void SuperNN::updateWeights(vector<vector<VectorXd>> &batchErrors, vector<vector<VectorXd>> &batchXNet, double leaningRate) {
    int totalLayers = nn.size();
    vector<MatrixXd> avgGradsWeights;
    vector<VectorXd> avgGradsBias;

    for (int i = 0; i < batchErrors.size(); i++) {
        for (int j = 0; j < nn.size(); j++) {
            MatrixXd curGrad = batchErrors[i][totalLayers - 1 - j] * batchXNet[i][j].transpose();
            if (i == 0) {
                avgGradsWeights.push_back(curGrad);
                avgGradsBias.push_back(batchErrors[i][totalLayers - 1 - j]);
            }
            else {
                avgGradsWeights[j] += curGrad;
                avgGradsBias[j] += batchErrors[i][totalLayers - 1 - j];
            }
        }
    }

    // do average
    for (int i = 0; i < nn.size(); i++) {
        avgGradsWeights[i] /= batchErrors.size();
        avgGradsBias[i] /= batchErrors.size();
    }

    // update weights and bias
    for (int i = 0; i < nn.size(); i++) {
        nn[i].weights = nn[i].weights - leaningRate * avgGradsWeights[i];
        if (nn[i].bias.size() == nn[i].weights.rows())
            nn[i].bias = nn[i].bias - leaningRate * avgGradsBias[i];
    }
}

/**
 * Support mini-batching Gradient Descent compared with NN's train.
 * @param xs
 * @param ys
 * @param learningRate
 * @return loss value of the batch.
 */
double SuperNN::train(vector<VectorXd> xs, vector<VectorXd> ys, double learningRate) {
    vector<vector<VectorXd>> batchErrors;
    vector<vector<VectorXd>> batchXNet;
    double lossValue = 0.0;

    double curLossValue = 0.0;
    for (int i = 0; i < xs.size(); i++) {
        auto xNetAndzNet = forwardPropagation(xs[i]);
        auto curErrors = backPropagation(ys[i], xNetAndzNet.first, xNetAndzNet.second);

        batchErrors.push_back(curErrors);
        batchXNet.push_back(xNetAndzNet.first);

        lossValue += getLossValue(xNetAndzNet.first, ys[i]);
    }

    lossValue /= xs.size();

    updateWeights(batchErrors, batchXNet, learningRate);

    return lossValue;
}
