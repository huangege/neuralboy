//
// Created by HuanGege on 2020/6/21.
//

#include "activations.h"
#include "data_helper.h"
#include <fstream>

//#include "resources/iris.data"

using namespace Eigen;
using namespace std;

pair<vector<VectorXd>, vector<VectorXd>> simulateData(int num, int inputLength, int outputLength) {
    MatrixXd weights = MatrixXd::Random(outputLength, inputLength);
    vector<VectorXd> xs;
    vector<VectorXd> ys;

    for (int i = 0; i < num; i++) {
        VectorXd x = VectorXd::Random(inputLength);
        xs.push_back(x);
        VectorXd y = weights * x;
        y = Activation::sigmoid(y);
        ys.push_back(y);
    }

    return {xs, ys};
}

void writeData(vector<double> loss, string filePath, string mode) {
    ofstream file;
    if (mode == "append") {
        file.open(filePath, ios_base::app);
    } else {
        file.open(filePath, ios_base::out);
    }

    for (auto i: loss) {
        file << i << endl;
    }

    file.close();
}

void writeData(vector<vector<double>> loss, string filePath, string mode) {
    ofstream file;
    if (mode == "append") {
        file.open(filePath, ios_base::app);
    } else {
        file.open(filePath, ios_base::out);
    }

    for (auto i: loss) {
        file  << i[0];
        for (int j = 1; j < i.size(); j++) file << "," << i[j] << endl;
    }

    file.close();
}

