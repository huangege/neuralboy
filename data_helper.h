//
// Created by HuanGege on 2020/6/21.
//

#ifndef NEURALBOY_DATA_HELPER_H
#define NEURALBOY_DATA_HELPER_H

#include <vector>

using namespace Eigen;
using namespace std;

pair<vector<VectorXd>, vector<VectorXd>> simulateData(int num, int inputLength, int outputLength);

void writeData(vector<double> loss, string filePath, string mode = "overwrite");
void writeData(vector<vector<double>> loss, string filePath, string mode = "overwrite");

#endif //NEURALBOY_DATA_HELPER_H
