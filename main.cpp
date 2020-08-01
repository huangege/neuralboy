#include <iostream>
#include <vector>
#include "net_helper.h"
#include "data_helper.h"

using namespace Eigen;
using namespace std;

int main() {
    int inputSize = 3;
    int outputSize = 1;
    string logPath = "/Users/huangege/JDProjects/NeuralBoy/loss_log";

    auto xsAndYs = simulateData(100, inputSize, outputSize);
    vector<VectorXd> xs = xsAndYs.first;
    vector<VectorXd> ys = xsAndYs.second;

    auto newNN = SuperNN(inputSize);
    newNN.add_layer(3, true, "sigmoid");
    newNN.add_layer(4, false, "tanh");
    newNN.add_layer(1, false);
    newNN.printNet();

    vector<double> lossVec;
    for (int i = 0; i < 1000; i++) {
        double curLoss = newNN.train(xs, ys, 1e-1);
        lossVec.push_back(curLoss);
    }

    writeData(lossVec, logPath);

    cout << "hello" << endl;

    return 0;
}
