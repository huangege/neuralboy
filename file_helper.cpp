//
// Created by HuanGege on 2020/4/25.
//
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
//#include "resources/iris.data"

using namespace std;

class FileHelper {
public:
    vector<vector<double>> getData() {
        string path = "/Users/huangege/ClionProjects/NeuralBoy/resources/iris.data";
        vector<vector<double>> result;
        map<string, double> flowerIndex;

        flowerIndex["Iris-setosa"] = 0;
        flowerIndex["Iris-versicolor"] = 1;
        flowerIndex["Iris-virginica"] = 2;

        ifstream iFile(path);
        string line;

        if (iFile.is_open()) {
            while(getline(iFile, line) && line.size() > 1) {
                vector<double> lineVec;
                int start = 0;
                for (int i = 0; i < 4; i++) {
                    int end = line.find(',', start);
                    lineVec.push_back(stod(line.substr(start, end - start)));
                    start = end + 1;
                }
                lineVec.push_back(flowerIndex[line.substr(start)]);
                result.push_back(lineVec);
            }
            iFile.close();
        } else {
            cout << "Something wrong, open failed." << endl;
        }

        return result;
    }
};



