//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_FULLYCONNECTEDLAYER_H
#define INC_12_FINALPROJ_2_FULLYCONNECTEDLAYER_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include "Tensor.h"

typedef std::vector<std::vector<double>> WeightsMatrix;

class FullyConnectedLayer {
public:
    int in_features;
    int out_features;

    WeightsMatrix weights;
    std::vector<double> biases;

    // gradients
    WeightsMatrix dWeights;
    std::vector<double> dBiases;

    FullyConnectedLayer(int in_features, int out_features);

    void initializeParams();

    void zeroGrad();
};


#endif //INC_12_FINALPROJ_2_FULLYCONNECTEDLAYER_H
