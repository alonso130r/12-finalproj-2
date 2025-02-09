//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_FULLYCONNECTEDLAYER_H
#define INC_12_FINALPROJ_2_FULLYCONNECTEDLAYER_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include "../tools/Tensor.h"
#include "Layer.h"

template <typename Type>
class FullyConnectedLayer : public Layer<Type> {
    typedef std::vector<std::vector<Type>> WeightsMatrix;
public:
    int in_features;
    int out_features;

    WeightsMatrix weights;
    std::vector<Type> biases;

    // gradients
    WeightsMatrix dWeights;
    std::vector<Type> dBiases;

    FullyConnectedLayer(int in_features, int out_features);

    void initializeParams();

    void zeroGrad() override;

    std::shared_ptr<WeightStruct<Type>> saveWeights() override;

    [[nodiscard]] ssize_t getNumParams() const override;
};

#include "FullyConnectedLayer.tpp"

#endif //INC_12_FINALPROJ_2_FULLYCONNECTEDLAYER_H
