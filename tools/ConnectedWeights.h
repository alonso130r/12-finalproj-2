//
// Created by adamk on 2025-01-14.
//

#ifndef CONNECTEDWEIGHTS_H
#define CONNECTEDWEIGHTS_H

#include <vector>
#include "../layers/FullyConnectedLayer.h"

template <typename Type>
struct ConnectedWeights : public WeightStruct<Type> {
    typedef std::vector<std::vector<Type>> WeightsMatrix;

    int in_features;
    int out_features;

    WeightsMatrix weights;
    std::vector<Type> biases;

    ConnectedWeights(const FullyConnectedLayer<Type>& layer);
    WeightStructType getType() const override;
    void serialize(std::ofstream& out) const override;
    static std::shared_ptr<FullyConnectedLayer<Type>> deserialize(std::ifstream& in);
};

#include "ConnectedWeights.tpp"

#endif //CONNECTEDWEIGHTS_H
