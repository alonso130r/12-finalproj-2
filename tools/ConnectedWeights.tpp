//
// Created by adamk on 2025-01-14.
//

#include "ConnectedWeights.h"

template <typename Type>
ConnectedWeights<Type>::ConnectedWeights(FullyConnectedLayer<Type> const& layer) {
    in_features = layer.in_features;
    out_features = layer.out_features;
    weights = layer.weights;
    biases = layer.biases;
}

template <typename Type>
WeightStructType ConnectedWeights<Type>::getType() const {
    return WeightStructType::ConnectedWeights;
}

template<typename Type>
void ConnectedWeights<Type>::serialize(std::ofstream &out) const {
    out.write(reinterpret_cast<const char*>(&in_features), sizeof(in_features));
    out.write(reinterpret_cast<const char*>(&out_features), sizeof(out_features));
    out.write(reinterpret_cast<const char*>(&weights), sizeof(weights));
    out.write(reinterpret_cast<const char*>(&biases), sizeof(biases));
}

template<typename Type>
std::shared_ptr<FullyConnectedLayer<Type>> ConnectedWeights<Type>::deserialize(std::ifstream &in) {
    int in_features_t;
    int out_features_t;
    WeightsMatrix weights_t;
    std::vector<Type> biases_t;

    in.read(reinterpret_cast<char*>(&in_features_t), sizeof(in_features_t));
    in.read(reinterpret_cast<char*>(&out_features_t), sizeof(out_features_t));
    in.read(reinterpret_cast<char*>(&weights_t), sizeof(weights_t));
    in.read(reinterpret_cast<char*>(&biases_t), sizeof(biases_t));

    auto temp = std::make_shared<FullyConnectedLayer<Type>>(in_features_t, out_features_t);
    temp->weights = weights_t;
    temp->biases = biases_t;
    return temp;
}

