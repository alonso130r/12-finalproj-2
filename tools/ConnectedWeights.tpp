//
// Created by adamk on 2025-01-14.
//

#include "ConnectedWeights.h"

template <typename Type>
ConnectedWeights<Type>::ConnectedWeights(FullyConnectedLayer<Type> const& layer) {
    in_features = layer->in_features;
    out_features = layer->out_features;
    weights = layer->weights;
    biases = layer->biases;
}

template <typename Type>
WeightStructType ConnectedWeights<Type>::getType() const {
    return WeightStructType::ConnectedWeights;
}