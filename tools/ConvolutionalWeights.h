//
// Created by adamk on 2025-01-14.
//

#ifndef CONVOLUTIONALWEIGHTS_H
#define CONVOLUTIONALWEIGHTS_H

#include <vector>
#include "../layers/ConvolutionLayer.h"

template <typename Type>
struct ConvolutionalWeights {
    typedef std::vector<std::vector<std::vector<Type>>> Tensor3D; // (channels, height, width)
    typedef std::vector<std::vector<std::vector<std::vector<Type>>>> Tensor4D; // (batch_size, channels, height, width)
    typedef std::vector<Tensor3D> Filters; // (out_channels, in_channels, filter_height, filter_width)

    int in_channels;
    int out_channels;
    int filter_height;
    int filter_width;
    int stride;
    int padding;
    Filters filters;
    std::vector<Type> biases;

    ConvolutionalWeights(const ConvolutionLayer<Type>& layer);
};

#include "ConvolutionalWeights.tpp"

#endif //CONVOLUTIONALWEIGHTS_H
