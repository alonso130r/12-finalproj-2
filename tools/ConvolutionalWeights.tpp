//
// Created by adamk on 2025-01-14.
//

#include "ConvolutionalWeights.h"

template <typename Type>
ConvolutionalWeights<Type>::ConvolutionalWeights(ConvolutionLayer<Type> const& layer) {
    in_channels = layer->in_channels;
    out_channels = layer->out_channels;
    filter_height = layer->filter_height;
    filter_width = layer->filter_width;
    stride = layer->stride;
    padding = layer->padding;
    filters = layer->filters;
    biases = layer->biases;
}