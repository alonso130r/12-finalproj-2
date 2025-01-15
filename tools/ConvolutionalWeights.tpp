//
// Created by adamk on 2025-01-14.
//

#include "ConvolutionalWeights.h"
#include "../layers/ConvolutionLayer.h"
#include "../layers/ConvolutionLayer.h"

template <typename Type>
ConvolutionalWeights<Type>::ConvolutionalWeights(ConvolutionLayer<Type> const& layer) {
    in_channels = layer.in_channels;
    out_channels = layer.out_channels;
    filter_height = layer.filter_height;
    filter_width = layer.filter_width;
    stride = layer.stride;
    padding = layer.padding;
    filters = layer.filters;
    biases = layer.biases;
}

template <typename Type>
WeightStructType ConvolutionalWeights<Type>::getType() const {
    return WeightStructType::ConvolutionalWeights;
}

template<typename Type>
void ConvolutionalWeights<Type>::serialize(std::ofstream &out) const {
    out.write(reinterpret_cast<const char*>(&in_channels), sizeof(in_channels));
    out.write(reinterpret_cast<const char*>(&out_channels), sizeof(out_channels));
    out.write(reinterpret_cast<const char*>(&filter_height), sizeof(filter_height));
    out.write(reinterpret_cast<const char*>(&filter_width), sizeof(filter_width));
    out.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    out.write(reinterpret_cast<const char*>(&padding), sizeof(padding));
    out.write(reinterpret_cast<const char*>(&filters), sizeof(filters));
    out.write(reinterpret_cast<const char*>(&biases), sizeof(biases));
}

template<typename Type>
std::shared_ptr<ConvolutionLayer<Type>> ConvolutionalWeights<Type>::deserialize(std::ifstream &in) {
    int in_channels_t;
    int out_channels_t;
    int filter_height_t;
    int filter_width_t;
    int stride_t;
    int padding_t;
    Filters filters_t;
    std::vector<Type> biases_t;

    in.read(reinterpret_cast<char*>(&in_channels_t), sizeof(in_channels_t));
    in.read(reinterpret_cast<char*>(&out_channels_t), sizeof(out_channels_t));
    in.read(reinterpret_cast<char*>(&filter_height_t), sizeof(filter_height_t));
    in.read(reinterpret_cast<char*>(&filter_width_t), sizeof(filter_width_t));
    in.read(reinterpret_cast<char*>(&stride_t), sizeof(stride_t));
    in.read(reinterpret_cast<char*>(&padding_t), sizeof(padding_t));
    in.read(reinterpret_cast<char*>(&filters_t), sizeof(filters_t));
    in.read(reinterpret_cast<char*>(&biases_t), sizeof(biases_t));


    auto temp = std::make_shared<ConvolutionLayer<Type>>(in_channels_t, out_channels_t, filter_height_t, filter_width_t, stride_t, padding_t);
    temp->filters = filters_t;
    temp->biases = biases_t;
    return temp;
}