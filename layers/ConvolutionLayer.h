// ConvolutionLayer.h
#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include "../tools/Tensor.h"
#include "Layer.h"
#include <iostream>
#include <fstream>

template <typename Type>
class ConvolutionLayer : public Layer<Type> {
public:
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

    Filters dFilters;
    std::vector<Type> dBiases;

    // cached pre-activation outputs for backpropagation
    Tensor4D pre_activation;

    ConvolutionLayer(int in_channels, int out_channels, int filter_height, int filter_width, int stride = 1, int padding = 0);

    void initializeFilters();

    std::shared_ptr<Tensor<Type>> forward(const std::shared_ptr<Tensor<Type>>& input);

    Tensor4D backward(const std::shared_ptr<Tensor<Type>>& dOut);

    void zeroGrad();

    [[nodiscard]] ssize_t getNumParams() const;

    void setFilters(const Filters& new_filters);
    void setBiases(const std::vector<Type>& new_biases);
    std::shared_ptr<WeightStruct<Type>> saveWeights(const std::string location) override;
};

#include "ConvolutionLayer.tpp"

#endif // CONVOLUTION_LAYER_H
