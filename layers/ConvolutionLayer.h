// ConvolutionLayer.h
#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include "Tensor.h"

typedef std::vector<std::vector<std::vector<double>>> Tensor3D; // (channels, height, width)
typedef std::vector<std::vector<std::vector<std::vector<double>>>> Tensor4D; // (batch_size, channels, height, width)
typedef std::vector<Tensor3D> Filters; // (out_channels, in_channels, filter_height, filter_width)

class ConvolutionLayer {
public:
    int in_channels;
    int out_channels;
    int filter_height;
    int filter_width;
    int stride;
    int padding;
    Filters filters;
    std::vector<double> biases;

    Filters dFilters;
    std::vector<double> dBiases;

    // cached pre-activation outputs for backpropagation
    Tensor4D pre_activation;

    ConvolutionLayer(int in_channels, int out_channels, int filter_height, int filter_width, int stride = 1, int padding = 0);

    void initializeFilters();

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input);

    void backward(const std::shared_ptr<Tensor>& dOut);

    void zeroGrad();

    void setFilters(const Filters& new_filters);
    void setBiases(const std::vector<double>& new_biases);
};

#endif // CONVOLUTION_LAYER_H
