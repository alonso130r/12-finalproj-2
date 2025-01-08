//
// Created by Vijay Goyal on 2025-01-06.
//
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

#ifndef INC_12_FINALPROJ_2_CONVOLUTIONLAYER_H
#define INC_12_FINALPROJ_2_CONVOLUTIONLAYER_H

typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
typedef std::vector<std::vector<std::vector<std::vector<double>>>> Tensor4D; // (batch_size, channels, height, width)
typedef std::vector<Tensor3D> Filters;

class ConvolutionLayer {
private:
    // layer parameters
    int channel_in;
    int channel_out;
    int filter_height;
    int filter_width;
    int stride;
    int padding;

    // parameters
    Filters filters;
    std::vector<double> biases;

    // gradients
    Filters dFilters;
    std::vector<double> dBiases;

    // cached input for backprop
    Tensor4D cached_input;

    // zero-padding helper
    Tensor3D padInputSingle(const Tensor3D &input) const;

public:
    ConvolutionLayer(int in_channels, int out_channels, int filter_height, int filter_width, int stride = 1, int padding = 0);

    void initializeFilters();

    Tensor4D forward(const Tensor4D &input);

    Tensor4D backward(const Tensor4D &dOut);

    // gradient getters
    Filters getDFilters() const;
    std::vector<double> getDBiases() const;

    // setters for filter/bias
    void setFilters(const Filters &new_filters);
    void setBiases(const std::vector<double> &new_biases);
};


#endif //INC_12_FINALPROJ_2_CONVOLUTIONLAYER_H
