//
// Created by Vijay Goyal on 2025-01-06.
//

#include "ConvolutionLayer.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

ConvolutionLayer::ConvolutionLayer(int in_channels, int out_channels, int filter_height, int filter_width, int stride,
                                   int padding) : channel_in(in_channels), channel_out(out_channels),
                                                  filter_height(filter_height), filter_width(filter_width),
                                                  stride(stride), padding(padding) {
    initializeFilters();
}

/*
 * Initialize the filters and biases with random values using He initialization
 */
void ConvolutionLayer::initializeFilters() {
    // calculate fan in and standard deviation
    int fan_in = filter_height * filter_width * channel_in;
    double std_dev = sqrt(2.0 / static_cast<double>(fan_in));

    // initialize random generators (mersenne twister engine)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std_dev);

    // resize filters and biases
    filters.resize(out_channels, Tensor3D(in_channels, std::vector<std::vector<double>>(filter_height, std::vector<double>(filter_width, 0.0))));
    biases.resize(out_channels, 0.0);

    // initialize filters with He initialization
    for (int f = 0; f < out_channels; ++f) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < filter_height; ++h) {
                for (int w = 0; w < filter_width; ++w) {
                    filters[f][c][h][w] = dist(gen); // sampled from N(0, std_dev^2)
                }
            }
        }
        biases[f] = 0.0; // initialize biases to zero
    }

    // initialize gradients to zero
    dFilters.resize(out_channels, Tensor3D(in_channels, std::vector<std::vector<double>>(filter_height, std::vector<double>(filter_width, 0.0))));
    dBiases.resize(out_channels, 0.0);
}

// inline relu function
inline double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

// inline relu derivative function
inline double reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/*
 * helper to pad a single input
 */
Tensor3D ConvolutionLayer::padInputSingle(const Tensor3D &input) const {
    if (padding == 0) return input;

    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();
    int new_height = height + 2 * padding;
    int new_width = width + 2 * padding;

    Tensor3D padded(channels, std::vector<std::vector<double>>(new_height, std::vector<double>(new_width, 0.0)));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                padded[c][h + padding][w + padding] = input[c][h][w];
            }
        }
    }

    return padded;
}

/*
 * forward pass through the convolutional layer
 */
Tensor4D ConvolutionLayer::forward(const Tensor4D &input) {
    int batch_size = input.size();
    if (batch_size == 0) throw std::invalid_argument("Input tensor is empty");

    int in_channels_input = input[0].size();
    if (in_channels_input != channel_in) throw std::invalid_argument("Input tensor has incorrect number of channels");

    cached_input = input;

    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    Tensor4D padded_input(batch_size, Tensor3D(channel_in, std::vector<std::vector<double>>(input_height + 2 * padding, std::vector<double>(input_width + 2 * padding, 0.0))));

    for (int n = 0; n < batch_size; ++n) {
        padded_input[n] = padInputSingle(input[n]);
    }

    int padded_height = padded_input[0][0].size();
    int padded_width = padded_input[0][0][0].size();

    int output_height = (padded_height - filter_height) / stride + 1;
    int output_width = (padded_width - filter_width) / stride + 1;

    // perform convolution for each sample in the batch
    for (int n = 0; n < batch_size; ++n) {
        for (int f = 0; f < out_channels; ++f) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        double sum = 0.0;
                        for (int kh = 0; kh < filter_height; ++kh) {
                            for (int kw = 0; kw < filter_width; ++kw) {
                                int in_h = h * stride + kh;
                                int in_w = w * stride + kw;
                                sum += padded_input[n][c][in_h][in_w] * filters[f][c][kh][kw];
                            }
                        }
                        sum += biases[f]; // add bias
                        output[n][f][h][w] += relu(sum); // relu activation
                    }
                }
            }
        }
    }

    return output;
}