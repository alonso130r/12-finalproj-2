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

/*
 * forward pass through the convolutional layer
 */
std::shared_ptr<Tensor> ConvolutionLayer::forward(const std::shared_ptr<Tensor>& input) {
    int batch_size = input->data.size();
    if(batch_size == 0) {
        throw std::invalid_argument("Input batch size is zero.");
    }

    int in_channels_input = input->data[0].size();
    if(in_channels_input != in_channels) {
        throw std::invalid_argument("Input channels do not match layer's in_channels.");
    }

    int input_height = input->data[0][0].size();
    int input_width = input->data[0][0][0].size();

    Tensor4D padded_input(batch_size, Tensor3D(in_channels, std::vector<std::vector<double>>(input_height + 2 * padding, std::vector<double>(input_width + 2 * padding, 0.0))));

    for(int n = 0; n < batch_size; ++n) {
        for(int c = 0; c < in_channels; ++c) {
            for(int h = 0; h < input_height; ++h) {
                for(int w = 0; w < input_width; ++w) {
                    padded_input[n][c][h + padding][w + padding] = input->data[n][c][h][w];
                }
            }
        }
    }

    int padded_height = padded_input[0][0].size();
    int padded_width = padded_input[0][0][0].size();

    int out_height = (padded_height - filter_height) / stride + 1;
    int out_width = (padded_width - filter_width) / stride + 1;

    auto output = std::make_shared<Tensor>(batch_size, out_channels, out_height, out_width, 0.0);

    // initialize pre_activation cache
    pre_activation = Tensor4D(batch_size, Tensor3D(out_channels, std::vector<std::vector<double>>(out_height, std::vector<double>(out_width, 0.0))));

    // perform convolution for each sample in the batch
    for(int n = 0; n < batch_size; ++n) {
        for(int f = 0; f < out_channels; ++f) {
            for(int h = 0; h < out_height; ++h) {
                for(int w = 0; w < out_width; ++w) {
                    double sum = 0.0;
                    for(int c = 0; c < in_channels; ++c) {
                        for(int kh = 0; kh < filter_height; ++kh) {
                            for(int kw = 0; kw < filter_width; ++kw) {
                                int in_h = h * stride + kh;
                                int in_w = w * stride + kw;
                                sum += padded_input[n][c][in_h][in_w] * filters[f][c][kh][kw];
                            }
                        }
                    }
                    sum += biases[f]; // bias
                    pre_activation[n][f][h][w] = sum; // cache pre-activation
                    // relu
                    output->data[n][f][h][w] = sum > 0 ? sum : 0.0;
                }
            }
        }
    }

    // set creator operation(for computation graph)
     output->creator = ...;

    return output;
}

/*
 * backward pass through the convolutional layer
 */
void ConvolutionLayer::backward(const std::shared_ptr<Tensor>& dOut) {
    int batch_size = dOut->data.size();
    if (batch_size == 0) {
        throw std::invalid_argument("dOut batch size is zero.");
    }

    int out_channels_dOut = dOut->data[0].size();
    int out_height = dOut->data[0][0].size();
    int out_width = dOut->data[0][0][0].size();

    int input_height = stride * (out_height - 1) + filter_height - 2 * padding;
    int input_width = stride * (out_width - 1) + filter_width - 2 * padding;

    // initialize gradients
    dFilters.assign(out_channels, Tensor3D(in_channels,
                                           std::vector<std::vector<double>> (filter_height, std::vector<double>(filter_width, 0.0))));
    dBiases.assign(out_channels, 0.0);

    Tensor4D dInput(batch_size, Tensor3D(in_channels,
                                         std::vector<std::vector <double>> (input_height + 2 * padding,
                                                 std::vector<double>(input_width + 2 * padding, 0.0))));

    // perform backward pass for each sample in the batch
    for (int n = 0; n < batch_size; ++n) {
        for (int f = 0; f < out_channels; ++f) {
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    // compute derivative of relu using cached pre-activation
                    double pre_act = pre_activation[n][f][h][w];
                    double grad_activation = pre_act > 0 ? dOut->grad[n][f][h][w] : 0.0;

                    // accumulate bias gradients
                    dBiases[f] += grad_activation;

                    for (int c = 0; c < in_channels; ++c) {
                        for (int kh = 0; kh < filter_height; ++kh) {
                            for (int kw = 0; kw < filter_width; ++kw) {
                                int in_h = h * stride + kh;
                                int in_w = w * stride + kw;
                                // accumulate filter gradients
                                dFilters[f][c][kh][kw] +=
                                        pre_act > 0 ? dOut->grad[n][f][h][w] * dInput[n][c][in_h][in_w] : 0.0;
                                // accumulate input gradients
                                dInput[n][c][in_h][in_w] += filters[f][c][kh][kw] * grad_activation;
                            }
                        }
                    }
                }
            }
        }
    }

    // remove padding from dInput if necessary
    if (padding > 0) {
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < input_height; ++h) {
                    for (int w = 0; w < input_width; ++w) {
                        dOut->grad[n][c][h][w] = dInput[n][c][h + padding][w + padding];
                    }
                }
            }
        }
    } else {
        dOut->grad = dInput;
    }
}

void ConvolutionLayer::setFilters(const Filters& new_filters) {
    if(new_filters.size() != out_channels) {
        throw std::invalid_argument("Number of filters does not match out_channels.");
    }
    for(int f = 0; f < out_channels; ++f) {
        if(new_filters[f].size() != in_channels ||
           new_filters[f][0].size() != filter_height ||
           new_filters[f][0][0].size() != filter_width) {
            throw std::invalid_argument("Filter dimensions do not match.");
        }
    }
    filters = new_filters;
}

void ConvolutionLayer::setBiases(const std::vector<double>& new_biases) {
    if(new_biases.size() != out_channels) {
        throw std::invalid_argument("Number of biases does not match out_channels.");
    }
    biases = new_biases;
}

void ConvolutionLayer::zeroGrad() {
    dFilters.assign(out_channels, Tensor3D(in_channels,
                                           std::vector<std::vector<double>> (filter_height, std::vector<double>(filter_width, 0.0))));
    dBiases.assign(out_channels, 0.0);
}

size_t ConvolutionLayer::getNumParams() const {
    size_t out_channels = filters.size();
    if(out_channels == 0) return 0;
    size_t in_channels  = filters[0].size();
    size_t filter_height = filters[0][0].size();
    size_t filter_width  = filters[0][0][0].size();

    // param for filters
    size_t filterParams = out_channels * in_channels * filter_height * filter_width;
    // param for biases
    size_t biasParams = biases.size();

    return filterParams + biasParams;
}