//
// Created by Vijay Goyal on 2025-01-06.
//

#include "ConvolutionLayer.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <random>
#include <omp.h>


template <typename Type>
ConvolutionLayer<Type>::ConvolutionLayer(int in_channels, int out_channels, int filter_height, int filter_width, int stride,
                                   int padding) : in_channels(in_channels), out_channels(out_channels),
                                                  filter_height(filter_height), filter_width(filter_width),
                                                  stride(stride), padding(padding) {
    initializeFilters();
}

/*
 * Initialize the filters and biases with random values using He initialization
 */
template <typename Type>
void ConvolutionLayer<Type>::initializeFilters() {
    // calculate fan in and standard deviation
    int fan_in = filter_height * filter_width * in_channels;
    Type std_dev = sqrt(static_cast<Type>(8.0) / static_cast<Type>(fan_in));

    // initialize random generators (mersenne twister engine)
    std::random_device rd;

    // resize filters and biases
    filters.resize(out_channels, Tensor3D(in_channels, std::vector<std::vector<Type>>(filter_height, std::vector<Type>(filter_width, static_cast<Type>(0.0)))));
    biases.resize(out_channels, static_cast<Type>(0.0));

    /// thread-safe He initialization
    #pragma omp parallel
    {
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::normal_distribution<Type> dist(static_cast<Type>(0.0), std_dev);

        #pragma omp for
        for (int f = 0; f < out_channels; ++f) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < filter_height; ++h) {
                    for (int w = 0; w < filter_width; ++w) {
                        filters[f][c][h][w] = dist(gen);
                    }
                }
            }
            biases[f] = dist(gen) * static_cast<Type>(0.1);
        }
    }

    // initialize gradients to zero
    dFilters.resize(out_channels, Tensor3D(in_channels, std::vector<std::vector<Type>>(filter_height, std::vector<Type>(filter_width, static_cast<Type>(0.0)))));
    dBiases.resize(out_channels, static_cast<Type>(0.0));
}

/*
 * forward pass through the convolutional layer
 */
template <typename Type>
std::shared_ptr<Tensor<Type>> ConvolutionLayer<Type>::forward(const std::shared_ptr<Tensor<Type>>& input) {
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

    Tensor4D padded_input(batch_size, Tensor3D(in_channels, std::vector<std::vector<Type>>(input_height + 2 * padding, std::vector<Type>(input_width + 2 * padding, static_cast<Type>(0.0)))));

    #pragma omp parallel for
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

    auto output = std::make_shared<Tensor<Type>>(batch_size, out_channels, out_height, out_width, static_cast<Type>(0.0));

    // initialize pre_activation cache
    pre_activation = Tensor4D(batch_size, Tensor3D(out_channels, std::vector<std::vector<Type>>(out_height, std::vector<Type>(out_width, static_cast<Type>(0.0)))));

    // perform convolution for each sample in the batch
    #pragma omp parallel for 
    for(int n = 0; n < batch_size; ++n) {
        for(int f = 0; f < out_channels; ++f) {
            for(int h = 0; h < out_height; ++h) {
                for(int w = 0; w < out_width; ++w) {
                    Type sum = static_cast<Type>(0.0);
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
                    pre_activation[n][f][h][w] = static_cast<Type>(sum); // cache pre-activation
                    // relu
                    output->data[n][f][h][w] = sum > static_cast<Type>(0) ? sum : static_cast<Type>(0.0);
                }
            }
        }
    }

    return output;
}

/*
 * backward pass through the convolutional layer
 */
template <typename Type>
std::vector<std::vector<std::vector<std::vector<Type>>>> ConvolutionLayer<Type>::backward(const std::shared_ptr<Tensor<Type>>& dOut) {
    int batch_size = dOut->data.size();
    if (batch_size == 0) {
        throw std::invalid_argument("dOut batch size is zero.");
    }

    // get dimensions
    int out_channels_dOut = dOut->data[0].size();
    int out_height = dOut->data[0][0].size();
    int out_width = dOut->data[0][0][0].size();
    int padded_height = out_height * stride + filter_height - stride;
    int padded_width = out_width * stride + filter_width - stride;

    // prepare dInput
    Tensor4D dPaddedInput(batch_size, Tensor3D(in_channels, 
        std::vector<std::vector<Type>>(padded_height, std::vector<Type>(padded_width, static_cast<Type>(0.0)))));

    // zero gradients
    #pragma omp parallel for
    for (int f = 0; f < out_channels; ++f) {
        dBiases[f] = static_cast<Type>(0.0);
        for (int c = 0; c < in_channels; ++c) {
            for (int kh = 0; kh < filter_height; ++kh) {
                for (int kw = 0; kw < filter_width; ++kw) {
                    dFilters[f][c][kh][kw] = static_cast<Type>(0.0);
                }
            }
        }
    }

    // accumulate gradients
    #pragma omp parallel
    {
        auto dFiltersLocal = dFilters; 
        auto dBiasesLocal = dBiases;

        #pragma omp for
        for (int n = 0; n < batch_size; ++n) {
            for (int f = 0; f < out_channels; ++f) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        Type grad_val = dOut->data[n][f][oh][ow]; // if activation grad was 1, else multiply
                        dBiasesLocal[f] += grad_val;
                        for (int c = 0; c < in_channels; ++c) {
                            for (int kh = 0; kh < filter_height; ++kh) {
                                for (int kw = 0; kw < filter_width; ++kw) {
                                    int ph = oh * stride + kh;
                                    int pw = ow * stride + kw;
                                    dFiltersLocal[f][c][kh][kw] += /* input[n][c][ph][pw] */ 
                                                                    // you need stored padded input or pre_activation
                                                                    grad_val;
                                    // compute dPaddedInput for backprop
                                    dPaddedInput[n][c][ph][pw] += filters[f][c][kh][kw] * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        // reduce local accumulations
        #pragma omp critical
        {
            for (int f = 0; f < out_channels; ++f) {
                dBiases[f] += dBiasesLocal[f];
                for (int c = 0; c < in_channels; ++c) {
                    for (int kh = 0; kh < filter_height; ++kh) {
                        for (int kw = 0; kw < filter_width; ++kw) {
                            dFilters[f][c][kh][kw] += dFiltersLocal[f][c][kh][kw];
                        }
                    }
                }
            }
        }
    }

    // remove any padding from dPaddedInput
    std::vector<std::vector<std::vector<std::vector<Type>>>> dInput(batch_size,
        std::vector<std::vector<std::vector<Type>>>(in_channels,
        std::vector<std::vector<Type>>(padded_height - 2 * padding,
        std::vector<Type>(padded_width - 2 * padding, static_cast<Type>(0.0)))));

    #pragma omp parallel for
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < static_cast<int>(dInput[n][c].size()); ++h) {
                for (int w = 0; w < static_cast<int>(dInput[n][c][h].size()); ++w) {
                    dInput[n][c][h][w] = dPaddedInput[n][c][h + padding][w + padding];
                }
            }
        }
    }
    return dInput;
}

template <typename Type>
void ConvolutionLayer<Type>::setFilters(const Filters& new_filters) {
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

template <typename Type>
void ConvolutionLayer<Type>::setBiases(const std::vector<Type>& new_biases) {
    if(new_biases.size() != out_channels) {
        throw std::invalid_argument("Number of biases does not match out_channels.");
    }
    biases = new_biases;
}

template <typename Type>
void ConvolutionLayer<Type>::zeroGrad() {
    dFilters.assign(out_channels, Tensor3D(in_channels,
                                           std::vector<std::vector<Type>> (filter_height, std::vector<Type>(filter_width, static_cast<Type>(0.0)))));
    dBiases.assign(out_channels, static_cast<Type>(0.0));
}

template <typename Type>
ssize_t ConvolutionLayer<Type>::getNumParams() const {
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

#include "../tools/ConvolutionalWeights.h"
template <typename Type>
std::shared_ptr<WeightStruct<Type>> ConvolutionLayer<Type>::saveWeights() {
    return std::make_shared<ConvolutionalWeights<Type>>(*this);
}