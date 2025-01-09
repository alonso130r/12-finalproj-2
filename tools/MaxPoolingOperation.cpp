//
// Created by Vijay Goyal on 2025-01-08.
//

#include "MaxPoolingOperation.h"
#include <algorithm>
#include <limits>

MaxPoolingOperation::MaxPoolingOperation(int pool_height, int pool_width, int stride, int padding) :
        pool_height(pool_height), pool_width(pool_width), stride(stride), padding(padding)
        {}

Tensor MaxPoolingOperation::forward(const Tensor &input) {
    int batch_size = input.shape[0];
    int channels = input.data[0].size();
    int input_height = input.data[0][0].size();
    int input_width = input.data[0][0][0].size();

    // output dimensions
    int out_height = (input_height + 2 * padding - pool_height) / stride + 1;
    int out_width = (input_width + 2 * padding - pool_width) / stride + 1;

    // output tensor
    Tensor output(batch_size, channels, out_height, out_width, 0.0);

    // max_indices
    max_indices.resize(batch_size, std::vector<std::vector<std::vector<std::pair<int, int>>>>(
            channels, std::vector<std::vector<std::pair<int, int>>>(out_height, std::vector<std::pair<int, int>>(out_width, {0, 0}))));

    for(int n = 0; n < batch_size; ++n) {
        for(int c = 0; c < channels; ++c) {
            for(int h = 0; h < out_height; ++h) {
                for(int w = 0; w < out_width; ++w) {
                    // define pooling window boundaries
                    int h_start = h * stride - padding;
                    int w_start = w * stride - padding;
                    int h_end = std::min(h_start + pool_height, input_height);
                    int w_end = std::min(w_start + pool_width, input_width);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);

                    double max_val = -std::numeric_limits<double>::infinity();
                    std::pair<int, int> max_pos = {h_start, w_start};

                    // iterate over the pooling window to find the max value and its position
                    for(int ph = h_start; ph < h_end; ++ph) {
                        for(int pw = w_start; pw < w_end; ++pw) {
                            if(input.data[n][c][ph][pw] > max_val) {
                                max_val = input.data[n][c][ph][pw];
                                max_pos = {ph, pw};
                            }
                        }
                    }

                    output.data[n][c][h][w] = max_val;
                    max_indices[n][c][h][w] = max_pos;
                }
            }
        }
    }

    return output;
}

std::shared_ptr<Tensor> MaxPoolingOperation::backward(const std::shared_ptr<Tensor>& output_grad) {
    if(this->inputs.empty()) {
        throw std::runtime_error("MaxPoolingOperation has no input tensors stored. Perform forward pass before backward.");
    }

    // assume single input tensor
    if(this->inputs.size() != 1) {
        throw std::invalid_argument("MaxPoolingOperation expects exactly one input tensor.");
    }

    auto input_tensor = this->inputs[0];

    int batch_size = output_grad->data.size();
    int channels = output_grad->data[0].size();
    int out_height = output_grad->data[0][0].size();
    int out_width = output_grad->data[0][0][0].size();

    // calculate input dimensions based on stride and padding
    int input_height_calc = stride * (out_height - 1) + pool_height - 2 * padding;
    int input_width_calc = stride * (out_width - 1) + pool_width - 2 * padding;

    // initialize dInput_padded with zeros
    Tensor4D dInput_padded(batch_size, Tensor3D(channels, std::vector<std::vector<double>>(
            input_height_calc + 2 * padding, std::vector<double>(input_width_calc + 2 * padding, 0.0))));

    // gradients based on max_indices
    for(int n = 0; n < batch_size; ++n) {
        for(int c = 0; c < channels; ++c) {
            for(int h = 0; h < out_height; ++h) {
                for(int w = 0; w < out_width; ++w) {
                    // position of the max value from the forward pass
                    std::pair<int, int> max_pos = max_indices[n][c][h][w];
                    int ph = max_pos.first;
                    int pw = max_pos.second;

                    // accumulate the gradient
                    dInput_padded[n][c][ph][pw] += output_grad->grad[n][c][h][w];
                }
            }
        }
    }

    // remove padding from dInput if necessary
    if(padding > 0) {
        Tensor4D dInput_unpadded(batch_size, Tensor3D(channels, std::vector<std::vector<double>>(
                input_height_calc, std::vector<double>(input_width_calc, 0.0))));

        for(int n = 0; n < batch_size; ++n) {
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < input_height_calc; ++h) {
                    for(int w = 0; w < input_width_calc; ++w) {
                        dInput_unpadded[n][c][h][w] = dInput_padded[n][c][h + padding][w + padding];
                    }
                }
            }
        }

        // assign dInputUnpadded to the input tensor's grad
        for(int n = 0; n < batch_size; ++n) {
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < input_height_calc; ++h) {
                    for(int w = 0; w < input_width_calc; ++w) {
                        input_tensor->grad[n][c][h][w] += dInput_unpadded[n][c][h][w];
                    }
                }
            }
        }

        auto dInput_tensor = std::make_shared<Tensor>();
        dInput_tensor->data = dInput_unpadded;
        dInput_tensor->grad = Tensor4D(dInput_unpadded.size(), Tensor3D(dInput_unpadded[0].size(),
                                                                        std::vector<std::vector<double>>(dInput_unpadded[0][0].size(),
                                                                                std::vector<double>(dInput_unpadded[0][0][0].size(), 0.0))));

        return dInput_tensor;
    }
    else {
        // assign dInput_padded to the input tensor's grad
        for(int n = 0; n < batch_size; ++n) {
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < input_height_calc; ++h) {
                    for(int w = 0; w < input_width_calc; ++w) {
                        input_tensor->grad[n][c][h][w] += dInput_padded[n][c][h][w];
                    }
                }
            }
        }

        auto dInput_tensor = std::make_shared<Tensor>();
        dInput_tensor->data = dInput_padded;
        dInput_tensor->grad = Tensor4D(dInput_padded.size(), Tensor3D(dInput_padded[0].size(),
                                                                      std::vector<std::vector<double>>(dInput_padded[0][0].size(),
                                                                              std::vector<double>(dInput_padded[0][0][0].size(), 0.0))));

        return dInput_tensor;
    }
}