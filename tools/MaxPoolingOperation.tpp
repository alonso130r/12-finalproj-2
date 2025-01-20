//
// Created by Vijay Goyal on 2025-01-08.
//

#include "MaxPoolingOperation.h"
#include <algorithm>
#include <limits>
#include <omp.h>

template <typename Type>
MaxPoolingOperation<Type>::MaxPoolingOperation(int pool_height, int pool_width, int stride, int padding) :
        pool_height(pool_height), pool_width(pool_width), stride(stride), padding(padding) {}

 
template <typename Type>
std::shared_ptr<Tensor<Type>> MaxPoolingOperation<Type>::forward(const std::shared_ptr<Tensor<Type>> &input) {
    int batch_size = input->data.size();
    int channels = input->data[0].size();
    int input_height = input->data[0][0].size();
    int input_width = input->data[0][0][0].size();

    this->inputs = input;

    // output dimensions
    int out_height = (input_height + 2 * padding - pool_height) / stride + 1;
    int out_width = (input_width + 2 * padding - pool_width) / stride + 1;

    // output tensor
    Tensor<Type> output(batch_size, channels, out_height, out_width, static_cast<Type>(0.0));

    // max_indices
    max_indices.resize(batch_size, std::vector<std::vector<std::vector<std::pair<int, int>>>>(
            channels, std::vector<std::vector<std::pair<int, int>>>(out_height, std::vector<std::pair<int, int>>(out_width, {0, 0}))));

    // Parallelize over the batch and channels dimensions
    #pragma omp parallel for collapse(2)
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

                    Type max_val = -std::numeric_limits<Type>::infinity();
                    std::pair<int, int> max_pos = {h_start, w_start};

                    // iterate over the pooling window to find the max value and its position
                    for(int ph = h_start; ph < h_end; ++ph) {
                        for(int pw = w_start; pw < w_end; ++pw) {
                            if(input->data[n][c][ph][pw] > max_val) {
                                max_val = input->data[n][c][ph][pw];
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

    return std::make_shared<Tensor<Type>>(output);
}

template <typename Type>
std::shared_ptr<Tensor<Type>> MaxPoolingOperation<Type>::backward(const std::shared_ptr<Tensor<Type>>& output_grad) {
    if(this->inputs->data.empty()) {
        throw std::runtime_error("MaxPoolingOperation has no input tensors stored. Perform forward pass before backward.");
    }

    auto input_tensor = this->inputs;

    if (output_grad == nullptr) {
        std::cerr << "Error: output_grad is null in MaxPoolingOperation::backward." << std::endl;
        throw std::invalid_argument("output_grad is null.");
    }

    if (output_grad->grad.empty()) {
        std::cerr << "Error: output_grad->grad is empty in MaxPoolingOperation::backward." << std::endl;
        throw std::invalid_argument("output_grad->grad is empty.");
    }

    int batch_size = output_grad->data.size();
    if (batch_size == 0) {
        std::cerr << "Error: output_grad->data has zero batch size." << std::endl;
        throw std::invalid_argument("output_grad->data has zero batch size.");
    }

    int channels = output_grad->data[0].size();
    int out_height = output_grad->data[0][0].size();
    int out_width = output_grad->data[0][0][0].size();

    // Validate grad dimensions
    if (output_grad->grad.size() != static_cast<size_t>(batch_size)) {
        std::cerr << "Error: output_grad->grad size (" << output_grad->grad.size()
                  << ") does not match batch_size (" << batch_size << ")." << std::endl;
        throw std::invalid_argument("output_grad->grad size does not match batch_size.");
    }

    for(int n = 0; n < batch_size; ++n) {
        if (output_grad->grad[n].size() != static_cast<size_t>(channels)) {
            std::cerr << "Error: output_grad->grad[" << n << "].size() ("
                      << output_grad->grad[n].size() << ") does not match channels ("
                      << channels << ")." << std::endl;
            throw std::invalid_argument("output_grad->grad at sample does not match channels.");
        }
        for(int c = 0; c < channels; ++c) {
            if (output_grad->grad[n][c].size() != static_cast<size_t>(out_height)) {
                std::cerr << "Error: output_grad->grad[" << n << "][" << c
                          << "].size() (" << output_grad->grad[n][c].size()
                          << ") does not match out_height (" << out_height << ")." << std::endl;
                throw std::invalid_argument("output_grad->grad at sample feature has incorrect height.");
            }
            for(int h = 0; h < out_height; ++h) {
                if (output_grad->grad[n][c][h].size() != static_cast<size_t>(out_width)) {
                    std::cerr << "Error: output_grad->grad[" << n << "][" << c << "]["
                              << h << "].size() (" << output_grad->grad[n][c][h].size()
                              << ") does not match out_width (" << out_width << ")." << std::endl;
                    throw std::invalid_argument("output_grad->grad at sample feature has incorrect width.");
                }
            }
        }
    }

    // calculate input dimensions based on stride and padding
    int input_height_calc = stride * (out_height - 1) + pool_height - 2 * padding;
    int input_width_calc = stride * (out_width - 1) + pool_width - 2 * padding;

    // Validate max_indices dimensions
    if (max_indices.size() != static_cast<size_t>(batch_size)) {
        std::cerr << "Error: max_indices size (" << max_indices.size()
                << ") does not match batch_size (" << batch_size << ")." << std::endl;
        throw std::invalid_argument("max_indices size does not match batch_size.");
    }

    for(int n = 0; n < batch_size; ++n) {
        if (max_indices[n].size() != static_cast<size_t>(channels)) {
            std::cerr << "Error: max_indices[" << n << "].size() ("
                    << max_indices[n].size() << ") does not match channels ("
                    << channels << ")." << std::endl;
            throw std::invalid_argument("max_indices at sample does not match channels.");
        }
        for(int c = 0; c < channels; ++c) {
            if (max_indices[n][c].size() != static_cast<size_t>(out_height)) {
                std::cerr << "Error: max_indices[" << n << "][" << c
                        << "].size() (" << max_indices[n][c].size()
                        << ") does not match out_height (" << out_height << ")." << std::endl;
                throw std::invalid_argument("max_indices at sample feature has incorrect height.");
            }
            for(int h = 0; h < out_height; ++h) {
                if (max_indices[n][c][h].size() != static_cast<size_t>(out_width)) {
                    std::cerr << "Error: max_indices[" << n << "][" << c << "]["
                            << h << "].size() (" << max_indices[n][c][h].size()
                            << ") does not match out_width (" << out_width << ")." << std::endl;
                    throw std::invalid_argument("max_indices at sample feature has incorrect width.");
                }
            }
        }
    }

    // Debug: Verify max_indices content
    for(int n = 0; n < batch_size; ++n) {
        for(int c = 0; c < channels; ++c) {
            for(int h = 0; h < out_height; ++h) {
                for(int w = 0; w < out_width; ++w) {
                    auto& pos = max_indices[n][c][h][w];
                    if(pos.first < 0 || pos.first >= input_height_calc || pos.second < 0 || pos.second >= input_width_calc) {
                        std::cerr << "Error: Invalid max_indices[" << n << "][" << c << "][" << h << "][" << w << "] = ("
                                  << pos.first << ", " << pos.second << ")." << std::endl;
                        throw std::out_of_range("max_indices contains out-of-bound positions.");
                    }
                }
            }
        }
    }

    // initialize dInput_padded with zeros
    Tensor4D dInput_padded(batch_size, Tensor3D(channels, std::vector<std::vector<Type>>(
            input_height_calc + 2 * padding, std::vector<Type>(input_width_calc + 2 * padding, static_cast<Type>(0.0)))));

    // Parallelize the gradient assignment
    #pragma omp parallel for collapse(2)
    for(int n = 0; n < batch_size; ++n) {
        for(int c = 0; c < channels; ++c) {
            for(int h = 0; h < out_height; ++h) {
                for(int w = 0; w < out_width; ++w) {
                    // position of the max value from the forward pass
                    std::pair<int, int> max_pos = max_indices[n][c][h][w];
                    int ph = max_pos.first;
                    int pw = max_pos.second;

                    // Accumulate the gradient
                    #pragma omp atomic
                    dInput_padded[n][c][ph][pw] += output_grad->grad[n][c][h][w];
                }
            }
        }
    }

    // Remove padding from dInput if necessary
    if(padding > 0) {
        Tensor4D dInput_unpadded(batch_size, Tensor3D(channels, std::vector<std::vector<Type>>(
                input_height_calc, std::vector<Type>(input_width_calc, static_cast<Type>(0.0)))));

        // Parallelize the unpadding process
        #pragma omp parallel for collapse(2)
        for(int n = 0; n < batch_size; ++n) {
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < input_height_calc; ++h) {
                    for(int w = 0; w < input_width_calc; ++w) {
                        dInput_unpadded[n][c][h][w] = dInput_padded[n][c][h + padding][w + padding];
                    }
                }
            }
        }

        // Assign dInputUnpadded to the input tensor's grad
        #pragma omp parallel for collapse(2)
        for(int n = 0; n < batch_size; ++n) {
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < input_height_calc; ++h) {
                    for(int w = 0; w < input_width_calc; ++w) {
                        input_tensor->grad[n][c][h][w] += dInput_unpadded[n][c][h][w];
                    }
                }
            }
        }

        auto dInput_tensor = std::make_shared<Tensor<Type>>();
        dInput_tensor->data = dInput_unpadded;
        dInput_tensor->grad = Tensor4D(dInput_unpadded.size(), Tensor3D(dInput_unpadded[0].size(),
                                                                        std::vector<std::vector<Type>>(dInput_unpadded[0][0].size(),
                                                                                std::vector<Type>(dInput_unpadded[0][0][0].size(), static_cast<Type>(0.0)))));

        return dInput_tensor;
    }
    else {
        // Assign dInput_padded to the input tensor's grad
        #pragma omp parallel for collapse(2)
        for(int n = 0; n < batch_size; ++n) {
            for(int c = 0; c < channels; ++c) {
                for(int h = 0; h < input_height_calc; ++h) {
                    for(int w = 0; w < input_width_calc; ++w) {
                        input_tensor->grad[n][c][h][w] += dInput_padded[n][c][h][w];
                    }
                }
            }
        }

        auto dInput_tensor = std::make_shared<Tensor<Type>>();
        dInput_tensor->data = dInput_padded;
        dInput_tensor->grad = Tensor4D(dInput_padded.size(), Tensor3D(dInput_padded[0].size(),
                                                                      std::vector<std::vector<Type>>(dInput_padded[0][0].size(),
                                                                              std::vector<Type>(dInput_padded[0][0][0].size(), static_cast<Type>(0.0)))));

        return dInput_tensor;
    }
}