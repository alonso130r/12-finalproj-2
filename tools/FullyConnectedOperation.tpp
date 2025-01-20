//
// Created by Vijay Goyal on 2025-01-08.
//

#include "FullyConnectedOperation.h"
#include <stdexcept>
#include <algorithm>
#include <omp.h>

template <typename Type>
FullyConnectedOperation<Type>::FullyConnectedOperation(FullyConnectedLayer<Type>& layer, bool is_activated): fcLayer(layer), is_activated(is_activated) {}

template <typename Type>
std::vector<Type> FullyConnectedOperation<Type>::flattenSample(const Tensor4D& data, int n) {
    // channels = data[0].size(), height = data[0][0].size(), width = data[0][0][0].size()
    int channels = data[n].size();
    int height = data[n][0].size();
    int width = data[n][0][0].size();

    std::vector<Type> flattened(channels * height * width, static_cast<Type>(0.0));
    int idx = 0;
    for(int c = 0; c < channels; ++c) {
        for(int h = 0; h < height; ++h) {
            for(int w = 0; w < width; ++w) {
                flattened[idx++] = data[n][c][h][w];
            }
        }
    }
    return flattened;
}

template <typename Type>
std::shared_ptr<Tensor<Type>> FullyConnectedOperation<Type>::forward(const std::shared_ptr<Tensor<Type>> &inputs) {
    auto input = inputs;
    this->inputs = inputs; // store for backward

    // assume input has shape: (batch_size, channels, height, width)
    int batch_size = input->data.size();
    int channels   = input->data[0].size();
    int height     = input->data[0][0].size();
    int width      = input->data[0][0][0].size();

    // flatten dimension = channels*height*width must match fcLayer.in_features
    int flatten_dim = channels * height * width;
    if(flatten_dim != fcLayer.in_features) {
        throw std::invalid_argument("FullyConnectedOperation: Flattened input size does not match fcLayer.in_features.");
    }

    auto output = std::make_shared<Tensor<Type>>(batch_size, fcLayer.out_features, 1, 1, static_cast<Type>(0.0));

    // Parallelize over the batch dimension
    #pragma omp parallel for
    for(int n = 0; n < batch_size; ++n) {
        std::vector<Type> x = flattenSample(input->data, n);
        for(int out_i = 0; out_i < fcLayer.out_features; ++out_i) {
            Type sum = fcLayer.biases[out_i];
            for(int in_j = 0; in_j < fcLayer.in_features; ++in_j) {
                sum += fcLayer.weights[out_i][in_j] * x[in_j];
            }
            if (is_activated) {
                sum = std::max(static_cast<Type>(0.0), sum); // ReLU activation
            }
            output->data[n][out_i][0][0] = sum;
        }
    }

    return output;
}

template <typename Type>
std::shared_ptr<Tensor<Type>> FullyConnectedOperation<Type>::backward(const std::shared_ptr<Tensor<Type>> &output_grad) {
    if(this->inputs->data.empty() || this->inputs->data[0].empty()) {
        throw std::runtime_error("FullyConnectedOperation has no stored inputs. Perform forward pass first.");
    }
    auto input = this->inputs->data; // original input
    int batch_size = static_cast<int>(input.size());

    if(batch_size == 0) {
        throw std::invalid_argument("Input batch size is zero.");
    }

    int channels = static_cast<int>(input[0].size());
    int height   = static_cast<int>(input[0][0].size());
    int width    = static_cast<int>(input[0][0][0].size());
    int flatten_dim = channels * height * width;

    // Validate fcLayer dimensions
    if(fcLayer.in_features != flatten_dim) {
        throw std::invalid_argument("fcLayer.in_features does not match input dimensions.");
    }

    // Add checks for output_grad dimensions before the parallel region
    if(output_grad == nullptr) {
        throw std::invalid_argument("output_grad is null.");
    }
    if(output_grad->grad.size() != static_cast<size_t>(batch_size)) {
        throw std::invalid_argument("output_grad->grad size does not match batch_size.");
    }
    for(int n = 0; n < batch_size; ++n) {
        if(output_grad->grad.at(n).size() != static_cast<size_t>(fcLayer.out_features)) {
            throw std::invalid_argument("output_grad->grad at sample " + std::to_string(n) + " does not match fcLayer.out_features.");
        }
        for(int out_i = 0; out_i < fcLayer.out_features; ++out_i) {
            if(output_grad->grad.at(n).at(out_i).size() != 1 || output_grad->grad.at(n).at(out_i).at(0).size() != 1) {
                throw std::invalid_argument("output_grad->grad at sample " + std::to_string(n) + ", feature " + std::to_string(out_i) + " has incorrect dimensions.");
            }
        }
    }

    // Initialize dInput with zeros
    auto dInput = std::make_shared<Tensor<Type>>(batch_size, channels, height, width, static_cast<Type>(0.0));

    // Zero existing gradients
    fcLayer.zeroGrad();

    // Determine number of threads
    int num_threads = omp_get_max_threads();

    // Validate thread count
    if(num_threads <= 0) {
        throw std::runtime_error("Number of threads must be positive.");
    }

    // Create thread-local accumulators for dWeights and dBiases
    std::vector<std::vector<std::vector<Type>>> dWeights_local(
        num_threads,
        std::vector<std::vector<Type>>(fcLayer.out_features, std::vector<Type>(fcLayer.in_features, static_cast<Type>(0.0)))
    );
    std::vector<std::vector<Type>> dBiases_local(
        num_threads,
        std::vector<Type>(fcLayer.out_features, static_cast<Type>(0.0))
    );

    // Initialize thread-local dInput_local as 4D: [num_threads][batch_size][channels][height * width]
    std::vector<std::vector<std::vector<std::vector<Type>>>> dInput_local(
        num_threads,
        std::vector<std::vector<std::vector<Type>>>(batch_size,
            std::vector<std::vector<Type>>(channels,
                std::vector<Type>(height * width, static_cast<Type>(0.0))))
    );

    // Parallelize over the batch dimension
    #pragma omp parallel for
    for(int n = 0; n < batch_size; ++n) {
        int thread_id = omp_get_thread_num();
        if(thread_id >= num_threads) {
            throw std::out_of_range("Thread ID exceeds number of threads.");
        }

        std::vector<Type> x = flattenSample(input, n);

         // Ensure that flattenSample returns the correct size
        if(x.size() != static_cast<size_t>(fcLayer.in_features)) {
            throw std::out_of_range("flattenSample returned incorrect size.");
        }

        for(int out_i = 0; out_i < fcLayer.out_features; ++out_i) {
            // Removed redundant out_i range check
            Type go = output_grad->grad.at(n).at(out_i).at(0).at(0); // problem line
            dBiases_local[thread_id].at(out_i) += go;

            for(int in_j = 0; in_j < fcLayer.in_features; ++in_j) {
                // Removed redundant in_j range check
                dWeights_local[thread_id].at(out_i).at(in_j) += go * x.at(in_j);

                // Compute mapping from in_j to channel, height, and width
                int channel_idx = in_j / (height * width);
                int hw_reduced = in_j % (height * width);
                int h = hw_reduced / width;
                int w = hw_reduced % width;

                // Bounds check for channel_idx, h, and w
                if(channel_idx < 0 || channel_idx >= channels || h < 0 || h >= height || w < 0 || w >= width) {
                    throw std::out_of_range("channel_idx, h, or w out of range.");
                }

                dInput_local[thread_id][n][channel_idx][hw_reduced] += fcLayer.weights.at(out_i).at(in_j) * go;
            }
        }
    }

    // Aggregate thread-local dWeights and dBiases into the global gradients
    for(int t = 0; t < num_threads; ++t) {
        for(int out_i = 0; out_i < fcLayer.out_features; ++out_i) {
            fcLayer.dBiases.at(out_i) += dBiases_local.at(t).at(out_i);
            for(int in_j = 0; in_j < fcLayer.in_features; ++in_j) {
                fcLayer.dWeights.at(out_i).at(in_j) += dWeights_local.at(t).at(out_i).at(in_j);
            }
        }
    }

    // Aggregate thread-local dInput
    #pragma omp parallel for collapse(3)
    for(int n = 0; n < batch_size; ++n) {
        for(int c = 0; c < channels; ++c) {
            for(int hw_reduced = 0; hw_reduced < height * width; ++hw_reduced) {
                int h = hw_reduced / width;
                int w = hw_reduced % width;

                if(h < 0 || h >= height || w < 0 || w >= width) {
                    throw std::out_of_range("h or w index out of range during aggregation.");
                }

                for(int t = 0; t < num_threads; ++t) {
                    dInput->grad.at(n).at(c).at(h).at(w) += dInput_local.at(t).at(n).at(c).at(hw_reduced);
                }
            }
        }
    }

    return dInput; // let the graph handle adding this to the actual input->grad if needed
}
