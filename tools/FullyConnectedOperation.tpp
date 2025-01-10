//
// Created by Vijay Goyal on 2025-01-08.
//

#include "FullyConnectedOperation.h"
#include <stdexcept>
#include <algorithm>

template <typename Type>
FullyConnectedOperation<Type>::FullyConnectedOperation(FullyConnectedLayer<Type>& layer, bool is_activated): fcLayer(layer), is_activated(is_activated) {}

template <typename Type>
std::vector<Type> FullyConnectedOperation<Type>::flattenSample(const Tensor4D& data, int n) {
    // channels = data[0].size(), height = data[0][0].size(), width = data[0][0][0].size()
    int channels = data[n].size();
    int height = data[n][0].size();
    int width = data[n][0][0].size();

    std::vector<Type> flattened(channels * height * width, 0.0);
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
std::shared_ptr <Tensor<Type>> FullyConnectedOperation<Type>::forward(const std::vector <std::shared_ptr<Tensor<Type>>> &inputs) {
    if(inputs.size() != 1) {
        throw std::invalid_argument("FullyConnectedOperation expects exactly one input tensor.");
    }
    auto input = inputs[0];
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

    auto output = std::make_shared<Tensor>(batch_size, fcLayer.out_features, 1, 1, 0.0);

    // for each sample in the batch
    for(int n = 0; n < batch_size; ++n) {
        std::vector<Type> x = flattenSample(input->data, n);
        // multiply
        for(int out_i = 0; out_i < fcLayer.out_features; ++out_i) {
            Type sum = 0.0;
            for(int in_j = 0; in_j < fcLayer.in_features; ++in_j) {
                sum += fcLayer.weights[out_i][in_j] * x[in_j];
            }
            sum += fcLayer.biases[out_i]; // add bias
            if (is_activated) sum = std::max(0.0, sum); // activation on condition
            output->data[n][out_i][0][0] = sum;
        }
    }

    return output;
}

template <typename Type>
std::shared_ptr <Tensor<Type>> FullyConnectedOperation<Type>::backward(const std::shared_ptr <Tensor<Type>> &output_grad) {
    if(this->inputs.empty()) {
        throw std::runtime_error("FullyConnectedOperation has no stored inputs. Perform forward pass first.");
    }
    auto input = this->inputs[0]; // original input
    int batch_size = input->data.size();

    // flatten dimension = channels * height * width
    int channels = input->data[0].size();
    int height   = input->data[0][0].size();
    int width    = input->data[0][0][0].size();
    int flatten_dim = channels * height * width;

    auto dInput = std::make_shared<Tensor>(batch_size, channels, height, width, 0.0);

    fcLayer.zeroGrad();

    // for each sample in the batch
    for(int n = 0; n < batch_size; ++n) {
        // flatten the sample input
        std::vector<Type> x = flattenSample(input->data, n);

        // out_grad: shape (out_features), i.e., output_grad->data[n][out_i][0][0]
        std::vector<Type> grad_out(fcLayer.out_features, 0.0);
        for(int out_i = 0; out_i < fcLayer.out_features; ++out_i) {
            grad_out[out_i] = output_grad->grad[n][out_i][0][0];
        }

        // compute gradients w.r.t. W, b, and x
        for(int out_i = 0; out_i < fcLayer.out_features; ++out_i) {
            Type go = grad_out[out_i];  // gradient w.r.t. output neuron out_i
            // bias gradient
            fcLayer.dBiases[out_i] += go;

            for(int in_j = 0; in_j < fcLayer.in_features; ++in_j) {
                // dWeights = x * grad
                fcLayer.dWeights[out_i][in_j] += go * x[in_j];
                // dInput = W^T * grad
                // Add to the input gradient at position in_j
                dInput->grad[n][in_j / (width*height)]
                [ (in_j % (width*height)) / width ]
                [ (in_j % (width*height)) % width ]
                        += fcLayer.weights[out_i][in_j] * go;
            }
        }
    }
    return dInput; // let the graph handle adding this to the actual input->grad if needed
}