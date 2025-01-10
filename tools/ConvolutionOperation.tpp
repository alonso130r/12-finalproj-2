//
// Created by Vijay Goyal on 2025-01-08.
//

#include "ConvolutionOperation.h"

template <typename Type>
ConvolutionOperation<Type>::ConvolutionOperation(ConvolutionLayer<Type>& layer) : convLayer(layer) {}

template <typename Type>
Tensor<Type> ConvolutionOperation<Type>::forward(const Tensor<Type>& input_tensor) {
    input = input_tensor; // cache input for backpropagation
    Tensor<Type> output = convLayer.forward(input.data); // perform convolution
    return output;
}

template <typename Type>
void ConvolutionOperation<Type>::backward(Tensor<Type>& output_grad) {
    Tensor4D dInput = convLayer.backward(output_grad.data);

    // accumulate gradients
    for(int n = 0; n < dInput.size(); ++n) {
        for(int c = 0; c < dInput[0].size(); ++c) {
            for(int h = 0; h < dInput[0][0].size(); ++h) {
                for(int w = 0; w < dInput[0][0][0].size(); ++w) {
                    input.grad[n][c][h][w] += dInput[n][c][h][w];
                }
            }
        }
    }
}