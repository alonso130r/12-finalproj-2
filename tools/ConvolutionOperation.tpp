//
// Created by Vijay Goyal on 2025-01-08.
//

#include "ConvolutionOperation.h"

template <typename Type>
ConvolutionOperation<Type>::ConvolutionOperation(ConvolutionLayer<Type>& layer) : convolutionLayer(layer), input(std::make_shared<Tensor<Type>>(0, 0, 0, 0, static_cast<Type>(0.0))) {}

template <typename Type>
std::shared_ptr<Tensor<Type>> ConvolutionOperation<Type>::forward(const std::shared_ptr<Tensor<Type>>& input_tensor) {
    input = input_tensor; // cache input for backpropagation
    std::shared_ptr<Tensor<Type>> output = convolutionLayer.forward(input); // perform convolution
    return output;
}

template <typename Type>
std::shared_ptr<Tensor<Type>> ConvolutionOperation<Type>::backward(const std::shared_ptr<Tensor<Type>>& output_grad) {
    Tensor4D dInput = convolutionLayer.backward(output_grad);

    // accumulate gradients
    for(int n = 0; n < dInput.size(); ++n) {
        for(int c = 0; c < dInput[0].size(); ++c) {
            for(int h = 0; h < dInput[0][0].size(); ++h) {
                for(int w = 0; w < dInput[0][0][0].size(); ++w) {
                    input->grad[n][c][h][w] += dInput[n][c][h][w];
                }
            }
        }
    }

    return input;
}