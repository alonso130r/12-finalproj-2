//
// Created by Vijay Goyal on 2025-01-08.
//

#include "Tensor.h"

template <typename Type>
Tensor<Type>::Tensor(int batch_size, int channels, int height, int width, Type value) {
    data = Tensor4D(batch_size, Tensor3D(channels, std::vector<std::vector<Type>>(height, std::vector<Type>(width, value))));
    grad = Tensor4D(batch_size, Tensor3D(channels, std::vector<std::vector<Type>>(height, std::vector<Type>(width, static_cast<Type>(0.0)))));
}

template <typename Type>
void Tensor<Type>::zeroGrad() {
    for(auto& batch : grad) {
        for(auto& channel : batch) {
            for(auto& row : channel) {
                std::fill(row.begin(), row.end(), static_cast<Type>(0.0));
            }
        }
    }
}

template <typename Type>
void Tensor<Type>::setValue(size_t b, size_t c, size_t h, size_t w, Type value) {
    data[b][c][h][w] = value;
}