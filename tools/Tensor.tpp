//
// Created by Vijay Goyal on 2025-01-08.
//

#include "Tensor.h"

// add better constructor for initializing data
template <typename Type>
Tensor<Type>::Tensor(int batch_size, int channels, int height, int width, Type value) {
    data = Tensor4D(batch_size, Tensor3D(channels, std::vector<std::vector<Type>>(height, std::vector<Type>(width, value))));
    grad = Tensor4D(batch_size, Tensor3D(channels, std::vector<std::vector<Type>>(height, std::vector<Type>(width, 0.0))));
}

template <typename Type>
void Tensor<Type>::zeroGrad() {
    for(auto& batch : grad) {
        for(auto& channel : batch) {
            for(auto& row : channel) {
                std::fill(row.begin(), row.end(), 0.0);
            }
        }
    }
}