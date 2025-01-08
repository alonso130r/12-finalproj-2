//
// Created by Vijay Goyal on 2025-01-08.
//

#include "Tensor.h"

Tensor::Tensor(int batch_size, int channels, int height, int width, double value) {
    data = Tensor4D(batch_size, Tensor3D(channels, std::vector<std::vector<double>>(height, std::vector<double>(width, value))));
    grad = Tensor4D(batch_size, Tensor3D(channels, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0))));
}

void Tensor::zeroGrad() {
    for(auto& batch : grad) {
        for(auto& channel : batch) {
            for(auto& row : channel) {
                std::fill(row.begin(), row.end(), 0.0);
            }
        }
    }
}