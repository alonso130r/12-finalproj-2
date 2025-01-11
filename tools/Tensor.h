//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_TENSOR_H
#define INC_12_FINALPROJ_2_TENSOR_H

#include <vector>
#include <memory>

template <typename Type>
class Operation;

template <typename Type>
class Tensor {
    typedef std::vector<std::vector<std::vector<Type>>> Tensor3D; // (channels, height, width)
    typedef std::vector<std::vector<std::vector<std::vector<Type>>>> Tensor4D; // (batch_size, channels, height, width)
public:
    Tensor4D data;
    Tensor4D grad;
    std::shared_ptr<Operation<Type>> creator; // points to operation that made it/edited it, for computation graphs

//    Tensor() = default;
    Tensor(int batch_size, int channels, int height, int width, Type value = 0.0);

    void zeroGrad(); // to clear the gradients
};

#include "Tensor.tpp"

#endif //INC_12_FINALPROJ_2_TENSOR_H
