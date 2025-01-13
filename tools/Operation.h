//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_OPERATION_H
#define INC_12_FINALPROJ_2_OPERATION_H

#include "Tensor.h"
#include <vector>
#include <memory>

template <typename Type>
class Operation {
public:
    std::vector<std::shared_ptr<Tensor<Type>>> inputs;

    virtual ~Operation() = default;

    virtual std::shared_ptr<Tensor<Type>> forward(const std::shared_ptr<Tensor<Type>>& inputs) = 0;

    virtual std::shared_ptr<Tensor<Type>> backward(const std::shared_ptr<Tensor<Type>>& output_grad) = 0;
};


#endif //INC_12_FINALPROJ_2_OPERATION_H
