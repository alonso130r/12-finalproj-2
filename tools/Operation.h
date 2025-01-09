//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_OPERATION_H
#define INC_12_FINALPROJ_2_OPERATION_H

#include <Tensor.h>
#include <vector>
#include <memory>

class Operation {
public:
    std::vector<std::shared_ptr<Tensor>> inputs;

    virtual ~Operation() = default;

    virtual std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;

    virtual std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& output_grad) = 0;
};


#endif //INC_12_FINALPROJ_2_OPERATION_H
