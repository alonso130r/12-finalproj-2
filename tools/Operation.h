//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_OPERATION_H
#define INC_12_FINALPROJ_2_OPERATION_H

#include <Tensor.h>

class Operation {
public:
    virtual ~Operation() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    virtual void backward(Tensor& output_grad) = 0;
};


#endif //INC_12_FINALPROJ_2_OPERATION_H
