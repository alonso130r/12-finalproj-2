//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_MAXPOOLINGLAYER_H
#define INC_12_FINALPROJ_2_MAXPOOLINGLAYER_H

#include "Operation.h"
#include "MaxPoolingOperation.h"
#include "Tensor.h"
#include <memory>

class MaxPoolingLayer {
private:
    int pool_height;
    int pool_width;
    int stride;
    int padding;
    std::shared_ptr<MaxPoolingOperation> maxPoolOp;

public:
    MaxPoolingLayer(int pool_height, int pool_width, int stride = 1, int padding = 0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> &dOut);
};


#endif //INC_12_FINALPROJ_2_MAXPOOLINGLAYER_H
