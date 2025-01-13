//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_MAXPOOLINGLAYER_H
#define INC_12_FINALPROJ_2_MAXPOOLINGLAYER_H

#include "../tools/Operation.h"
#include "../tools/MaxPoolingOperation.h"
#include "../tools/Tensor.h"
#include <memory>

template <typename Type>
class MaxPoolingLayer : public Layer<Type> {
private:
    int pool_height;
    int pool_width;
    int stride;
    int padding;
    std::shared_ptr<MaxPoolingOperation<Type>> maxPoolOp;

public:
    MaxPoolingLayer(int pool_height, int pool_width, int stride = 1, int padding = 0);
    std::shared_ptr<Tensor<Type>> forward(std::shared_ptr<Tensor<Type>> &input);
    std::shared_ptr<Tensor<Type>> backward(std::shared_ptr<Tensor<Type>> &dOut);
    [[nodiscard]] ssize_t getNumParams() const;
};

#include "MaxPoolingLayer.tpp"

#endif //INC_12_FINALPROJ_2_MAXPOOLINGLAYER_H
