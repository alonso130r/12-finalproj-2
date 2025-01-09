//
// Created by Vijay Goyal on 2025-01-08.
//

#include "MaxPoolingLayer.h"

MaxPoolingLayer::MaxPoolingLayer(int pool_height, int pool_width, int stride, int padding)
        : pool_height(pool_height), pool_width(pool_width), stride(stride), padding(padding)
{
    maxPoolOp = std::make_shared<MaxPoolingOperation>(pool_height, pool_width, stride, padding); // will find alternative soon
}

std::shared_ptr<Tensor> MaxPoolingLayer::forward(const std::shared_ptr<Tensor> &input) {
    auto output = maxPoolOp->forward({input});
    return output;
}

std::shared_ptr<Tensor> MaxPoolingLayer::backward(const std::shared_ptr<Tensor> &dOut) {
    auto input_grad = maxPoolOp->backward(dOut);
    return input_grad;
}

size_t MaxPoolingLayer::getNumParams() const {
    return 0;
}