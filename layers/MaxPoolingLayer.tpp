//
// Created by Vijay Goyal on 2025-01-08.
//

#include "MaxPoolingLayer.h"

template <typename Type>
MaxPoolingLayer<Type>::MaxPoolingLayer(int pool_height, int pool_width, int stride, int padding)
        : pool_height(pool_height), pool_width(pool_width), stride(stride), padding(padding)
{
    maxPoolOp = std::make_shared<MaxPoolingOperation<Type>>(pool_height, pool_width, stride, padding); // will find alternative soon
}

template <typename Type>
std::shared_ptr<Tensor<Type>> MaxPoolingLayer<Type>::forward(std::shared_ptr<Tensor<Type>> &input) {
    auto output = maxPoolOp->forward({input});
    return output;
}

template <typename Type>
std::shared_ptr<Tensor<Type>> MaxPoolingLayer<Type>::backward(std::shared_ptr<Tensor<Type>> &dOut) {
    auto input_grad = maxPoolOp->backward(dOut);
    return input_grad;
}

template <typename Type>
ssize_t MaxPoolingLayer<Type>::getNumParams() const {
    return 0;
}

template<typename Type>
void MaxPoolingLayer<Type>::zeroGrad() {
    // do nothing
}

template <typename Type>
std::shared_ptr<WeightStruct<Type>> MaxPoolingLayer<Type>::saveWeights(const std::string location) {
    return std::make_shared<PoolingWeights<Type>>(this);
}