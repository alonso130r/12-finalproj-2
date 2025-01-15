//
// Created by adamk on 2025-01-14.
//

#ifndef WEIGHTSTRUCT_H
#define WEIGHTSTRUCT_H

#include <vector>
#include "Tensor.h"
#include <memory>
#include <string>

enum class WeightStructType : int {
    ConvolutionalWeights = 0,
    PoolingWeights = 1,
    ConnectedWeights = 2
};

template <typename Type>
struct WeightStruct {
    virtual WeightStructType getType() const = 0;
};



#endif //WEIGHTSTRUCT_H
