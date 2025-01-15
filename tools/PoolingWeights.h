//
// Created by adamk on 2025-01-14.
//

#ifndef POOLINGWEIGHTS_H
#define POOLINGWEIGHTS_H

#include "WeightStruct.h"
#include "../layers/MaxPoolingLayer.h"


template <typename Type>
struct PoolingWeights : public WeightStruct<Type> {
    int pool_height;
    int pool_width;
    int stride;
    int padding;

    PoolingWeights(const MaxPoolingLayer<Type>& layer);
    WeightStructType getType() const override;

};

#include "PoolingWeights.tpp"

#endif //POOLINGWEIGHTS_H
