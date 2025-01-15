//
// Created by adamk on 2025-01-14.
//

#ifndef POOLINGWEIGHTS_H
#define POOLINGWEIGHTS_H

#include "WeightStruct.h"

template <typename Type>
class MaxPoolingLayer;

template <typename Type>
struct PoolingWeights : public WeightStruct<Type> {
    int pool_height;
    int pool_width;
    int stride;
    int padding;

    explicit PoolingWeights(const MaxPoolingLayer<Type>& layer);
    [[nodiscard]] WeightStructType getType() const override;
    void serialize(std::ofstream& out) const override;
    static std::shared_ptr<MaxPoolingLayer<Type>> deserialize(std::ifstream& in);

};

#include "PoolingWeights.tpp"

#endif //POOLINGWEIGHTS_H
