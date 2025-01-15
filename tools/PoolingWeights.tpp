//
// Created by adamk on 2025-01-14.
//

#include "PoolingWeights.h"

template <typename Type>
PoolingWeights<Type>::PoolingWeights(MaxPoolingLayer<Type> const& layer) {
      pool_height = layer->pool_height;
      pool_width = layer->pool_width;
      stride = layer->stride;
      padding = layer->padding;
}

template <typename Type>
WeightStructType PoolingWeights<Type>::getType() const {
      return WeightStructType::PoolingWeights;
}