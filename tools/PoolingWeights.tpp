//
// Created by adamk on 2025-01-14.
//

#include "PoolingWeights.h"

template <typename Type>
PoolingWeights<Type>::PoolingWeights(MaxPoolingLayer<Type> const& layer) {
      pool_height = layer.pool_height;
      pool_width = layer.pool_width;
      stride = layer.stride;
      padding = layer.padding;
}

template <typename Type>
WeightStructType PoolingWeights<Type>::getType() const {
      return WeightStructType::PoolingWeights;
}

template<typename Type>
void PoolingWeights<Type>::serialize(std::ofstream &out) const {
      out.write(reinterpret_cast<const char*>(&pool_height), sizeof(pool_height));
      out.write(reinterpret_cast<const char*>(&pool_width), sizeof(pool_width));
      out.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
      out.write(reinterpret_cast<const char*>(&padding), sizeof(padding));
}

template<typename Type>
std::shared_ptr<MaxPoolingLayer<Type>> PoolingWeights<Type>::deserialize(std::ifstream &in) {
      int pool_height_t;
      int pool_width_t;
      int stride_t;
      int padding_t;

      in.read(reinterpret_cast<char*>(&pool_height_t), sizeof(pool_height_t));
      in.read(reinterpret_cast<char*>(&pool_width_t), sizeof(pool_width_t));
      in.read(reinterpret_cast<char*>(&stride_t), sizeof(stride_t));
      in.read(reinterpret_cast<char*>(&padding_t), sizeof(padding_t));


      auto temp = std::make_shared<MaxPoolingLayer<Type>>(pool_height_t, pool_width_t, stride_t, padding_t);
      return temp;
}