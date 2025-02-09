//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_MAXPOOLINGOPERATION_H
#define INC_12_FINALPROJ_2_MAXPOOLINGOPERATION_H

#include "Operation.h"
#include "Tensor.h"
#include <vector>
#include <utility>

template <typename Type>
class MaxPoolingOperation : public Operation<Type> {
    typedef std::vector<std::vector<std::vector<Type>>> Tensor3D; // (channels, height, width)
    typedef std::vector<std::vector<std::vector<std::vector<Type>>>> Tensor4D; // (batch_size, channels, height, width)
private:
    int pool_height;
    int pool_width;
    int stride;
    int padding;

    // indices of maxima for each pooling window
    std::vector<std::vector<std::vector<std::vector<std::pair<int, int>>>>> max_indices;

    std::shared_ptr<Tensor<Type>> inputs;

public:
    MaxPoolingOperation(int pool_height, int pool_width, int stride = 1, int padding = 0);

    std::shared_ptr<Tensor<Type>> forward(const std::shared_ptr<Tensor<Type>> &input) override;

    std::shared_ptr<Tensor<Type>> backward(const std::shared_ptr<Tensor<Type>>& output_grad) override;
};

#include "MaxPoolingOperation.tpp"

#endif //INC_12_FINALPROJ_2_MAXPOOLINGOPERATION_H
