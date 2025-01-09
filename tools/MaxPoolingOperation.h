//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_MAXPOOLINGOPERATION_H
#define INC_12_FINALPROJ_2_MAXPOOLINGOPERATION_H

#include "Operation.h"
#include "Tensor.h"
#include <vector>
#include <utility>

typedef std::vector<std::vector<std::vector<double>>> Tensor3D; // (channels, height, width)
typedef std::vector<std::vector<std::vector<std::vector<double>>>> Tensor4D; // (batch_size, channels, height, width)


class MaxPoolingOperation : public Operation {
private:
    int pool_height;
    int pool_width;
    int stride;
    int padding;

    // indices of maxima for each pooling window
    std::vector<std::vector<std::vector<std::vector<std::pair<int, int>>>>> max_indices;

public:
    MaxPoolingOperation(int pool_height, int pool_width, int stride = 1, int padding = 0);

    Tensor forward(const Tensor &input) override;

    void backward(Tensor &output_grad) override;
};


#endif //INC_12_FINALPROJ_2_MAXPOOLINGOPERATION_H
