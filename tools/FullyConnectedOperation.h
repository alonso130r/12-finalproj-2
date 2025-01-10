//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_FULLYCONNECTEDOPERATION_H
#define INC_12_FINALPROJ_2_FULLYCONNECTEDOPERATION_H

#include "Operation.h"
#include "FullyConnectedLayer.h"
#include "Tensor.h"
#include <memory>

template <typename Type>
class FullyConnectedOperation : public Operation {
private:
    FullyConnectedLayer& fcLayer<Type>;

    bool is_activated;

    std::vector<Type> flattenSample(const Tensor4D& data, int n);

public:
    FullyConnectedOperation(FullyConnectedLayer& fcLayer<Type>, bool is_activated = true);

    std::shared_ptr<Tensor<Type>> forward(const std::vector<std::shared_ptr<Tensor<Type>>>& inputs) override;
    std::shared_ptr<Tensor<Type>> backward(const std::shared_ptr<Tensor<Type>>& output_grad) override;
};

#include "FullyConnectedOperation.tpp"

#endif //INC_12_FINALPROJ_2_FULLYCONNECTEDOPERATION_H
