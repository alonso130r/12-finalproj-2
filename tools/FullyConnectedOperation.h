//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_FULLYCONNECTEDOPERATION_H
#define INC_12_FINALPROJ_2_FULLYCONNECTEDOPERATION_H

#include "Operation.h"
#include "FullyConnectedLayer.h"
#include "Tensor.h"
#include <memory>

class FullyConnectedOperation : public Operation {
private:
    FullyConnectedLayer& fcLayer;

    bool is_activated;

    std::vector<double> flattenSample(const Tensor4D& data, int n);

public:
    FullyConnectedOperation(FullyConnectedLayer& fcLayer, bool is_activated = true);

    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& output_grad) override;
};


#endif //INC_12_FINALPROJ_2_FULLYCONNECTEDOPERATION_H
