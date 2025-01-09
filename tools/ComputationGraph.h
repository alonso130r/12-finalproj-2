//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_COMPUTATIONGRAPH_H
#define INC_12_FINALPROJ_2_COMPUTATIONGRAPH_H

#include "Operation.h"
#include "Tensor.h"
#include <vector>
#include <memory>

class ComputationGraph {
private:
    std::vector<std::shared_ptr<Operation>> operations; // list of operations in the graph

public:
    void addOperation(std::shared_ptr<Operation> &operation); // add an operation to the graph
    Tensor forward(const std::shared_ptr<Tensor> &input); // perform a forward pass through the graph
    void backward(const std::shared_ptr<Tensor> &loss_grad); // perform a backward pass through the graph
};


#endif //INC_12_FINALPROJ_2_COMPUTATIONGRAPH_H
