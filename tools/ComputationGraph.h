//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_COMPUTATIONGRAPH_H
#define INC_12_FINALPROJ_2_COMPUTATIONGRAPH_H

#include "Operation.h"
#include "Tensor.h"
#include <vector>
#include <memory>

template <typename Type>
class ComputationGraph {
private:
    std::vector<std::shared_ptr<Operation<Type>>> operations; // list of operations in the graph

public:
    void addOperation(std::shared_ptr<Operation<Type>> &operation); // add an operation to the graph
    std::shared_ptr<Tensor<Type>> forward(const std::shared_ptr<Tensor<Type>> &input); // perform a forward pass through the graph
    void backward(const std::shared_ptr<Tensor<Type>> &loss_grad); // perform a backward pass through the graph
};

//#include "ComputationGraph.tpp"

#endif //INC_12_FINALPROJ_2_COMPUTATIONGRAPH_H
