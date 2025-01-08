//
// Created by Vijay Goyal on 2025-01-08.
//

#include "ComputationGraph.h"

void ComputationGraph::addOperation(std::shared_ptr<Operation> operation) {
    operations.push_back(operation);
}

Tensor ComputationGraph::forward(const Tensor& input) {
    Tensor current = input;
    for(auto& op : operations) {
        current = op->forward(current);
    }
    return current;
}

void ComputationGraph::backward(const Tensor& loss_grad) {
    Tensor current = loss_grad;
    for(auto it = operations.rbegin(); it != operations.rend(); ++it) {
        current = (*it)->backward(current);
    }
}