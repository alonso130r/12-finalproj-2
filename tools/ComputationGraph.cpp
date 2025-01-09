//
// Created by Vijay Goyal on 2025-01-08.
//

#include "ComputationGraph.h"

void ComputationGraph::addOperation(std::shared_ptr<Operation> operation) {
    operations.push_back(operation);
}

std::shared_ptr<Tensor> ComputationGraph::forward(const std::shared_ptr<Tensor>& input) {
    std::shared_ptr<Tensor> current = input;
    for(auto& op : operations) {
        current = op->forward({current});
    }
    return current;
}

void ComputationGraph::backward(const std::shared_ptr<Tensor>& loss_grad) {
    std::shared_ptr<Tensor> current_grad = loss_grad;
    for(auto it = operations.rbegin(); it != operations.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
}