//
// Created by Vijay Goyal on 2025-01-08.
//

#include "ComputationGraph.h"

template <typename Type>
void ComputationGraph<Type>::addOperation(std::shared_ptr<Operation<Type>> operation) {
    operations.push_back(operation);
}

template <typename Type>
std::shared_ptr<Tensor<Type>> ComputationGraph<Type>::forward(const std::shared_ptr<Tensor<Type>>& input) {
    std::shared_ptr<Tensor<Type>> current = input;
    for(auto& op : operations) {
        current = op->forward({current});
    }
    return current;
}

template <typename Type>
void ComputationGraph<Type>::backward(const std::shared_ptr<Tensor<Type>>& loss_grad) {
    std::shared_ptr<Tensor<Type>> current_grad = loss_grad;
    for(auto it = operations.rbegin(); it != operations.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
}