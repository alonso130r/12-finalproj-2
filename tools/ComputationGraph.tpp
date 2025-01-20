//
// Created by Vijay Goyal on 2025-01-08.
//

#include "ComputationGraph.h"
#include <iostream>

template <typename Type>
void ComputationGraph<Type>::addOperation(const std::shared_ptr<Operation<Type>>& operation) {
    operations.push_back(operation);
}

template <typename Type>
std::shared_ptr<Tensor<Type>> ComputationGraph<Type>::forward(const std::shared_ptr<Tensor<Type>>& input) {
    std::shared_ptr<Tensor<Type>> current = input;
    for(auto& op : operations) {
        current = op->forward({current});
    }
    return current;

    // // Print input shape and values
    // std::cout << "Input shape: " << input->data.size() << "x" 
    //           << input->data[0].size() << "x"
    //           << input->data[0][0].size() << "x" 
    //           << input->data[0][0][0].size() << std::endl;

    // std::shared_ptr<Tensor<Type>> current = input;
    
    // // Track and print each operation's output
    // for(size_t i = 0; i < operations.size(); i++) {
    //     current = operations[i]->forward({current});
        
    //     std::cout << "Operation " << i << " output shape: "
    //               << current->data.size() << "x"
    //               << current->data[0].size() << "x" 
    //               << current->data[0][0].size() << "x"
    //               << current->data[0][0][0].size() << std::endl;
        
    //     std::cout << "First few values: ";
    //     if(!current->data.empty() && !current->data[0].empty() && 
    //        !current->data[0][0].empty() && !current->data[0][0][0].empty()) {
    //         for(int j = 0; j < std::min(5, (int)current->data[0][0][0].size()); j++) {
    //             std::cout << current->data[0][0][0][j] << " ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    
    // return current;
}

template <typename Type>
void ComputationGraph<Type>::backward(const std::shared_ptr<Tensor<Type>>& loss_grad) {
    std::shared_ptr<Tensor<Type>> current_grad = loss_grad;
    for(auto it = operations.rbegin(); it != operations.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
}