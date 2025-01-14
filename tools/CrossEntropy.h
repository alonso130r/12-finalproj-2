//
// Created by Vijay Goyal on 2025-01-14.
//

#ifndef MODULARCNN_CROSSENTROPY_H
#define MODULARCNN_CROSSENTROPY_H

#include <memory>
#include <cmath>
#include "Tensor.h"

/**
 * @brief A template cross-entropy loss class that uses Tensor<Type> objects
 *        (storing data in `data` and gradients in `grad`).
 *
 *   - pred->data: shape [N, C, 1, 1], predicted probabilities
 *   - target->data: shape [N, C, 1, 1], one-hot ground truth
 *   - pred->grad: Will be filled by backward() with dL/dPred
 */
template <typename Type>
class CrossEntropy {
private:
    bool reductionMean;

public:
    explicit CrossEntropy(bool reductionMean = true) : reductionMean(reductionMean) {}

    Type forward(const std::shared_ptr<Tensor<Type>>& pred, const std::shared_ptr<Tensor<Type>>& target);

    void backward(const std::shared_ptr<Tensor<Type>>& pred, const std::shared_ptr<Tensor<Type>>& target);};

#include "CrossEntropy.tpp"

#endif //MODULARCNN_CROSSENTROPY_H
