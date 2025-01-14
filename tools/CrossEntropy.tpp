//
// Created by Vijay Goyal on 2025-01-14.
//

#include "CrossEntropy.h"

template <typename Type>
Type CrossEntropy<Type>::forward(const std::shared_ptr<Tensor<Type>>& pred, const std::shared_ptr<Tensor<Type>>& target) {
    // pred->data: shape [N, C, 1, 1]
    // target->data: same shape
    int batchSize = static_cast<int>(pred->data.size());
    if(batchSize == 0) {
        return static_cast<Type>(0);
    }
    int numClasses = static_cast<int>(pred->data[0].size());
    // last two dims presumably = 1

    Type lossVal = static_cast<Type>(0);

    // cross-entropy: -sum(target * log(pred))
    // sum over (n, c), then optionally divide by N if mean reduction
    for(int n = 0; n < batchSize; ++n) {
        for(int c = 0; c < numClasses; ++c) {
            Type t = target->data[n][c][0][0]; // one-hot
            if(t > 0) {
                Type p = pred->data[n][c][0][0];
                // clamp p to avoid log(0)
                if(p < static_cast<Type>(1e-15)) {
                    p = static_cast<Type>(1e-15);
                }
                lossVal -= t * static_cast<Type>(std::log(p));
            }
        }
    }
    if(reductionMean) {
        lossVal /= static_cast<Type>(batchSize);
    }
    return lossVal;
}

template <typename Type>
void CrossEntropy<Type>::backward(const std::shared_ptr<Tensor<Type>>& pred, const std::shared_ptr<Tensor<Type>>& target) {
    // the gradient dL/dPred = (pred - target) / N, if pred is a softmax
    // shape: [N, C, 1, 1]
    int batchSize = static_cast<int>(pred->data.size());
    if(batchSize == 0) {
        return;
    }
    int numClasses = static_cast<int>(pred->data[0].size());

    // fill pred->grad
    Type scale = reductionMean ? (static_cast<Type>(1) / static_cast<Type>(batchSize))
                               : static_cast<Type>(1);

    for(int n = 0; n < batchSize; ++n) {
        for(int c = 0; c < numClasses; ++c) {
            Type p = pred->data[n][c][0][0];   // prob
            Type t = target->data[n][c][0][0]; // one-hot
            // derivative
            pred->grad[n][c][0][0] = (p - t) * scale;
        }
    }
}