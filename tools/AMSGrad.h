//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_AMSGRAD_H
#define INC_12_FINALPROJ_2_AMSGRAD_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "../layers/ConvolutionLayer.h"
#include "../layers/FullyConnectedLayer.h"

/**
 * @brief AdamW + AMSGrad optimizer.
 *        - Adam with decoupled weight decay (AdamW)
 *        - AMSGrad ensures a non-decreasing second-moment estimate v_hat.
 */

template <typename Type>
class AMSGrad {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay; // decoupled weight decay factor
    int time_step;

    bool initializedConv;
    std::vector<std::vector<std::vector<std::vector<Type>>>> m_filters;     // first moment
    std::vector<std::vector<std::vector<std::vector<Type>>>> v_filters;     // second moment
    std::vector<std::vector<std::vector<std::vector<Type>>>> v_hat_filters; // max of v

    std::vector<Type> m_biases;
    std::vector<Type> v_biases;
    std::vector<Type> v_hat_biases;

    bool initializedFC;
    std::vector<std::vector<Type>> m_fc_weights;
    std::vector<std::vector<Type>> v_fc_weights;
    std::vector<std::vector<Type>> v_hat_fc_weights;

    std::vector<Type> m_fc_biases;
    std::vector<Type> v_fc_biases;
    std::vector<Type> v_hat_fc_biases;

public:
    explicit AMSGrad(double lr = 1e-3, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 1e-2);

    // convolution
    void initializeConv(const ConvolutionLayer<Type>& layer);
    void update(ConvolutionLayer<Type>& layer,
                const std::vector<std::vector<std::vector<std::vector<Type>>>>& dFilters,
                const std::vector<Type>& dBiases);

    // fully connected
    void initializeFC(const FullyConnectedLayer<Type>& layer);
    void update(FullyConnectedLayer<Type>& layer,
                const std::vector<std::vector<Type>>& dWeights,
                const std::vector<Type>& dBiases);
};

#include "AMSGrad.tpp"

#endif //INC_12_FINALPROJ_2_AMSGRAD_H
