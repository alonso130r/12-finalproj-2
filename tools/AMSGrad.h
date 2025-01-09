//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_AMSGRAD_H
#define INC_12_FINALPROJ_2_AMSGRAD_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "ConvolutionLayer.h"
#include "FullyConnectedLayer.h"

/**
 * @brief AdamW + AMSGrad optimizer.
 *        - Adam with decoupled weight decay (AdamW)
 *        - AMSGrad ensures a non-decreasing second-moment estimate v_hat.
 */class AMSGrad {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay; // decoupled weight decay factor
    int time_step;

    bool initializedConv;
    std::vector<std::vector<std::vector<std::vector<double>>>> m_filters;     // first moment
    std::vector<std::vector<std::vector<std::vector<double>>>> v_filters;     // second moment
    std::vector<std::vector<std::vector<std::vector<double>>>> v_hat_filters; // max of v

    std::vector<double> m_biases;
    std::vector<double> v_biases;
    std::vector<double> v_hat_biases;

    bool initializedFC;
    std::vector<std::vector<double>> m_fc_weights;
    std::vector<std::vector<double>> v_fc_weights;
    std::vector<std::vector<double>> v_hat_fc_weights;

    std::vector<double> m_fc_biases;
    std::vector<double> v_fc_biases;
    std::vector<double> v_hat_fc_biases;

public:
    AMSGrad(double lr = 1e-3, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 1e-2);

    // convolution
    void initializeConv(const ConvolutionLayer& layer);
    void update(ConvolutionLayer& layer,
                const std::vector<std::vector<std::vector<std::vector<double>>>>& dFilters,
                const std::vector<double>& dBiases);

    // fully connected
    void initializeFC(const FullyConnectedLayer& layer);
    void update(FullyConnectedLayer& layer,
                const std::vector<std::vector<double>>& dWeights,
                const std::vector<double>& dBiases);
};


#endif //INC_12_FINALPROJ_2_AMSGRAD_H
