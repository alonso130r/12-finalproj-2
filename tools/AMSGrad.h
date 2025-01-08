//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_AMSGRAD_H
#define INC_12_FINALPROJ_2_AMSGRAD_H

#include "ConvolutionLayer.h"
#include <vector>
#include <cmath>

class AMSGrad {
private:
    // hyperparams
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double time_step;

    // filter moments
    std::vector<std::vector<std::vector<std::vector<double>>>> m_filters; // first moment
    std::vector<std::vector<std::vector<std::vector<double>>>> v_filters; // second moment (gradient squared)
    std::vector<std::vector<std::vector<std::vector<double>>>> v_hat_filters; // max of v (AMSGrad specific)

    // bias moments
    std::vector<double> m_biases; // first moment
    std::vector<double> v_biases; // second moment (gradient squared)
    std::vector<double> v_hat_biases; // max of v (AMSGrad specific)

public:
    AMSGrad(double lr = 1e-3, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);

    // initialize states based on parameters
    void initialize(const ConvolutionLayer& layer);

    // update the layer parameters
    void update(ConvolutionLayer& layer, const std::vector<std::vector<std::vector<std::vector<double>>>>& dFilters, const std::vector<double>& dBiases);
};


#endif //INC_12_FINALPROJ_2_AMSGRAD_H
