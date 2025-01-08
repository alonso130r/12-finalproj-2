//
// Created by Vijay Goyal on 2025-01-08.
//

#include "AMSGrad.h"

AMSGrad::AMSGrad(double lr, double b1, double b2, double eps) : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void AMSGrad::initialize(const ConvolutionLayer &layer) {
    int out_channels = layer.filters.size();
    int in_channels = layer.filters[0].size();
    int filter_height = layer.filters[0][0].size();
    int filter_width = layer.filters[0][0][0].size();

    // initialize filters
    m_filters.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels,
            std::vector<std::vector<double>>(filter_height, std::vector<double>(filter_width, 0.0))));
    v_filters.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels,
            std::vector<std::vector<double>>(filter_height, std::vector<double>(filter_width, 0.0))));
    v_hat_filters.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels,
            std::vector<std::vector<double>>(filter_height, std::vector<double>(filter_width, 0.0))));

    // initialize biases
    m_biases.resize(out_channels, std::vector<double>(1, 0.0));
    v_biases.resize(out_channels, std::vector<double>(1, 0.0));
    v_hat_biases.resize(out_channels, std::vector<double>(1, 0.0));
}

void AMSGrad::update(ConvolutionLayer &layer,
                     const std::vector <std::vector<std::vector < std::vector < double>>>> &dFilters,
                     const std::vector<double> &dBiases) {
    if (m_filters.empty()) {
        initialize(layer);
    }
    t++;

    int out_channels = layer.filters.size();
    int in_channels = layer.filters[0].size();
    int filter_height = layer.filters[0][0].size();
    int filter_width = layer.filters[0][0][0].size();

    for(int f = 0; f < out_channels; ++f) {
        for(int c = 0; c < in_channels; ++c) {
            for(int h = 0; h < filter_height; ++h) {
                for(int w = 0; w < filter_width; ++w) {
                    // update first moment
                    m_filters[f][c][h][w] = beta1 * m_filters[f][c][h][w] + (1 - beta1) * dFilters[f][c][h][w];
                    // update second moment
                    v_filters[f][c][h][w] = beta2 * v_filters[f][c][h][w] + (1 - beta2) * dFilters[f][c][h][w] * dFilters[f][c][h][w];
                    // update v_max
                    if(v_filters[f][c][h][w] > v_hat_filters[f][c][h][w]) {
                        v_hat_filters[f][c][h][w] = v_filters[f][c][h][w];
                    }
                    // compute bias-corrected first moment
                    double m_hat = m_filters[f][c][h][w] / (1 - std::pow(beta1, t));
                    // compute parameter update
                    layer.filters[f][c][h][w] -= learning_rate * m_hat / (std::sqrt(v_hat_filters[f][c][h][w]) + epsilon);
                }
            }
        }
        m_biases[f][0] = beta1 * m_biases[f][0] + (1 - beta1) * dBiases[f];
        v_biases[f][0] = beta2 * v_biases[f][0] + (1 - beta2) * dBiases[f] * dBiases[f];
        if(v_biases[f][0] > v_hat_biases[f][0]) {
            v_hat_biases[f][0] = v_biases[f][0];
        }
        double m_hat_bias = m_biases[f][0] / (1 - std::pow(beta1, t));
        layer.biases[f] -= learning_rate * m_hat_bias / (std::sqrt(v_hat_biases[f][0]) + epsilon);
    }
}
