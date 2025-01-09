//
// Created by Vijay Goyal on 2025-01-08.
//

#include "AMSGrad.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

AMSGrad::AMSGrad(double lr, double b1, double b2,
                           double eps, double wd)
        : learning_rate(lr),
          beta1(b1),
          beta2(b2),
          epsilon(eps),
          weight_decay(wd),
          time_step(0),
          initializedConv(false),
          initializedFC(false)
{}

void AMSGrad::initializeConv(const ConvolutionLayer &layer) {
    int out_channels  = layer.filters.size();
    if(out_channels == 0) return;
    int in_channels   = layer.filters[0].size();
    int filter_height = layer.filters[0][0].size();
    int filter_width  = layer.filters[0][0][0].size();

    m_filters.resize(out_channels,
                     std::vector<std::vector<std::vector<double>>>(
            in_channels, std::vector<std::vector<double>>(
                    filter_height, std::vector<double>(filter_width, 0.0))));
    v_filters  = m_filters;
    v_hat_filters = m_filters;

    m_biases.resize(out_channels, 0.0);
    v_biases.resize(out_channels, 0.0);
    v_hat_biases.resize(out_channels, 0.0);

    initializedConv = true;
}

void AMSGrad::initializeFC(const FullyConnectedLayer &layer) {
    int out_features = layer.out_features;
    int in_features  = layer.in_features;

    m_fc_weights.resize(out_features, std::vector<double>(in_features, 0.0));
    v_fc_weights = m_fc_weights;
    v_hat_fc_weights = m_fc_weights;

    m_fc_biases.resize(out_features, 0.0);
    v_fc_biases.resize(out_features, 0.0);
    v_hat_fc_biases.resize(out_features, 0.0);

    initializedFC = true;
}

/**
 * @brief AdamWAMSGrad update for FullyConnectedLayer
 */
void AMSGrad::update(ConvolutionLayer &layer,
                     const std::vector <std::vector<std::vector < std::vector < double>>>> &dFilters,
                     const std::vector<double> &dBiases) {
    if(!initializedConv) {
        initializeConv(layer);
    }
    time_step++;

    int out_channels = layer.filters.size();
    if(out_channels == 0) return;
    int in_channels  = layer.filters[0].size();
    int filter_height = layer.filters[0][0].size();
    int filter_width  = layer.filters[0][0][0].size();

    // param update
    for (int f = 0; f < out_channels; ++f) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < filter_height; ++h) {
                for (int w = 0; w < filter_width; ++w) {
                    double g = dFilters[f][c][h][w];
                    // Adam moments
                    m_filters[f][c][h][w] = beta1*m_filters[f][c][h][w] + (1 - beta1)*g;
                    v_filters[f][c][h][w] = beta2*v_filters[f][c][h][w] + (1 - beta2)*(g*g);
                    // AMSGrad
                    v_hat_filters[f][c][h][w] =
                    std::max(v_hat_filters[f][c][h][w], v_filters[f][c][h][w]);

                    // bias corrections
                    double m_hat = m_filters[f][c][h][w] / (1 - std::pow(beta1, time_step));
                    double v_hat_corr = v_hat_filters[f][c][h][w] / (1 - std::pow(beta2, time_step));

                    // decoupled weight decay: param *= (1 - lr*wd)
                    layer.filters[f][c][h][w] *= (1.0 - learning_rate*weight_decay);

                    // final update
                    layer.filters[f][c][h][w] -= learning_rate * (m_hat / (std::sqrt(v_hat_corr) + epsilon));
                }
            }
        }
        // bias
        double gb = dBiases[f];
        m_biases[f] = beta1*m_biases[f] + (1 - beta1)*gb;
        v_biases[f] = beta2*v_biases[f] + (1 - beta2)*gb*gb;
        v_hat_biases[f] = std::max(v_hat_biases[f], v_biases[f]);

        double m_hat_b = m_biases[f] / (1 - std::pow(beta1, time_step));
        double v_hat_corr_b = v_hat_biases[f] / (1 - std::pow(beta2, time_step));

        // decoupled weight decay for bias (often zero):
        layer.biases[f] *= (1.0 - learning_rate*weight_decay);

        layer.biases[f] -= learning_rate * (m_hat_b / (std::sqrt(v_hat_corr_b) + epsilon));
    }
}

/**
 * @brief AdamWAMSGrad update for FullyConnectedLayer
 */
void AMSGrad::update(FullyConnectedLayer &layer,
                     const std::vector <std::vector<double>> &dWeights,
                     const std::vector<double> &dBiases) {
    if(!initializedFC) {
        initializeFC(layer);
    }
    time_step++;

    int out_features = layer.out_features;
    int in_features  = layer.in_features;

    for(int i = 0; i < out_features; ++i) {
        for(int j = 0; j < in_features; ++j) {
            double g = dWeights[i][j];
            // Adam moments
            m_fc_weights[i][j] = beta1*m_fc_weights[i][j] + (1 - beta1)*g;
            v_fc_weights[i][j] = beta2*v_fc_weights[i][j] + (1 - beta2)*(g*g);
            // AMSGrad
            v_hat_fc_weights[i][j] = std::max(v_hat_fc_weights[i][j], v_fc_weights[i][j]);

            double m_hat = m_fc_weights[i][j] / (1 - std::pow(beta1, time_step));
            double v_hat_corr = v_hat_fc_weights[i][j] / (1 - std::pow(beta2, time_step));

            // decoupled weight decay
            layer.weights[i][j] *= (1.0 - learning_rate*weight_decay);

            // final update
            layer.weights[i][j] -= learning_rate*(m_hat / (std::sqrt(v_hat_corr) + epsilon));
        }
        double gb = dBiases[i];
        m_fc_biases[i] = beta1*m_fc_biases[i] + (1 - beta1)*gb;
        v_fc_biases[i] = beta2*v_fc_biases[i] + (1 - beta2)*gb*gb;
        v_hat_fc_biases[i] = std::max(v_hat_fc_biases[i], v_fc_biases[i]);

        double m_hat_b = m_fc_biases[i] / (1 - std::pow(beta1, time_step));
        double v_hat_corr_b = v_hat_fc_biases[i] / (1 - std::pow(beta2, time_step));

        // decoupled weight decay for bias
        layer.biases[i] *= (1.0 - learning_rate*weight_decay);

        layer.biases[i] -= learning_rate*(m_hat_b / (std::sqrt(v_hat_corr_b) + epsilon));
    }
}