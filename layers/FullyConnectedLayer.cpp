//
// Created by Vijay Goyal on 2025-01-08.
//

#include "FullyConnectedLayer.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

FullyConnectedLayer::FullyConnectedLayer(int in_features, int out_features) : in_features(in_features), out_features(out_features) {
    initializeParams();
}

/*
 * Initialize the weights and biases with random values using He initialization
 */
void FullyConnectedLayer::initializeParams() {
    // calculate fan in and standard deviation
    double fan_in = static_cast<double>(in_features);
    double std_dev = sqrt(2.0 / fan_in);

    // initialize random generators (mersenne twister engine)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std_dev);

    // resize weights and biases
    weights.resize(out_features, std::vector<double>(in_features, 0.0));
    biases.resize(out_features, 0.0);

    // initialize weights with He initialization
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights[i][j] = dist(gen); // sampled from N(0, std_dev^2)
        }
        biases[i] = 0.0; // initialize biases to zero
    }

    // initialize gradients to zero
    dWeights.resize(out_features, std::vector<double>(in_features, 0.0));
    dBiases.resize(out_features, 0.0);
}

/*
 * Zero the gradients
 */
void FullyConnectedLayer::zeroGrad() {
    for(int i = 0; i < out_features; ++i) {
        std::fill(dWeights[i].begin(), dWeights[i].end(), 0.0);
    }
    std::fill(dBiases.begin(), dBiases.end(), 0.0);
}

/*
 * Get the number of parameters in the layer
 */
size_t FullyConnectedLayer::getNumParams() const {
    size_t wParams = (size_t)out_features * (size_t)in_features;
    size_t bParams = biases.size();
    return wParams + bParams;
}