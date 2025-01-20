//
// Created by Vijay Goyal on 2025-01-08.
//

#include "AMSGrad.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

template <typename Type>
AMSGrad<Type>::AMSGrad(double lr, double b1, double b2,
                       double eps, double wd)
        : learning_rate(lr),
          beta1(b1),
          beta2(b2),
          epsilon(eps),
          weight_decay(wd),
          time_step(0)
{}


// Initialize optimizer state for a ConvolutionLayer
template <typename Type>
void AMSGrad<Type>::initializeConv(const ConvolutionLayer<Type> &layer) {
    auto* layer_ptr = const_cast<ConvolutionLayer<Type>*>(&layer); // Remove constness for map key
    if(conv_states.find(layer_ptr) != conv_states.end()) {
        // Already initialized
        return;
    }

    int out_channels  = static_cast<int>(layer.filters.size());
    if(out_channels == 0) return;
    int in_channels   = static_cast<int>(layer.filters.at(0).size());
    int filter_height = static_cast<int>(layer.filters.at(0).at(0).size());
    int filter_width  = static_cast<int>(layer.filters.at(0).at(0).at(0).size());

    ConvLayerState state;
    state.m_filters.resize(out_channels,
                           std::vector<std::vector<std::vector<Type>>>(
                               in_channels, std::vector<std::vector<Type>>(
                                   filter_height, std::vector<Type>(
                                       filter_width, static_cast<Type>(0.0)))));

    state.v_filters = state.m_filters;
    state.v_hat_filters = state.m_filters;

    state.m_biases.resize(out_channels, static_cast<Type>(0.0));
    state.v_biases.resize(out_channels, static_cast<Type>(0.0));
    state.v_hat_biases.resize(out_channels, static_cast<Type>(0.0));

    conv_states[layer_ptr] = state;
}

// Update optimizer state for a ConvolutionLayer
template <typename Type>
void AMSGrad<Type>::update(ConvolutionLayer<Type> &layer,
                           const std::vector<std::vector<std::vector<std::vector<Type>>>> &dFilters,
                           const std::vector<Type> &dBiases) {
    auto* layer_ptr = &layer;
    // Initialize if not already done
    if(conv_states.find(layer_ptr) == conv_states.end()) {
        initializeConv(layer);
    }

    auto &state = conv_states[layer_ptr];
    time_step++;

    int out_channels = static_cast<int>(layer.filters.size());
    if(out_channels == 0) return;
    int in_channels  = static_cast<int>(layer.filters.at(0).size());
    int filter_height = static_cast<int>(layer.filters.at(0).at(0).size());
    int filter_width  = static_cast<int>(layer.filters.at(0).at(0).at(0).size());

    // Pre-validate vector sizes before entering parallel region
    if(dFilters.size() < static_cast<size_t>(out_channels)) {
        throw std::out_of_range("dFilters size is smaller than out_channels");
    }
    for(int f = 0; f < out_channels; ++f) {
        if(dFilters.at(f).size() < static_cast<size_t>(in_channels)) {
            throw std::out_of_range("dFilters inner size is smaller than in_channels");
        }
        for(int c = 0; c < in_channels; ++c) {
            if(dFilters.at(f).at(c).size() < static_cast<size_t>(filter_height)) {
                throw std::out_of_range("dFilters height size is smaller than filter_height");
            }
            for(int h = 0; h < filter_height; ++h) {
                if(dFilters.at(f).at(c).at(h).size() < static_cast<size_t>(filter_width)) {
                    throw std::out_of_range("dFilters width size is smaller than filter_width");
                }
            }
        }
    }

    if(dBiases.size() < static_cast<size_t>(out_channels)) {
        throw std::out_of_range("dBiases size is smaller than out_channels");
    }

    // Ensure bias vectors are properly sized
    if(state.m_biases.size() < static_cast<size_t>(out_channels)) {
        state.m_biases.resize(out_channels, static_cast<Type>(0.0));
    }
    if(state.v_biases.size() < static_cast<size_t>(out_channels)) {
        state.v_biases.resize(out_channels, static_cast<Type>(0.0));
    }
    if(state.v_hat_biases.size() < static_cast<size_t>(out_channels)) {
        state.v_hat_biases.resize(out_channels, static_cast<Type>(0.0));
    }

    // Shared variables for error handling
    bool has_error = false;
    std::string error_message = "";

    // Parallelize over the out_channels
    #pragma omp parallel for collapse(1) shared(has_error, error_message)
    for (int f = 0; f < out_channels; ++f) {
        try {
            for (int c = 0; c < in_channels; ++c) {
                for (int h = 0; h < filter_height; ++h) {
                    for (int w = 0; w < filter_width; ++w) {
                        Type g = dFilters.at(f).at(c).at(h).at(w);
                        // Adam moments
                        state.m_filters.at(f).at(c).at(h).at(w) = static_cast<Type>(beta1 * state.m_filters.at(f).at(c).at(h).at(w) + (1 - beta1) * g);
                        state.v_filters.at(f).at(c).at(h).at(w) = static_cast<Type>(beta2 * state.v_filters.at(f).at(c).at(h).at(w) + (1 - beta2) * (g * g));
                        // AMSGrad
                        state.v_hat_filters.at(f).at(c).at(h).at(w) = std::max(state.v_hat_filters.at(f).at(c).at(h).at(w), state.v_filters.at(f).at(c).at(h).at(w));

                        // Bias corrections
                        Type m_hat = static_cast<Type>(state.m_filters.at(f).at(c).at(h).at(w) / (1 - std::pow(beta1, time_step)));
                        Type v_hat_corr = static_cast<Type>(state.v_hat_filters.at(f).at(c).at(h).at(w) / (1 - std::pow(beta2, time_step)));

                        // Decoupled weight decay
                        layer.filters.at(f).at(c).at(h).at(w) *= static_cast<Type>((1.0 - learning_rate * weight_decay));

                        // Final update
                        layer.filters.at(f).at(c).at(h).at(w) -= static_cast<Type>(learning_rate * (m_hat / (std::sqrt(v_hat_corr) + epsilon)));
                    }
                }
            }

            // Update biases with bounds checking
            Type gb = dBiases.at(f);
            state.m_biases.at(f) = static_cast<Type>(beta1 * state.m_biases.at(f) + (1 - beta1) * gb);
            state.v_biases.at(f) = static_cast<Type>(beta2 * state.v_biases.at(f) + (1 - beta2) * (gb * gb));
            state.v_hat_biases.at(f) = std::max(state.v_hat_biases.at(f), state.v_biases.at(f));

            Type m_hat_b = static_cast<Type>(state.m_biases.at(f) / (1 - std::pow(beta1, time_step)));
            Type v_hat_corr_b = static_cast<Type>(state.v_hat_biases.at(f) / (1 - std::pow(beta2, time_step)));

            // Decoupled weight decay for bias (often zero)
            layer.biases.at(f) *= static_cast<Type>((1.0 - learning_rate * weight_decay));

            // Final bias update
            layer.biases.at(f) -= static_cast<Type>(learning_rate * (m_hat_b / (std::sqrt(v_hat_corr_b) + epsilon)));
        }
        catch (const std::out_of_range& e) {
            #pragma omp critical
            {
                if (!has_error) {
                    has_error = true;
                    error_message = e.what();
                }
            }
        }
    }

    if(has_error) {
        throw std::runtime_error(error_message);
    }
}

// Initialize optimizer state for a FullyConnectedLayer
template <typename Type>
void AMSGrad<Type>::initializeFC(const FullyConnectedLayer<Type> &layer) {
    auto* layer_ptr = const_cast<FullyConnectedLayer<Type>*>(&layer); // Remove constness for map key
    if(fc_states.find(layer_ptr) != fc_states.end()) {
        // Already initialized
        return;
    }

    int out_features = static_cast<int>(layer.out_features);
    int in_features  = static_cast<int>(layer.in_features);

    FCLayerState state;
    state.m_weights.resize(out_features, std::vector<Type>(in_features, static_cast<Type>(0.0)));
    state.v_weights = state.m_weights;
    state.v_hat_weights = state.m_weights;

    state.m_biases.resize(out_features, static_cast<Type>(0.0));
    state.v_biases.resize(out_features, static_cast<Type>(0.0));
    state.v_hat_biases.resize(out_features, static_cast<Type>(0.0));

    fc_states[layer_ptr] = state;
}

// Update optimizer state for a FullyConnectedLayer
template <typename Type>
void AMSGrad<Type>::update(FullyConnectedLayer<Type> &layer,
                           const std::vector<std::vector<Type>> &dWeights,
                           const std::vector<Type> &dBiases) {
    auto* layer_ptr = &layer;
    // Initialize if not already done
    if(fc_states.find(layer_ptr) == fc_states.end()) {
        initializeFC(layer);
    }

    auto &state = fc_states[layer_ptr];
    time_step++;

    int out_features = static_cast<int>(layer.out_features);
    int in_features  = static_cast<int>(layer.in_features);

    // Pre-validate vector sizes before entering parallel region
    if(dWeights.size() < static_cast<size_t>(out_features)) {
        throw std::out_of_range("dWeights size is smaller than out_features");
    }
    for(int i = 0; i < out_features; ++i) {
        if(dWeights.at(i).size() < static_cast<size_t>(in_features)) {
            throw std::out_of_range("dWeights inner size is smaller than in_features");
        }
    }

    if(dBiases.size() < static_cast<size_t>(out_features)) {
        throw std::out_of_range("dBiases size is smaller than out_features");
    }

    // Ensure bias vectors are properly sized
    if(state.m_biases.size() < static_cast<size_t>(out_features)) {
        state.m_biases.resize(out_features, static_cast<Type>(0.0));
    }
    if(state.v_biases.size() < static_cast<size_t>(out_features)) {
        state.v_biases.resize(out_features, static_cast<Type>(0.0));
    }
    if(state.v_hat_biases.size() < static_cast<size_t>(out_features)) {
        state.v_hat_biases.resize(out_features, static_cast<Type>(0.0));
    }

    // Shared variables for error handling
    bool has_error = false;
    std::string error_message = "";

    // Parallelize over the out_features
    #pragma omp parallel for collapse(1) shared(has_error, error_message)
    for(int i = 0; i < out_features; ++i) {
        try {
            for(int j = 0; j < in_features; ++j) {
                Type g = dWeights.at(i).at(j);
                // Adam moments
                state.m_weights.at(i).at(j) = static_cast<Type>(beta1 * state.m_weights.at(i).at(j) + (1 - beta1) * g);
                state.v_weights.at(i).at(j) = static_cast<Type>(beta2 * state.v_weights.at(i).at(j) + (1 - beta2) * (g * g));
                // AMSGrad
                state.v_hat_weights.at(i).at(j) = std::max(state.v_hat_weights.at(i).at(j), state.v_weights.at(i).at(j));

                Type m_hat = static_cast<Type>(state.m_weights.at(i).at(j) / (1 - std::pow(beta1, time_step)));
                Type v_hat_corr = static_cast<Type>(state.v_hat_weights.at(i).at(j) / (1 - std::pow(beta2, time_step)));

                // Decoupled weight decay
                layer.weights.at(i).at(j) *= static_cast<Type>((1.0 - learning_rate * weight_decay));

                // Final update
                layer.weights.at(i).at(j) -= static_cast<Type>(learning_rate * (m_hat / (std::sqrt(v_hat_corr) + epsilon)));
            }

            // Update biases with bounds checking
            Type gb = dBiases.at(i);
            state.m_biases.at(i) = static_cast<Type>(beta1 * state.m_biases.at(i) + (1 - beta1) * gb);
            state.v_biases.at(i) = static_cast<Type>(beta2 * state.v_biases.at(i) + (1 - beta2) * (gb * gb));
            state.v_hat_biases.at(i) = std::max(state.v_hat_biases.at(i), state.v_biases.at(i));

            Type m_hat_b = static_cast<Type>(state.m_biases.at(i) / (1 - std::pow(beta1, time_step)));
            Type v_hat_corr_b = static_cast<Type>(state.v_hat_biases.at(i) / (1 - std::pow(beta2, time_step)));

            // Decoupled weight decay for bias
            layer.biases.at(i) *= static_cast<Type>((1.0 - learning_rate * weight_decay));

            // Final bias update
            layer.biases.at(i) -= static_cast<Type>(learning_rate * (m_hat_b / (std::sqrt(v_hat_corr_b) + epsilon)));
        }
        catch (const std::out_of_range& e) {
            #pragma omp critical
            {
                if (!has_error) {
                    has_error = true;
                    error_message = e.what();
                }
            }
        }
    }

    if(has_error) {
        throw std::runtime_error(error_message);
    }
}