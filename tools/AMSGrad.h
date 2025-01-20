//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_AMSGRAD_H
#define INC_12_FINALPROJ_2_AMSGRAD_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "../layers/ConvolutionLayer.h"
#include "../layers/FullyConnectedLayer.h"

/**
 * @brief AdamW + AMSGrad optimizer.
 *        - Adam with decoupled weight decay (AdamW)
 *        - AMSGrad ensures a non-decreasing second-moment estimate v_hat.
 *        - TIME COMPLEXITY: O(n) for n parameters
 *            - There are n parameters to update, and each update requires O(1) operations, although there are up to 5 nested for-loops,
 *           these simply iterate over the dimensions of the weights/filters and biases, and do not depend on the number of parameters.
 *         - SPACE COMPLEXITY: O(n) for n parameters
 *              - The optimizer maintains 3 states: m, v, and v_hat, each of which requires the same amount of memory as the parameters being optimized.
 *              - Each of these states is stored in a separate map, which requires O(n) memory.
 *              - So technically, the space complexity is O(3n) -> O(n).
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
 
     // Structures to hold optimizer state for Convolutional Layers
     struct ConvLayerState {
         std::vector<std::vector<std::vector<std::vector<Type>>>> m_filters;     // first moment
         std::vector<std::vector<std::vector<std::vector<Type>>>> v_filters;     // second moment
         std::vector<std::vector<std::vector<std::vector<Type>>>> v_hat_filters; // max of v
 
         std::vector<Type> m_biases;
         std::vector<Type> v_biases;
         std::vector<Type> v_hat_biases;
     };
 
     // Structures to hold optimizer state for Fully Connected Layers
     struct FCLayerState {
         std::vector<std::vector<Type>> m_weights;     // first moment
         std::vector<std::vector<Type>> v_weights;     // second moment
         std::vector<std::vector<Type>> v_hat_weights; // max of v
 
         std::vector<Type> m_biases;
         std::vector<Type> v_biases;
         std::vector<Type> v_hat_biases;
     };
 
     // Maps to associate layers with their optimizer states
     std::unordered_map<ConvolutionLayer<Type>*, ConvLayerState> conv_states;
     std::unordered_map<FullyConnectedLayer<Type>*, FCLayerState> fc_states;
 
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
