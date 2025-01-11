//
// Created by Vijay Goyal on 2025-01-09.
//


#ifndef INC_12_FINALPROJ_2_MODULARCNN_H
#define INC_12_FINALPROJ_2_MODULARCNN_H

#include <vector>
#include <string>
#include <memory>
#include "../tools/LayerConfig.h"
#include "../layers/ConvolutionLayer.h"
#include "../layers/MaxPoolingLayer.h"
#include "../layers/FullyConnectedLayer.h"
#include "../tools/ComputationGraph.h"
#include "../tools/ConvolutionOperation.h"
#include "../tools/MaxPoolingOperation.h"
#include "../tools/FullyConnectedOperation.h"
#include "../tools/Tensor.h"

/**
 * @brief A fully modular CNN class that allows specifying an arbitrary sequence
 *        of convolution, pooling, and fully connected layers. The user can define
 *        the structure (layer types + shapes) up front, then we build a
 *        ComputationGraph from it.
 */
template <typename Type>
class ModularCNN {
private:
    // store the sequence of layers
    std::vector<std::string> layerTypes; // "conv", "pool", "fc", etc.

public:
    std::vector<ConvolutionLayer<Type>> convLayers;
    std::vector<MaxPoolingLayer<Type>> poolLayers;
    std::vector<FullyConnectedLayer<Type>> fcLayers;

    ComputationGraph<Type> graph;

    ModularCNN(const std::vector<LayerConfig>& configs);

    void buildGraph();

    std::shared_ptr<Tensor<Type>> forward(const std::shared_ptr<Tensor<Type>>& input);

    void zeroGrad();

    size_t getTotalParams() const;
};

#include "ModularCNN.tpp"

#endif //INC_12_FINALPROJ_2_MODULARCNN_H
