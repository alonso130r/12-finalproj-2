//
// Created by Vijay Goyal on 2025-01-09.
//

#include "ModularCNN.h"
#include <stdexcept>
#include <iostream>

template <typename Type>
ModularCNN<Type>::ModularCNN(const std::vector<LayerConfig>& configs) {
    // parse each config, create the corresponding layer object, store in 'layers' as a shared_ptr<Layer<Type>>, and track the type string in layerTypes
    for(const auto &cfg : configs) {
        if(cfg.type == "conv") {
            auto conv = std::make_shared<ConvolutionLayer<Type>>(
                    cfg.in_channels,
                    cfg.out_channels,
                    cfg.filter_height,
                    cfg.filter_width,
                    cfg.stride,
                    cfg.padding
            );
            layers.push_back(conv);
            layerTypes.emplace_back("conv");
        }
        else if(cfg.type == "pool") {
            auto pool = std::make_shared<MaxPoolingLayer<Type>>(
                    cfg.pool_height,
                    cfg.pool_width,
                    cfg.stride,
                    cfg.padding
            );
            layers.push_back(pool);
            layerTypes.emplace_back("pool");
        }
        else if(cfg.type == "fc") {
            auto fc = std::make_shared<FullyConnectedLayer<Type>>(
                    cfg.in_features,
                    cfg.out_features
            );
            layers.push_back(fc);
            layerTypes.emplace_back("fc");
        }
        else {
            throw std::runtime_error("Unknown layer type: " + cfg.type);
        }
    }

    // automatically build the graph
    buildGraph();
}

template <typename Type>
void ModularCNN<Type>::buildGraph() {
    // clear existing ops
    graph = ComputationGraph<Type>();

    // iterate over layers in order
    for(std::size_t i = 0; i < layers.size(); ++i) {
        std::string t = layerTypes[i];
        if(t == "conv") {
            // dynamic_cast to ConvolutionLayer<Type>*
            auto convPtr = std::dynamic_pointer_cast<ConvolutionLayer<Type>>(layers[i]);
            if(!convPtr) {
                throw std::runtime_error("Failed dynamic_cast to ConvolutionLayer in buildGraph");
            }
            graph.addOperation(std::make_shared<ConvolutionOperation<Type>>(*convPtr));
        } else if (t == "pool") {
            // dynamic_cast to MaxPoolingLayer<Type>*
            auto poolPtr = std::dynamic_pointer_cast<MaxPoolingLayer<Type>>(layers[i]);
            if(!poolPtr) {
                throw std::runtime_error("Failed dynamic_cast to MaxPoolingLayer in buildGraph");
            }
            graph.addOperation(std::make_shared<MaxPoolingOperation<Type>>(
                    poolPtr->pool_height, poolPtr->pool_width, poolPtr->stride, poolPtr->padding));
        } else if (t == "fc") {
            // dynamic_cast to FullyConnectedLayer<Type>*
            auto fcPtr = std::dynamic_pointer_cast<FullyConnectedLayer<Type>>(layers[i]);
            if(!fcPtr) {
                throw std::runtime_error("Failed dynamic_cast to FullyConnectedLayer in buildGraph");
            }
            graph.addOperation(std::make_shared<FullyConnectedOperation<Type>>(*fcPtr));
        } else {
            throw std::runtime_error("Unknown layer type in buildGraph: " + t);
        }
    }
}

template <typename Type>
std::shared_ptr<Tensor<Type>> ModularCNN<Type>::forward(const std::shared_ptr<Tensor<Type>>& input) {
    return graph.forward(input);
}

template <typename Type>
void ModularCNN<Type>::zeroGrad() {
    // loop over all layers in 'layers'
    for(auto &layerPtr : layers) {
        layerPtr->zeroGrad();
    }
}

template <typename Type>
ssize_t ModularCNN<Type>::getTotalParams() const {
    ssize_t total = 0;
    for(const auto &layerPtr : layers) {
        total += layerPtr->getNumParams();
    }
    return total;
}

template <typename Type>
void ModularCNN<Type>::saveWeights(const std::string path) {
    std::vector<WeightStruct<Type>> weights;

    for(auto &layerPtr : layers) {
        weights.push_back(layerPtr->saveWeights());
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);

    if (!file) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }

    uint32_t count = static_cast<uint32_t>(weights.size());
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto &obj : weights) {
        uint32_t typeVal = static_cast<uint32_t>(obj->getType());
        file.write(reinterpret_cast<const char*>(&typeVal), sizeof(typeVal));
    }
}
