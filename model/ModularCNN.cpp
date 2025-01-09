//
// Created by Vijay Goyal on 2025-01-09.
//

#include "ModularCNN.h"
#include <stdexcept>
#include <iostream>

ModularCNN::ModularCNN(const std::vector<LayerConfig>& configs) {
    // parse each config and create the corresponding internal layer
    size_t convCount = 0;
    size_t poolCount = 0;
    size_t fcCount   = 0;

    for(const auto &cfg : configs) {
        if(cfg.type == "conv") {
            // create a ConvolutionLayer
            convLayers.emplace_back(cfg.in_channels,
                                    cfg.out_channels,
                                    cfg.filter_height,
                                    cfg.filter_width,
                                    cfg.stride,
                                    cfg.padding);
            layerTypes.push_back("conv");
            convCount++;
        }
        else if(cfg.type == "pool") {
            // create a MaxPoolingLayer
            poolLayers.emplace_back(cfg.pool_height,
                                    cfg.pool_width,
                                    cfg.stride,
                                    cfg.padding);
            layerTypes.push_back("pool");
            poolCount++;
        }
        else if(cfg.type == "fc") {
            // create a FullyConnectedLayer
            fcLayers.emplace_back(cfg.in_features, cfg.out_features);
            layerTypes.push_back("fc");
            fcCount++;
        }
        else {
            throw std::runtime_error("Unknown layer type: " + cfg.type);
        }
    }
    buildGraph();
}

void ModularCNN::buildGraph() {
    // clear existing ops in case we're re-building
    graph = ComputationGraph();

    size_t convIndex = 0;
    size_t poolIndex = 0;
    size_t fcIndex   = 0;

    // build in the order of layerTypes
    for(const auto & t : layerTypes) {
        if(t == "conv") {
            // add ConvolutionOperation
            if(convIndex >= convLayers.size()) {
                throw std::runtime_error("Not enough convLayers for 'conv' in buildGraph");
            }
            graph.addOperation(std::make_shared<ConvolutionOperation>(convLayers[convIndex]));
            convIndex++;
        }
        else if(t == "pool") {
            if(poolIndex >= poolLayers.size()) {
                throw std::runtime_error("Not enough poolLayers for 'pool'");
            }
            auto &pl = poolLayers[poolIndex];
            graph.addOperation(std::make_shared<MaxPoolingOperation>(
                    pl.pool_height, pl.pool_width, pl.stride, pl.padding));
            poolIndex++;
        }
        else if(t == "fc") {
            if(fcIndex >= fcLayers.size()) {
                throw std::runtime_error("Not enough fcLayers for 'fc'");
            }
            graph.addOperation(std::make_shared<FullyConnectedOperation>(fcLayers[fcIndex]));
            fcIndex++;
        }
        else {
            throw std::runtime_error("Unknown layer type in buildGraph: " + t);
        }
    }
}

std::shared_ptr<Tensor> ModularCNN::forward(const std::shared_ptr<Tensor>& input) {
    return graph.forward(input);
}

void ModularCNN::zeroGrad() {
    // zero conv
    for (auto &c: convLayers) {
        c.zeroGrad();
    }
    // zero fc
    for (auto &f: fcLayers) {
        f.zeroGrad();
    }
}

size_t ModularCNN::getTotalParams() const {
    size_t total = 0;
    // conv
    for(const auto &c : convLayers) {
        total += c.getNumParams();
    }
    // fc
    for(const auto &f : fcLayers) {
        total += f.getNumParams();
    }
    return total;
}
