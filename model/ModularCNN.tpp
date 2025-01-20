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

template<typename Type>
ModularCNN<Type>::ModularCNN(const std::string path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr<< "Error opening file" << std::endl;
        return;
    }
    uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t typeVal = 0;
        file.read(reinterpret_cast<char*>(&typeVal), sizeof(typeVal));

        switch (auto type = static_cast<WeightStructType>(typeVal)) {
            case WeightStructType::ConvolutionalWeights:
                layers.emplace_back(ConvolutionalWeights<Type>::deserialize(file));
                layerTypes.emplace_back("conv");
                break;
            case WeightStructType::ConnectedWeights:
                layers.emplace_back(ConnectedWeights<Type>::deserialize(file));
                layerTypes.emplace_back("fc");
                break;
            case WeightStructType::PoolingWeights:
                layers.emplace_back(PoolingWeights<Type>::deserialize(file));
                layerTypes.emplace_back("pool");
                break;
        }
    }
    buildGraph();
}


template <typename Type>
std::shared_ptr<Tensor<Type>> ModularCNN<Type>::forward(const std::shared_ptr<Tensor<Type>>& input) {
    auto output = graph.forward(input);

    // Process each batch
    for (auto& batch : output->data) {
        // Extract logits
        std::vector<Type> logits(batch.size());
        for (size_t i = 0; i < batch.size(); i++) {
            logits[i] = batch[i][0][0];
        }

        // Find max for numerical stability
        Type maxVal = *std::max_element(logits.begin(), logits.end());
        
        // Scale values to prevent overflow
        std::vector<Type> scaled(logits.size());
        Type sum = 0;
        for (size_t i = 0; i < logits.size(); i++) {
            // Add small epsilon to prevent division by zero
            scaled[i] = std::exp((logits[i] - maxVal) / Type(100.0) + Type(1e-7));
            sum += scaled[i];
        }

        // Normalize and store back
        for (size_t i = 0; i < batch.size(); i++) {
            batch[i][0][0] = scaled[i] / sum;
        }
    }

    return output;
}

template <typename Type>
int ModularCNN<Type>::forwards(const std::shared_ptr<Tensor<Type>>& input) {
    auto output = graph.forward(input);
    int maxIndex = 0;
    for (int i = 0; i < output->data.size(); i++) {
        if (output->data[i][0][0][0] > output->data[maxIndex][0][0][0]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

template <typename Type>
void ModularCNN<Type>::backward(const std::shared_ptr<Tensor<Type>>& dOut) {
    graph.backward(dOut);
}

template <typename Type>
void ModularCNN<Type>::update(AMSGrad<Type>& optimizer) {
    int index = 0;
    for(auto &layerPtr : layers) {
        if (layerTypes[index] == "conv") {
            auto *temp = dynamic_cast<ConvolutionLayer<Type>*>(layerPtr.get());
            if (!temp) {
                throw std::invalid_argument("Layer cast to ConvolutionLayer failed.");
            }

            // Verify dBiases size matches out_channels
            size_t out_channels = temp->filters.size();
            if (temp->dBiases.size() != out_channels) {
                throw std::out_of_range(
                    "dBiases size (" + std::to_string(temp->dBiases.size()) +
                    ") does not match out_channels (" + std::to_string(out_channels) + ") in ConvolutionLayer."
                );
            }
            optimizer.update(*temp, temp->dFilters, temp->dBiases);
        }
        if (layerTypes[index] == "fc") {
            auto *temp = dynamic_cast<FullyConnectedLayer<Type>*>(layerPtr.get());
            if (!temp) {
                throw std::invalid_argument("Layer cast to FullyConnectedLayer failed.");
            }

            // Verify dBiases size matches out_features
            size_t out_features = temp->out_features;
            if (temp->dBiases.size() != out_features) {
                throw std::out_of_range(
                    "dBiases size (" + std::to_string(temp->dBiases.size()) +
                    ") does not match out_features (" + std::to_string(out_features) + ") in FullyConnectedLayer."
                );
            }
            optimizer.update(*temp, temp->dWeights, temp->dBiases);
        }
        index++;
    }
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
    std::vector<std::shared_ptr<WeightStruct<Type>>> weights;

    for(auto &layerPtr : layers) {
        weights.emplace_back(layerPtr->saveWeights());
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
        obj->serialize(file);
    }
    file.close();
}
