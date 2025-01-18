//
// Created by Vijay Goyal on 2025-01-10.
//

#include <cstddef>
#include <string>
typedef size_t rsize_t;
#include <_string.h>
#include <sys/types.h>
#include "../tools/LayerConfig.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdfloat>
#include <vector>
#include "../tools/Tensor.h"
#include "../tools/ConvolutionalWeights.h"
#include "../tools/ConnectedWeights.h"
#include "../tools/PoolingWeights.h"
#include "../layers/Layer.h"
#include "../tools/WeightStruct.h"
#include "../layers/ConvolutionLayer.h"
#include "../layers/FullyConnectedLayer.h"

#include "../tools/Operation.h"
#include "../tools/MaxPoolingOperation.h"
#include "../tools/FullyConnectedOperation.h"
#include "../tools/ConvolutionOperation.h"
#include "../tools/ComputationGraph.h"

#include "../layers/MaxPoolingLayer.h"

#include "../model/ModularCNN.h"
#include "../tools/AMSGrad.h"
#include "../tools/CrossEntropy.h"


using bfloat = float;

using namespace pybind11;

PYBIND11_MODULE(ModularCNN, m) {
    m.doc() = "Modular CNN implementation in C++";

    class_<Tensor<bfloat>>(m, "Tensor")
            .def(init<int, int, int, int, bfloat>())
            .def(init<>())
            .def_readwrite("data", &Tensor<bfloat>::data)
            .def_readwrite("grad", &Tensor<bfloat>::grad)
            .def_readwrite("creator", &Tensor<bfloat>::creator)
            .def("zeroGrad", &Tensor<bfloat>::zeroGrad);

    class_<Layer<bfloat>>(m, "Layer")
        .def("getNumParams", &Layer<bfloat>::getNumParams)
        .def("zeroGrad", &Layer<bfloat>::zeroGrad)
        .def("saveWeights", &Layer<bfloat>::saveWeights);

    class_<Operation<bfloat>>(m, "Operation")
        .def_readwrite("inputs", &Operation<bfloat>::inputs)
        .def("forward", &Operation<bfloat>::forward)
        .def("backward", &Operation<bfloat>::backward);

    class_<CrossEntropy<bfloat>>(m, "CrossEntropy")
        .def(init<bool>())
        .def("forward", &CrossEntropy<bfloat>::forward)
        .def("backward", &CrossEntropy<bfloat>::backward);

    class_<ComputationGraph<bfloat>>(m, "ComputationGraph")
        .def(init<>())
        .def("addOperation", &ComputationGraph<bfloat>::addOperation)
        .def("forward", &ComputationGraph<bfloat>::forward)
        .def("backward", &ComputationGraph<bfloat>::backward);

    class_<ModularCNN<bfloat>>(m, "ModularCNN")
        .def(init<std::vector<LayerConfig>>())
        .def(init<std::string>())
        .def("buildGraph", &ModularCNN<bfloat>::buildGraph)
        .def("forward", &ModularCNN<bfloat>::forward)
        .def("forwards", &ModularCNN<bfloat>::forwards)
        .def("backward", &ModularCNN<bfloat>::backward)
        .def("update", &ModularCNN<bfloat>::update)
        .def("zeroGrad", &ModularCNN<bfloat>::zeroGrad)
        .def("saveWeights", &ModularCNN<bfloat>::saveWeights)
        .def("getTotalParams", &ModularCNN<bfloat>::getTotalParams);

    class_<ConvolutionLayer<bfloat>>(m, "ConvolutionLayer")
        .def(init<int, int, int, int, int, int>())
        .def_readwrite("in_channels", &ConvolutionLayer<bfloat>::in_channels)
        .def_readwrite("out_channels", &ConvolutionLayer<bfloat>::out_channels)
        .def_readwrite("filter_height", &ConvolutionLayer<bfloat>::filter_height)
        .def_readwrite("filter_width", &ConvolutionLayer<bfloat>::filter_width)
        .def_readwrite("filters", &ConvolutionLayer<bfloat>::filters)
        .def_readwrite("biases", &ConvolutionLayer<bfloat>::biases)
        .def_readwrite("dFilters", &ConvolutionLayer<bfloat>::dFilters)
        .def_readwrite("dBiases", &ConvolutionLayer<bfloat>::dBiases)
        .def("initializeFilters", &ConvolutionLayer<bfloat>::initializeFilters)
        .def("forward", &ConvolutionLayer<bfloat>::forward)
        .def("backward", &ConvolutionLayer<bfloat>::backward)
        .def("getNumParams", &ConvolutionLayer<bfloat>::getNumParams)
        .def("zeroGrad", &ConvolutionLayer<bfloat>::zeroGrad)
        .def("setFilters", &ConvolutionLayer<bfloat>::setFilters)
        .def("setBiases", &ConvolutionLayer<bfloat>::setBiases)
        .def("saveWeights", &ConvolutionLayer<bfloat>::saveWeights);

    class_<MaxPoolingLayer<bfloat>>(m, "MaxPoolingLayer")
        .def(init<int, int, int, int>())
        .def_readwrite("pool_height", &MaxPoolingLayer<bfloat>::pool_height)
        .def_readwrite("pool_width", &MaxPoolingLayer<bfloat>::pool_width)
        .def_readwrite("stride", &MaxPoolingLayer<bfloat>::stride)
        .def_readwrite("padding", &MaxPoolingLayer<bfloat>::padding)
        .def("forward", &MaxPoolingLayer<bfloat>::forward)
        .def("backward", &MaxPoolingLayer<bfloat>::backward)
        .def("zeroGrad", &MaxPoolingLayer<bfloat>::zeroGrad)
        .def("getNumParams", &MaxPoolingLayer<bfloat>::getNumParams)
        .def("saveWeights", &MaxPoolingLayer<bfloat>::saveWeights);

    class_<FullyConnectedLayer<bfloat>>(m, "FullyConnectedLayer")
        .def(init<int, int>())
        .def_readwrite("in_features", &FullyConnectedLayer<bfloat>::in_features)
        .def_readwrite("out_features", &FullyConnectedLayer<bfloat>::out_features)
        .def_readwrite("weights", &FullyConnectedLayer<bfloat>::weights)
        .def_readwrite("biases", &FullyConnectedLayer<bfloat>::biases)
        .def_readwrite("dWeights", &FullyConnectedLayer<bfloat>::dWeights)
        .def_readwrite("dBiases", &FullyConnectedLayer<bfloat>::dBiases)
        .def("initializeParams", &FullyConnectedLayer<bfloat>::initializeParams)
        .def("zeroGrad", &FullyConnectedLayer<bfloat>::zeroGrad)
        .def("getNumParams", &FullyConnectedLayer<bfloat>::getNumParams)
        .def("saveWeights", &FullyConnectedLayer<bfloat>::saveWeights);

    class_<ConvolutionOperation<bfloat>>(m, "ConvolutionOperation")
        .def(init<ConvolutionLayer<bfloat>&>())
        .def("forward", &ConvolutionOperation<bfloat>::forward)
        .def("backward", &ConvolutionOperation<bfloat>::backward);

    class_<MaxPoolingOperation<bfloat>>(m, "MaxPoolingOperation")
        .def(init<int, int, int, int>())
        .def("forward", &MaxPoolingOperation<bfloat>::forward)
        .def("backward", &MaxPoolingOperation<bfloat>::backward);

    class_<FullyConnectedOperation<bfloat>>(m, "FullyConnectedOperation")
        .def(init<FullyConnectedLayer<bfloat>&>())
        .def("forward", &FullyConnectedOperation<bfloat>::forward)
        .def("backward", &FullyConnectedOperation<bfloat>::backward);

    class_<AMSGrad<bfloat>>(m, "AMSGrad")
        .def(init<double, double, double, double, double>())
        .def("initializeConv", &AMSGrad<bfloat>::initializeConv)
        .def("update", overload_cast<ConvolutionLayer<bfloat>&,
                const std::vector<std::vector<std::vector<std::vector<bfloat>>>>&,
                const std::vector<bfloat>&>(&AMSGrad<bfloat>::update))
        .def("initializeFC", &AMSGrad<bfloat>::initializeFC)
        .def("update", overload_cast<FullyConnectedLayer<bfloat>&,
                const std::vector<std::vector<bfloat>>&,
                const std::vector<bfloat>&>(&AMSGrad<bfloat>::update));

    class_<WeightStruct<bfloat>>(m, "WeightStruct")
        .def("getType", &WeightStruct<bfloat>::getType)
        .def("serialize", &WeightStruct<bfloat>::serialize);

    class_<ConvolutionalWeights<bfloat>>(m, "ConvolutionalWeights")
        .def(init<const ConvolutionLayer<bfloat>&>())
        .def_readwrite("in_channels", &ConvolutionalWeights<bfloat>::in_channels)
        .def_readwrite("out_channels", &ConvolutionalWeights<bfloat>::out_channels)
        .def_readwrite("filter_height", &ConvolutionalWeights<bfloat>::filter_height)
        .def_readwrite("filter_width", &ConvolutionalWeights<bfloat>::filter_width)
        .def_readwrite("stride", &ConvolutionalWeights<bfloat>::stride)
        .def_readwrite("padding", &ConvolutionalWeights<bfloat>::padding)
        .def_readwrite("filters", &ConvolutionalWeights<bfloat>::filters)
        .def_readwrite("biases", &ConvolutionalWeights<bfloat>::biases)
        .def("getType", &ConvolutionalWeights<bfloat>::getType)
        .def("serialize", &ConvolutionalWeights<bfloat>::serialize)
        .def_static("deserialize", &ConvolutionalWeights<bfloat>::deserialize);

    class_<ConnectedWeights<bfloat>>(m, "ConnectedWeights")
        .def(init<const FullyConnectedLayer<bfloat>&>())
        .def_readwrite("in_features", &ConnectedWeights<bfloat>::in_features)
        .def_readwrite("out_features", &ConnectedWeights<bfloat>::out_features)
        .def_readwrite("weights", &ConnectedWeights<bfloat>::weights)
        .def_readwrite("biases", &ConnectedWeights<bfloat>::biases)
        .def("getType", &ConnectedWeights<bfloat>::getType)
        .def("serialize", &ConnectedWeights<bfloat>::serialize)
        .def_static("deserialize", &ConnectedWeights<bfloat>::deserialize);

    class_<PoolingWeights<bfloat>>(m, "PoolingWeights")
        .def(init<const MaxPoolingLayer<bfloat>&>())
        .def_readwrite("pool_height", &PoolingWeights<bfloat>::pool_height)
        .def_readwrite("pool_width", &PoolingWeights<bfloat>::pool_width)
        .def_readwrite("stride", &PoolingWeights<bfloat>::stride)
        .def_readwrite("padding", &PoolingWeights<bfloat>::padding)
        .def("getType", &PoolingWeights<bfloat>::getType)
        .def("serialize", &PoolingWeights<bfloat>::serialize)
        .def_static("deserialize", &PoolingWeights<bfloat>::deserialize);

    class_<LayerConfig>(m, "LayerConfig")
            .def_static("conv", &LayerConfig::conv)
            .def_static("pool", &LayerConfig::pool)
            .def_static("fc", &LayerConfig::fc)
            .def_readwrite("type", &LayerConfig::type)
            .def_readwrite("in_channels", &LayerConfig::in_channels)
            .def_readwrite("out_channels", &LayerConfig::out_channels)
            .def_readwrite("filter_height", &LayerConfig::filter_height)
            .def_readwrite("filter_width", &LayerConfig::filter_width)
            .def_readwrite("stride", &LayerConfig::stride)
            .def_readwrite("padding", &LayerConfig::padding)
            .def_readwrite("pool_height", &LayerConfig::pool_height)
            .def_readwrite("pool_width", &LayerConfig::pool_width)
            .def_readwrite("in_features", &LayerConfig::in_features)
            .def_readwrite("out_features", &LayerConfig::out_features);
}