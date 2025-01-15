//
// Created by Vijay Goyal on 2025-01-10.
//

#include <cstddef>
#include <string>
typedef size_t rsize_t;
#include <_string.h>
#include <sys/types.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdfloat>
#include <vector>
#include "../model/ModularCNN.h"
#include "../tools/LayerConfig.h"
#include "../tools/Tensor.h"
#include "../tools/Operation.h"
#include "../tools/MaxPoolingOperation.h"
#include "../tools/FullyConnectedOperation.h"
#include "../tools/ConvolutionOperation.h"
#include "../tools/ComputationGraph.h"
#include "../tools/AMSGrad.h"
#include "../layers/ConvolutionLayer.h"
#include "../layers/FullyConnectedLayer.h"
#include "../layers/Layer.h"
#include "../layers/MaxPoolingLayer.h"
#include "../tools/CrossEntropy.h"
#include "../tools/ConvolutionalWeights.h"
#include "../tools/ConnectedWeights.h"
#include "../tools/PoolingWeights.h"
#include "../tools/WeightStruct.h"

using bfloat = std::bfloat16_t;

using namespace pybind11;

PYBIND11_MODULE(ModularCNN, m) {
    m.doc() = "Modular CNN implementation in C++";

    class_<LayerConfig>(m, "LayerConfig")
        .def_static("conv", &LayerConfig::conv)
        .def_static("pool", &LayerConfig::pool)
        .def_static("fc", &LayerConfig::fc);

    class_<Tensor<bfloat>>(m, "Tensor")
        .def(init<int, int, int, int, bfloat>())
        .def_readwrite("data", &Tensor<bfloat>::data)
        .def_readwrite("grad", &Tensor<bfloat>::grad)
        .def_readwrite("creator", &Tensor<bfloat>::creator)
        .def("zeroGrad", &Tensor<bfloat>::zeroGrad);

    class_<Layer<bfloat>>(m, "Layer")
        .def("getNumParams", &Layer<bfloat>::getNumParams)
        .def("zeroGrad", &Layer<bfloat>::zeroGrad);

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
        .def("zeroGrad", &ModularCNN<bfloat>::zeroGrad)
        .def("getTotalParams", &ModularCNN<bfloat>::getTotalParams);

    class_<ConvolutionLayer<bfloat>>(m, "ConvolutionLayer")
        .def(init<int, int, int, int, int, int>())
        .def_readwrite("filters", &ConvolutionLayer<bfloat>::filters)
        .def_readwrite("biases", &ConvolutionLayer<bfloat>::biases)
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
        .def("forward", &MaxPoolingLayer<bfloat>::forward)
        .def("backward", &MaxPoolingLayer<bfloat>::backward)
        .def("zeroGrad", &MaxPoolingLayer<bfloat>::zeroGrad)
        .def("getNumParams", &MaxPoolingLayer<bfloat>::getNumParams)
        .def("saveWeights", &MaxPoolingLayer<bfloat>::saveWeights);

    class_<FullyConnectedLayer<bfloat>>(m, "FullyConnectedLayer")
        .def(init<int, int>())
        .def_readwrite("weights", &FullyConnectedLayer<bfloat>::weights)
        .def_readwrite("biases", &FullyConnectedLayer<bfloat>::biases)
        .def("initializeParams", &FullyConnectedLayer<bfloat>::initializeParams)
        .def("zeroGrad", &FullyConnectedLayer<bfloat>::zeroGrad)
        .def("getNumParams", &FullyConnectedLayer<bfloat>::getNumParams);

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
        .def("getType", &ConvolutionalWeights<bfloat>::getType)
        .def("serialize", &ConvolutionalWeights<bfloat>::serialize)
        .def_static("deserialize", &ConvolutionalWeights<bfloat>::deserialize);

    class_<ConnectedWeights<bfloat>>(m, "ConnectedWeights")
        .def(init<const FullyConnectedLayer<bfloat>&>())
        .def("getType", &ConnectedWeights<bfloat>::getType)
        .def("serialize", &ConnectedWeights<bfloat>::serialize)
        .def_static("deserialize", &ConnectedWeights<bfloat>::deserialize);

    class_<PoolingWeights<bfloat>>(m, "PoolingWeights")
        .def(init<const MaxPoolingLayer<bfloat>&>())
        .def("getType", &PoolingWeights<bfloat>::getType)
        .def("serialize", &PoolingWeights<bfloat>::serialize)
        .def_static("deserialize", &PoolingWeights<bfloat>::deserialize);
}