//
// Created by Vijay Goyal on 2025-01-10.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdfloat>
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

using bfloat = std::bfloat16_t;

using namespace pybind11;

PYBIND11_MODULE(ModularCNN, m) {
    m.doc() = "Modular CNN implementation in C++";

    class_<LayerConfig>(m, "LayerConfig")
        .def_static("conv", &LayerConfig::conv)
        .def_static("pool", &LayerConfig::pool)
        .def_static("fc", &LayerConfig::fc);

    class_<Tensor<bfloat>>(m, "Tensor")
        .def(init<int, int, int, int, float>())
        .def_readwrite("data", &Tensor<bfloat>::data)
        .def_readwrite("grad", &Tensor<bfloat>::grad)
        .def_readwrite("creator", &Tensor<bfloat>::creator)
        .def("zeroGrad", &Tensor<bfloat>::zeroGrad);

    class_<ComputationGraph<bfloat>>(m, "ComputationGraph")
        .def(init<>())
        .def("addOperation", &ComputationGraph<bfloat>::addOperation)
        .def("forward", &ComputationGraph<bfloat>::forward)
        .def("backward", &ComputationGraph<bfloat>::backward);

    class_<ModularCNN<bfloat>>(m, "ModularCNN")
        .def(init<std::vector<LayerConfig>>())
        .def("buildGraph", &ModularCNN<bfloat>::buildGraph)
        .def("forward", &ModularCNN<bfloat>::forward)
        .def("zeroGrad", &ModularCNN<bfloat>::zeroGrad)
        .def("getTotalParams", &ModularCNN<bfloat>::getTotalParams);

    class_<ConvolutionLayer<bfloat>>(m, "ConvolutionLayer")
        .def(init<int, int, int, int, int, int>())
        .def("forward", &ConvolutionLayer<bfloat>::forward)
        .def("backward", &ConvolutionLayer<bfloat>::backward);

    class_<MaxPoolingLayer<bfloat>>(m, "MaxPoolingLayer")
        .def(init<int, int, int, int>())
        .def("forward", &MaxPoolingLayer<bfloat>::forward)
        .def("backward", &MaxPoolingLayer<bfloat>::backward)
        .def("getNumParams", &MaxPoolingLayer<bfloat>::getNumParams);

    class_<FullyConnectedLayer<bfloat>>(m, "FullyConnectedLayer")
        .def(init<int, int>())
        .def("initializeParams", &FullyConnectedLayer<bfloat>::initializeParams)
        .def("zeroGrad", &FullyConnectedLayer<bfloat>::zeroGrad)
        .def("getNumParams", &FullyConnectedLayer<bfloat>::getNumParams);
}