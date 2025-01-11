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

using bfloat16 = std::bfloat16;

using namespace pybind11;

PYBIND11_MODULE(ModularCNN, m) {

}