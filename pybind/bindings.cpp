//
// Created by Vijay Goyal on 2025-01-10.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdfloat.h>
#include <ModularCNN.h>
#include <LayerConfig.h>
#include <Tensor.h>
#include <Operation.h>
#include <MaxPoolingOperation.h>
#include <FullyConnectedOperation.h>
#include <ConvolutionOperation.h>
#include <ComputationGraph.h>
#include <AMSGrad.h>
#include <ConvolutionLayer.h>
#include <FullyConnectedLayer.h>
#include <Layer.h>
#include <MaxPoolingLayer.h>

using std::bfloat_16_t as bfloat16;

using namespace pybind11;

PYBIND11_MODULE(ModularCNN, m) {

}