//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H
#define INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H

#include "Operation.h"
#include "ConvolutionLayer.h"

template <typename Type>
class ConvolutionOperation : public Operation<Type> {
private:
    ConvolutionLayer<Type>& convolutionLayer;
    Tensor<Type> input;

public:
    ConvolutionOperation(ConvolutionLayer& convolutionLayer);
    Tensor<Type> forward(const Tensor<Type>& input) override;
    void backward(Tensor<Type>& output_grad) override;
};

#include "ConvolutionOperation.tpp"

#endif //INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H
