//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H
#define INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H

#include "Operation.h"
#include "ConvolutionLayer.h"

class ConvolutionOperation {
private:
    ConvolutionLayer& convolutionLayer;
    Tensor input;

public:
    ConvolutionOperation(ConvolutionLayer& convolutionLayer);
    Tensor forward(const Tensor& input) override;
    void backward(Tensor& output_grad) override;
};


#endif //INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H
