//
// Created by Vijay Goyal on 2025-01-08.
//

#ifndef INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H
#define INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H

#include "Operation.h"
#include "../layers/ConvolutionLayer.h"

template <typename Type>
class ConvolutionOperation : public Operation<Type> {
    typedef std::vector<std::vector<std::vector<std::vector<Type>>>> Tensor4D;
private:
    ConvolutionLayer<Type>& convolutionLayer;
    std::shared_ptr<Tensor<Type>> input;

public:
    explicit ConvolutionOperation(ConvolutionLayer<Type>& convolutionLayer);
    std::shared_ptr<Tensor<Type>> forward(const std::shared_ptr<Tensor<Type>>& input) override;
    std::shared_ptr<Tensor<Type>> backward(const std::shared_ptr<Tensor<Type>>& output_grad) override;
};

#include "ConvolutionOperation.tpp"

#endif //INC_12_FINALPROJ_2_CONVOLUTIONOPERATION_H
