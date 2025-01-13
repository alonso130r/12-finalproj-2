//
// Created by Vijay Goyal on 2025-01-10.
//

#ifndef INC_12_FINALPROJ_2_LAYER_H
#define INC_12_FINALPROJ_2_LAYER_H


template <typename Type>
class Layer {
public:
   [[nodiscard]] virtual ssize_t getNumParams() const = 0;
   virtual void zeroGrad() = 0;
};

//#include "Layer.tpp"

#endif //INC_12_FINALPROJ_2_LAYER_H
