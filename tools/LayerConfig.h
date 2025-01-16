//
// Created by Vijay Goyal on 2025-01-09.
//

#ifndef INC_12_FINALPROJ_2_LAYERCONFIG_H
#define INC_12_FINALPROJ_2_LAYERCONFIG_H

#include <cstddef>
#include <string>

/**
 * @brief A simple struct describing one layer in the CNN by type ("conv", "pool", "fc")
 *        and the associated parameters.
 */
struct LayerConfig {
    std::string type;  // "conv", "pool", or "fc"

    // Convolution parameters
    int in_channels = 0;
    int out_channels = 0;
    int filter_height = 0;
    int filter_width  = 0;
    int stride        = 1;
    int padding       = 0;

    // Pooling parameters
    int pool_height   = 0;
    int pool_width    = 0;
    // (stride, padding) can reuse the above if we want, or store them separately

    // Fully connected parameters
    int in_features  = 0;
    int out_features = 0;

    static LayerConfig conv(int in_c, int out_c, int fh, int fw, int st = 1, int pad = 0);

    static LayerConfig pool(int ph, int pw, int st = 1, int pad = 0);

    static LayerConfig fc(int in_f, int out_f);
};


#endif //INC_12_FINALPROJ_2_LAYERCONFIG_H
