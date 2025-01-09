//
// Created by Vijay Goyal on 2025-01-09.
//

#include "LayerConfig.h"

LayerConfig LayerConfig::conv(int in_c, int out_c, int fh, int fw, int st, int pad) {
    LayerConfig lc;
    lc.type = "conv";
    lc.in_channels  = in_c;
    lc.out_channels = out_c;
    lc.filter_height = fh;
    lc.filter_width  = fw;
    lc.stride = st;
    lc.padding = pad;
    return lc;
}

LayerConfig LayerConfig::pool(int ph, int pw, int st, int pad) {
    LayerConfig lc;
    lc.type = "pool";
    lc.pool_height = ph;
    lc.pool_width  = pw;
    lc.stride = st;
    lc.padding = pad;
    return lc;
}

LayerConfig LayerConfig::fc(int in_f, int out_f) {
    LayerConfig lc;
    lc.type = "fc";
    lc.in_features = in_f;
    lc.out_features = out_f;
    return lc;
}