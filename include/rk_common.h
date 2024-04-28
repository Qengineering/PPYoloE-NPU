#ifndef _RKNN_MODEL_ZOO_COMMON_H_
#define _RKNN_MODEL_ZOO_COMMON_H_

#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

void dump_tensor_attr(rknn_tensor_attr* attr);
unsigned char* load_model(const char* filename, int& fileSize);

#endif //_RKNN_MODEL_ZOO_COMMON_H_
