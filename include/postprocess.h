#ifndef _RKNN_PPYOLOE_DEMO_POSTPROCESS_H_
#define _RKNN_PPYOLOE_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "rk_common.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

// class rknn_app_context_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs, float conf_threshold, float nms_threshold, float scale_w, float scale_h, object_detect_result_list *od_results);

#endif //_RKNN_PPYOLOE_DEMO_POSTPROCESS_H_
