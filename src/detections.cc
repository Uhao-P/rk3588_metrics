// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>
#include <dirent.h>
#include <iostream>
#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1
/*-------------------------------------------
                  Functions
-------------------------------------------*/

using namespace std;
using namespace cv;

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

bool endsWith(std::string const &str, std::string const &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

void GetFileNames(std::string path, std::vector<std::string>& filenames, std::vector<std::string>& filenames_file)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cout << "Folder doesn't Exist!" << std::endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
          filenames.push_back(path + "/" + ptr->d_name);
          filenames_file.push_back(ptr->d_name);
      }
    }
    closedir(pDir);
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  int            status     = 0;
  char*          model_name = NULL;
  rknn_context   ctx;
  size_t         actual_size        = 0;
  int            img_width          = 0;
  int            img_height         = 0;
  int            img_channel        = 0;
  const float    nms_threshold      = NMS_THRESH;
  const float    box_conf_threshold = BOX_THRESH;
  struct timeval start_time, stop_time;
  int            ret;

  printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

  model_name       = (char*)argv[1];
  char* dataset_path = argv[2];

  /* Create the neural network */
  printf("Loading mode...\n");
  int            model_data_size = 0;
  unsigned char* model_data      = load_model(model_name, &model_data_size);
  ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }

  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  int channel = 3;
  int width   = 0;
  int height  = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    printf("model is NCHW input fmt\n");
    channel = input_attrs[0].dims[1];
    height  = input_attrs[0].dims[2];
    width   = input_attrs[0].dims[3];
  } else {
    printf("model is NHWC input fmt\n");
    height  = input_attrs[0].dims[1];
    width   = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
  }

  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  vector<string> img_path;
  vector<string> img_name;
  std::string _input_path = dataset_path;
  if (!endsWith(_input_path, "/"))
  {
    _input_path += "/";
  }
  std::string groundtruths_path = _input_path + "groundtruths";
  std::string imgs_path = _input_path + "imgs";
  GetFileNames(imgs_path, img_path, img_name);
  for(int j = 0; j <img_path.size(); j++) 
  {
    size_t found = img_name[j].find_last_of('.');
    std::string name = "";
    if(found != std::string::npos)
    {
      name = img_name[j].substr(0, found);
    }
    std::string detections_path_file = _input_path + "detections/" + name + ".txt";
    std::string res_path_file = _input_path + "res/" + name + ".jpg";
    std::cout << "-------------------------------"<<std::endl;
    std::cout << "detections_path_file = " << detections_path_file << std::endl;
    std::cout << "res_path_file = " << res_path_file <<std::endl;



    printf("Read %s ...\n", img_path[j].c_str());
    cv::Mat orig_img = cv::imread(img_path[j].c_str(), 1);
    if (!orig_img.data) {
      printf("cv::imread %s fail!\n", img_path[j].c_str());
      return -1;
    }
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    img_width  = img.cols;
    img_height = img.rows;
    printf("img width = %d, img height = %d\n", img_width, img_height);


    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = width * height * channel;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // You may not need resize when src resulotion equals to dst resulotion

    if (img_width != width || img_height != height) {
      cv::Mat img_resize = img.clone();
      cv::resize(img_resize, img_resize, cv::Size(640, 640));
      inputs[0].buf = (void*)img_resize.data;
      gettimeofday(&start_time, NULL);
      rknn_inputs_set(ctx, io_num.n_input, inputs);
    } else {
      inputs[0].buf = (void*)img.data;
      gettimeofday(&start_time, NULL);
      rknn_inputs_set(ctx, io_num.n_input, inputs);
    }

    

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
      outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // post process
    float scale_w = (float)width / img_width;
    float scale_h = (float)height / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float>    out_scales;
    std::vector<int32_t>  out_zps;
    for (int i = 0; i < io_num.n_output; ++i) {
      out_scales.push_back(output_attrs[i].scale);
      out_zps.push_back(output_attrs[i].zp);
    }
    post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
                box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    ofstream out_txt_file;
    out_txt_file.open(detections_path_file, ios::out | ios::trunc);
    out_txt_file << fixed;

    // Draw Objects
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++) {
      detect_result_t* det_result = &(detect_result_group.results[i]);
      sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
      printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
            det_result->box.right, det_result->box.bottom, det_result->prop);
      int x1 = det_result->box.left;
      int y1 = det_result->box.top;
      int x2 = det_result->box.right;
      int y2 = det_result->box.bottom;
      rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
      putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
      std::string out_txt_file_str = std::to_string(det_result->class_id) + " " + std::to_string(det_result->prop) + " " + std::to_string(int((x1+x2)/2)) + " " + std::to_string(int((y1+y2)/2)) + " " + std::to_string(x2-x1) + " " + std::to_string(y2-y1);
      out_txt_file << out_txt_file_str << endl;
    }
    out_txt_file.close();
    imwrite(res_path_file, orig_img);
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
  }

  deinitPostProcess();

  // release
  ret = rknn_destroy(ctx);

  if (model_data) {
    free(model_data);
  }

  return 0;
}