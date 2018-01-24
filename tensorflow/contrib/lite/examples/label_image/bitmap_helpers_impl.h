/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/version.h"
#include "tensorflow/contrib/lite/model.h"

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include "tensorflow/contrib/lite/examples/label_image/label_image.h"

namespace tflite {
namespace label_image {

template <class T>
void downsize_old(T* out, uint8_t* in, int image_height, int image_width,
              int image_channels, int wanted_height, int wanted_width,
              int wanted_channels, Settings* s) {
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    uint8_t* in_row = in + (in_y * image_width * image_channels);
    T* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      uint8_t* in_pixel = in_row + (in_x * image_channels);
      T* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        if (s->input_floating)
          out_pixel[c] = (in_pixel[c] - s->input_mean) / s->input_std;
        else
          out_pixel[c] = in_pixel[c];
      }
    }
  }
}

template <class T>
void downsize(T* out, uint8_t* in, int image_height, int image_width,
              int image_channels, int wanted_height, int wanted_width,
              int wanted_channels, Settings* s) {

  int fd = open("/tmp/in.rgb", O_CREAT|O_RDWR, 0755); 
  write(fd, in, image_height*image_width*image_channels);
  close(fd);

  float *fin = new float[image_height * image_width * image_channels];

  for (int i=0; i < image_height * image_width * image_channels; i++)
    fin[i] = in[i];

  tflite::Interpreter *bar = new Interpreter();
  int base_index = 0;

  // one input
  bar->AddTensors(1, &base_index);
  // one output
  bar->AddTensors(1, &base_index);
  // LOG(INFO) << "tensors: " << bar->tensors_size() << "\n";
  bar->SetInputs({0});
  bar->SetOutputs({1});

  TfLiteQuantizationParams quant;

  bar->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input", {1, image_height, image_width, image_channels}, quant);
  bar->SetTensorParametersReadWrite(1, kTfLiteFloat32, "output", {1, wanted_height, wanted_width, wanted_channels}, quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  TfLiteRegistration *resize = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR);
  TfLiteResizeBilinearParams wh = {wanted_height, wanted_width};
  printf("%d, %d\n", wh.new_height, wh.new_width);
  bar->AddNodeWithParameters({0}, {1}, nullptr, 0, &wh, resize, nullptr);

  TfLiteIntArray* dims = bar->tensor(1)->dims;
  printf("%s: %d, %d\n", bar->tensor(1)->name, bar->tensor(1)->bytes, bar->tensor(1)->dims->size);
  printf("%d, %d, %d, %d\n", dims->data[0], dims->data[1], dims->data[2], dims->data[3]);

  bar->AllocateTensors();
  auto myin = bar->typed_tensor<float>(0);
  myin = fin;
  bar->UseNNAPI(false);
  printf("i: %f, %f, %f\n", myin[0], myin[1], myin[2]);
  printf("i: %f, %f, %f\n", myin[0], myin[1], myin[2]);

  bar->Invoke();

  printf("%s: %d, %d\n", bar->tensor(1)->name, bar->tensor(1)->bytes, bar->tensor(1)->dims->size);
  dims = bar->tensor(1)->dims;
  printf("%d, %d, %d, %d\n", dims->data[0], dims->data[1], dims->data[2], dims->data[3]);

  float *output = bar->typed_tensor<float>(1);
  printf("output: %p\n", output);
  printf("output: %f\n", *output);
  printf("o: %f, %f, %f\n", output[0], output[1], output[2]);

#if 0
  for (int i=0; i < 224 * 224 * 3; i++) {
    if (s->input_floating)
      out[i] = (output[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)output[i];
  }
  if (s->input_floating)
    printf("o: %f, %f, %f\n", out[0], out[1], out[2]);
  else  {
    printf("o: %d, %d, %d\n", out[0], out[1], out[2]);
    printf("o: %d, %d, %d\n", out[0+224*3], out[1+224*3], out[2+224*3]);
    printf("o: %d, %d, %d\n", out[224*224*3-3], out[224*224*3-2], out[224*224*3-1]);
  }
  fd = open("/tmp/foo.rgb", O_CREAT|O_RDWR, 0755); 
  printf("len = %zu, %lu\n", bar->tensor(1)->bytes, bar->tensor(1)->bytes/4);
  write(fd, out, bar->tensor(1)->bytes/4);
#endif
}

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H
