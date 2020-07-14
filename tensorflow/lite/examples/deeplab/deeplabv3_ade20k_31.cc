/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "bitmap_helpers.h"

using namespace tflite;
using namespace tflite::deeplab;

#define TFLITE_DEEPLAB_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "minimal <tflite model> <bitmap file>\n");
    return 1;
  }

  auto filename = argv[1];
  auto ground_truth_file = argv[2];
  auto input_bitmap_file = argv[3];

  int w, h, d;
  std::ifstream stream(ground_truth_file, std::ios::in | std::ios::binary);
  std::vector<uint8_t> ground_truth_vector(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
#if __DEBUG__
  std::cout << "len: " << ground_truth_vector.size() << "\n";
#endif

  auto bitmap_vector = read_bmp(input_bitmap_file, w, h, d);
#if __DEBUG__
  std::cout << w << ", " << h << ", " << d << "\n";
#endif

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_DEEPLAB_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_DEEPLAB_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_DEEPLAB_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Fill input buffers
  // assuming single input
  auto input_tensor_index = interpreter->inputs()[0];
  auto input_type = interpreter->tensor(input_tensor_index)->type;
  switch (input_type) {
    case kTfLiteUInt8: {
      auto input = interpreter->typed_tensor<uint8_t>(input_tensor_index);
      memcpy(input, bitmap_vector.data(), bitmap_vector.size());
    } break;
    case kTfLiteFloat32: {
      std::cout << "input size: " << bitmap_vector.size() << "\n";
      auto input = interpreter->typed_tensor<float>(input_tensor_index);
      for (int i = 0; i < bitmap_vector.size(); i++)
        input[i] = bitmap_vector.data()[i] * 1.0;
    } break;
    default:
      break;
  }

#if __DEBUG__
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());
#endif

  // Run inference
  TFLITE_DEEPLAB_CHECK(interpreter->Invoke() == kTfLiteOk);
#if __DEBUG__
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());
#endif

  // Read output buffers
  auto output_tensor_index = interpreter->outputs()[0];
  auto output = interpreter->typed_tensor<int32_t>(output_tensor_index);
#if __DEBUG__
  std::cout << "output index: " << output_tensor_index << "\n";
  std::cout << "output_type: " << interpreter->tensor(output_tensor_index)->type
            << "\n";
#endif

  uint32_t tp_acc = 0, fp_acc = 0, fn_acc = 0;
  for (int c = 0; c < 32; c++) {
    uint64_t true_positive = 0;
    uint64_t false_positive = 0;
    uint64_t false_negative = 0;
    for (int i = 0; i < (512 * 512 - 1); i++) {
      auto p = (uint8_t)0x000000ff & output[i];
      auto g = ground_truth_vector[i];

      // 0xff means ignore
      if (ground_truth_vector[i] != 0xff) {
	// trichotomy
        if ((p == c) or (g == c)) {
          if (p == g)
            true_positive++;
          else if (p == c)
            false_positive++;
          else
            false_negative++;
        }
      }
    }

    tp_acc += true_positive;
    fp_acc += false_positive;
    fn_acc += false_negative;
  }

  // miou = true_positive / (true_positive + false_positive + false_negative)
  auto miou = tp_acc * 1.0 / (tp_acc + fp_acc + fn_acc);
  std::cout << "mIOU: " << miou << "\n";

  return 0;
}
