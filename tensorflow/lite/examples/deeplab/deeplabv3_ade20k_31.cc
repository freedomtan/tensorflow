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
#include <string>
#include <vector>

#include "bitmap_helpers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace tflite;
using namespace tflite::deeplab;

#define TFLITE_DEEPLAB_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

void run_model(std::unique_ptr<Interpreter> &interpreter,
               string ground_truth_file, string input_bitmap_file,
               std::vector<uint64_t> &true_positives,
               std::vector<uint64_t> &false_positives,
               std::vector<uint64_t> &false_negatives) {
  std::ifstream stream(ground_truth_file, std::ios::in | std::ios::binary);
  std::vector<uint8_t> ground_truth_vector(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
#if __DEBUG__
  std::cout << "len: " << ground_truth_vector.size() << "\n";
#endif

  int w, h, d;
  auto bitmap_vector = read_bmp(input_bitmap_file, w, h, d);
#if __DEBUG__
  std::cout << w << ", " << h << ", " << d << "\n";
#endif

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
      // std::cout << "input size: " << bitmap_vector.size() << "\n";
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
  for (int c = 1; c < 32; c++) {
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0;

    for (int i = 0; i < (512 * 512 - 1); i++) {
      auto p = (uint8_t)0x000000ff & output[i];
      auto g = ground_truth_vector[i];

      // trichotomy
      if ((p == c) or (g == c)) {
        if (p == g) {
          true_positive++;
        } else if (p == c) {
          if ((g > 0) && (g < 32)) false_positive++;
        } else {
          false_negative++;
        }
      }
    }

    true_positives.push_back(true_positive);
    false_positives.push_back(false_positive);
    false_negatives.push_back(false_negative);
  }
#if __DEBUG__
  for (int i = 0; i < 31; i++) {
    std::cout << true_positives[i] << ", " << false_positives[i] << ", "
              << false_negatives[i];
    if (i < 30) std::cout << ", ";
  }
  std::cout << "\n";
#endif
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }

  auto model_filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_filename);
  TFLITE_DEEPLAB_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_DEEPLAB_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_DEEPLAB_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  std::vector<uint64_t> tp_acc, fp_acc, fn_acc;

  for (int i = 1; i <= 2000; i++) {
    string ground_truth_prefix = "/tmp/ade20k_512/annotations/raw/ADE_val_0000";
    string input_bitmap_prefix = "/tmp/ade20k_512/images/bmp/ADE_val_0000";

    char foo[5];
    foo[4] = '\0';
    std::snprintf(foo, 5, "%04d", i);
    auto ground_truth_file = ground_truth_prefix + foo + ".raw";
    auto input_bitmap_file = input_bitmap_prefix + foo + ".bmp";

    std::vector<uint64_t> true_positives, false_positives, false_negatives;

    run_model(interpreter, ground_truth_file, input_bitmap_file, true_positives,
              false_positives, false_negatives);

    if (i == 1) {
      for (int j = 0; j < 31; j++) {
        tp_acc.push_back(true_positives[j]);
        fp_acc.push_back(false_positives[j]);
        fn_acc.push_back(false_negatives[j]);
      }
    } else {
      for (int j = 0; j < 31; j++) {
        tp_acc[j] += true_positives[j];
        fp_acc[j] += false_positives[j];
        fn_acc[j] += false_negatives[j];
      }
    }

    std::cout << i << "\n";
#if __DEBUG__
    std::cout << i << ": ";
    for (int j = 0; j < 31; j++) {
      std::cout << tp_acc[j] << ", " << fp_acc[j] << ", " << fn_acc[j];
      if (j < 30) std::cout << ", ";
    }
    std::cout << "\n";
    for (int j = 0; j < 31; j++) {
      std::cout << "mIOU class " << j + 1 << ": "
                << tp_acc[j] * 1.0 / (tp_acc[j] + fp_acc[j] + fn_acc[j])
                << "\n";
    }
#endif
  }

  float iou_sum = 0.0;
  for (int j = 0; j < 31; j++) {
    auto iou = tp_acc[j] * 1.0 / (tp_acc[j] + fp_acc[j] + fn_acc[j]);

    std::cout << "IOU class " << j + 1 << ": " << tp_acc[j] << ", " << fp_acc[j]
              << ", " << fn_acc[j] << ", " << iou << "\n";
    iou_sum += iou;
  }
  std::cout << "mIOU over_all: " << iou_sum / 31 << "\n";

  return 0;
}
