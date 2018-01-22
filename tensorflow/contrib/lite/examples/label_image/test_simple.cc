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

#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/version.h"
#include "tensorflow/contrib/lite/kernels/register.h"

#include "tensorflow/contrib/lite/examples/label_image/bitmap_helpers.h"

#define LOG(x) std::cerr

namespace tflite {
namespace test_simple {

struct TensorData {
  TensorType type;
  std::vector<int> shape;
  float min;
  float max;
  float scale;
  int32_t zero_point;
};

std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;

std::map<int, TensorData> tensor_data_;
std::vector<int32_t> inputs_;
std::vector<int32_t> outputs_;
std::vector<flatbuffers::Offset<Tensor>> tensors_;
std::vector<flatbuffers::Offset<OperatorCode>> opcodes_;
std::vector<flatbuffers::Offset<Operator>> operators_;

flatbuffers::FlatBufferBuilder builder_;

int AddTensor(TensorData t) {
  int id = tensors_.size();

  // This is slightly different depending on whether we are adding a
  // quantized or a regular tensor.
  bool is_quantized = (t.min != 0 || t.max != 0 || t.scale != 0);

  flatbuffers::Offset<QuantizationParameters> q_params = 0;

  if (is_quantized) {
    q_params = CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0, builder_.CreateVector<float>({t.scale}),
        builder_.CreateVector<int64_t>({t.zero_point}));
  }

  tensors_.push_back(CreateTensor(builder_, builder_.CreateVector<int>({}),
                                  t.type, /*buffer=*/0,
                                  /*name=*/0, q_params));

  tensor_data_[id] = t;

  return id;
}

int AddInput(const TensorData& t) {
  int id = AddTensor(t);
  inputs_.push_back(id);
  return id;
}

int AddOutput(const TensorData& t) {
  int id = AddTensor(t);
  outputs_.push_back(id);
  return id;
}

void SetBuiltinOp(BuiltinOperator type,
                                 BuiltinOptions builtin_options_type,
                                 flatbuffers::Offset<void> builtin_options) {
  opcodes_.push_back(CreateOperatorCode(builder_, type, 0));
  operators_.push_back(CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(inputs_),
      builder_.CreateVector<int32_t>(outputs_), builtin_options_type,
      builtin_options,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS));
}


void InitImpl(const std::string& graph, const std::vector<int>& sizes,
              const std::string& input_layer_type, int num_threads) {
#if 0
  auto input1_ = AddInput({TensorType_FLOAT32, {1}, 0, 0, 0, 0});
  auto input2_ = AddInput({TensorType_FLOAT32, {1}, 0, 0, 0, 0});
  auto output_ = AddOutput({TensorType_FLOAT32, {1}, 0, 0, 0, 0});

  LOG(INFO) << "tensors: " << input1_ << ", " << input2_ << ", " << output_ << "\n";

  SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
  CreateAddOptions(builder_, ActivationFunctionType_NONE).Union());

  // to build interpreter
  auto opcodes = builder_.CreateVector(opcodes_);
  auto operators = builder_.CreateVector(operators_);
  auto tensors = builder_.CreateVector(tensors_);
  auto inputs = builder_.CreateVector<int32_t>(inputs_);
  auto outputs = builder_.CreateVector<int32_t>(outputs_);

  std::vector<flatbuffers::Offset<SubGraph>> subgraphs;
  auto subgraph = CreateSubGraph(builder_, tensors, inputs, outputs, operators);
  subgraphs.push_back(subgraph);
  auto subgraphs_flatbuffer = builder_.CreateVector(subgraphs);

  std::vector<flatbuffers::Offset<Buffer>> buffers_vec;
  auto buffers = builder_.CreateVector(buffers_vec);
  auto description = builder_.CreateString("programmatic model");
  builder_.Finish(CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                              subgraphs_flatbuffer, description, buffers));

  auto* smodel = GetModel(builder_.GetBufferPointer());

  std::unique_ptr<tflite::Interpreter> foo;
  // foo = BuildInterpreter({1, 1});

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(smodel, resolver)(&foo);

  LOG(INFO) << "before allocate tensors \n";
  foo->AllocateTensors();
  LOG(INFO) << "after allocate tensors \n";
  auto i1 = foo->typed_tensor<float>(input1_);
  auto i2 = foo->typed_tensor<float>(input2_);
  i1[0] = 1.02; i2[0] = 2.0;
#endif

  // std::unique_ptr<tflite::Interpreter> foo = new Interpreter();
  tflite::Interpreter *foo = new Interpreter();
  int base_index;

  // two inputs
  foo->AddTensors(2, &base_index);
  // one output
  foo->AddTensors(1, &base_index);
  LOG(INFO) << "tensors: " << foo->tensors_size() << "\n";
  foo->SetInputs({0, 1});
  foo->SetOutputs({2});
  LOG(INFO) << "tensors: " << foo->tensors_size() << "\n";

  TfLiteQuantizationParams quant;

  foo->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input1", {1}, quant);
  foo->SetTensorParametersReadWrite(1, kTfLiteFloat32, "input2", {1}, quant);
  foo->SetTensorParametersReadWrite(2, kTfLiteFloat32, "output", {1}, quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  TfLiteRegistration *addR = resolver.FindOp(BuiltinOperator_ADD);
  if (addR != nullptr)
    LOG(INFO) << "addR: " << addR->builtin_code << "\n";
  TfLiteAddParams f;
  foo->AddNodeWithParameters({0,1}, {2}, nullptr, 0, &f, addR, nullptr);

  foo->AllocateTensors();

  foo->typed_tensor<float>(0)[0] = 1.1;
  foo->typed_tensor<float>(1)[0] = 2.2;

  LOG(INFO) << "before Invoke()\n";
  foo->Invoke();
  LOG(INFO) << "after Invoke()\n";
  auto o = foo->typed_tensor<float>(2);
  LOG(INFO) << "output: " << o[0] << "\n";

  label_image::Settings s;
  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  uint8_t* in = label_image::read_bmp(s.input_bmp_name, &image_width, &image_height,
                         &image_channels, &s);
  LOG(INFO) << "(w, h, c): " << image_width << ", " << image_height << ", " << image_channels << "\n";
  float *fin = new float[image_width * image_height * image_channels];

  for (int i=0; i < image_width * image_height * image_channels; i++)
    fin[i] = in[i];

  tflite::Interpreter *bar = new Interpreter();
  base_index = 0;

  // one input
  bar->AddTensors(1, &base_index);
  // one output
  bar->AddTensors(1, &base_index);
  LOG(INFO) << "tensors: " << bar->tensors_size() << "\n";
  bar->SetInputs({0});
  bar->SetOutputs({1});

  bar->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input", {1, image_width, image_height, image_channels}, quant);
  bar->SetTensorParametersReadWrite(1, kTfLiteFloat32, "output", {1, 224, 224, image_channels}, quant);

  TfLiteRegistration *resize = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR);

  TfLiteResizeBilinearParams wh = {224, 224};;
  bar->AddNodeWithParameters({0}, {1}, nullptr, 0, &wh, resize, nullptr);

  bar->AllocateTensors();
  bar->typed_tensor<float*>(0)[0] = fin;
  bar->Invoke();
}

int Main(int argc, char** argv) {
  InitImpl("", {}, "", 1);
  return 0;
}

}  // namespace test_simple
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::test_simple::Main(argc, argv);
}
