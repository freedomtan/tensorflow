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
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <getopt.h>
#include <sys/time.h>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#ifdef TFLITE_CUSTOM_OPS_HEADER
void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);
#endif

#define LOG(x) std::cerr

#define CHECK(x)                  \
  if (!(x)) {                     \
    LOG(ERROR) << #x << "failed"; \
    exit(1);                      \
  }

namespace tensorflow {
namespace benchmark_tflite_model {

std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

void InitImpl(const std::string& graph, const std::vector<int>& sizes,
              const std::string& input_layer_type, int num_threads) {
  CHECK(graph.c_str());

  model = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model) {
    LOG(FATAL) << "Failed to mmap model " << graph << "\n";
  }
  LOG(INFO) << "Loaded model " << graph << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  tflite::InterpreterBuilder (*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
  }

  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  int input = interpreter->inputs()[0];

  if (input_layer_type != "string") {
    interpreter->ResizeInputTensor(input, sizes);
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!\n";
  }

  struct timeval t0, t1;
  gettimeofday(&t0, NULL);
  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke tflite!\n";
  }
  gettimeofday(&t1, NULL);
  LOG(INFO) << "model run successfully\n";
  LOG(INFO) << (get_us(t1) - get_us(t0)) / 1000 << "ms \n";

  interpreter->SetProfiling(true);
  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke tflite!\n";
  }
  LOG(INFO) << "model run successfully\n";
}

void display_usage(const char* myname) {
  LOG(INFO) << myname << "\n"
            << "--tflite_model, -m: model_name.tflite\n"
            << "--threads, -t: number of threads\n";
}

int Main(int argc, char** argv) {
  std::string model_name = "/data/local/tmp/mobilenet_quant_v1_224.tflite";
  std::vector<int> sizes = {1, 224, 224, 3};
  std::string layer_type = "int8";
  int num_threads = 4;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"tflite_model", required_argument, 0, 'm'},
        {"threads", required_argument, 0, 't'},
        {0, 0, 0, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "m:t:", long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'm':
        model_name = optarg;
        break;
      case 't':
        num_threads = strtol(optarg, (char**)NULL, 10);
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage(argv[0]);
        exit(-1);
      default:
        exit(-1);
    }
  }
  InitImpl(model_name, sizes, layer_type, num_threads);
  return 0;
}

}  // namespace benchmark_tflite_model
}  // namespace tensorflow

int main(int argc, char** argv) {
  return tensorflow::benchmark_tflite_model::Main(argc, argv);
}
