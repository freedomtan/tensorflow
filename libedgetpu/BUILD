# Description:
#   Edgetpu custom op api for tflite.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "arm64",
    values = {"cpu": "arm64-v8a"},
)

config_setting(
    name = "armhf",
    values = {"cpu": "armeabi-v7a"},
)

cc_library(
    name = "header",
    hdrs = ["edgetpu.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite:context",
    ],
)

cc_library(
    name = "lib",
    srcs = select({
        ":aarch64": ["libedgetpu_arm64.so"],
        ":arm64": ["libedgetpu_arm64.so"],
        ":armhf": ["libedgetpu_arm32.so"],
        "//conditions:default": ["libedgetpu_x86_64.so"],
    }),
)
