# Description:
# TensorFlow Lite Example Label Image.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

exports_files(glob([
    "testdata/*.bmp",
]))

cc_binary(
    name = "label_image",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":bitmap_helpers",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
    ] + select({
        "//tensorflow:android": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//tensorflow:android_arm64": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//conditions:default": [],
    }),
)

apple_binary(
    name = "label_image_metal",
    platform_type = "macos",
    deps = [":label_image_m"],
)

objc_library(
    name = "label_image_m",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image.mm",
    ],
    deps = [
        ":bitmap_helpers_m",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
    ] + select({
        "//tensorflow:android_arm64": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//conditions:default": ["//tensorflow/lite/delegates/gpu:metal_delegate"],
    }),
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "label_image.h",
    ],
    deps = [
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "//tensorflow:android": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//tensorflow:android_arm64": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//conditions:default": [],
    }),
)

objc_library(
    name = "bitmap_helpers_m",
    srcs = ["bitmap_helpers.mm"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "label_image.h",
    ],
    deps = [
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "//tensorflow:android": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//tensorflow:android_arm64": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "label_image_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        ":bitmap_helpers",
        "@com_google_googletest//:gtest",
    ],
)
