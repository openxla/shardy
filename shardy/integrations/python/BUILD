# Python integration for SDY.

load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "sdy",
    srcs = [
        "sdy.py",
    ],
    deps = [
        "//shardy/integrations/python/ir",
    ],
)

py_test(
    name = "sdy_test",
    srcs = ["sdy_test.py"],
    deps = [
        ":sdy",
        "//third_party/py/mlir:ir",
    ],
)
