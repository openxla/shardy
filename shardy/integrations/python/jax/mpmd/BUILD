load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "mpmd",
    srcs = ["__init__.py"],
    deps = [":api"],
)

py_library(
    name = "api",
    srcs = ["api.py"],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [":api"],
)
