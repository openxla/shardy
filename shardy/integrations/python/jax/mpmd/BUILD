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

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        ":partitioning_options",
        "//third_party/py/jax",
        "//third_party/py/jaxtyping",
    ],
)

py_test(
    name = "types_test",
    srcs = ["types_test.py"],
    deps = [
        ":types",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/chex",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "partitioning_options",
    srcs = ["partitioning_options.py"],
)
