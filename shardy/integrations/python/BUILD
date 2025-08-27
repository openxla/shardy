# Python integration for SDY.

load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "jax_mpmd_test",
    srcs = ["jax_mpmd_test.py"],
    deps = [
        "//shardy/integrations/python/jax/mpmd",
    ],
)
