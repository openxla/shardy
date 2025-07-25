# The MPMD dialect passes and pipelines.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "passes",
    srcs = ["passes.cc"],
    hdrs = ["passes.h"],
    deps = [
        "//shardy/dialect/mpmd/transforms/common:passes",
        "//shardy/dialect/mpmd/transforms/export:passes",
        "//shardy/dialect/mpmd/transforms/import:passes",
        "//shardy/dialect/mpmd/transforms/optimize:passes",
        "//shardy/dialect/mpmd/transforms/sharding_propagation:passes",
        "@llvm-project//mlir:Pass",
    ],
)
