# The SDY dialect passes and pipelines.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "passes",
    srcs = ["passes.cc"],
    hdrs = ["passes.h"],
    deps = [
        "//shardy/dialect/sdy/transforms/export:passes",
        "//shardy/dialect/sdy/transforms/import:passes",
        "//shardy/dialect/sdy/transforms/propagation:passes",
        "@llvm-project//mlir:Pass",
    ],
)
