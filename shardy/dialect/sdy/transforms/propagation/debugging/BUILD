# Systems to debug SDY propagation.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "source_sharding",
    srcs = ["source_sharding.cc"],
    hdrs = ["source_sharding.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/propagation:sharding_projection",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
