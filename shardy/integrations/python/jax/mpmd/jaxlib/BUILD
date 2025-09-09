# load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mpmd_program",
    srcs = ["mpmd_program.cc"],
    hdrs = ["mpmd_program.h"],
    deps = [
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/ir:fragment_execution_rules",
        "//shardy/dialect/mpmd/ir:register",
        "//shardy/dialect/mpmd/transforms/export:passes",
        "//shardy/dialect/mpmd/transforms/export:utils",
        "//shardy/dialect/mpmd/transforms/import:mesh_assignment_map",
        "//shardy/dialect/mpmd/transforms/import:passes",
        "//shardy/dialect/mpmd/transforms/optimize:passes",
        "//shardy/dialect/mpmd/transforms/optimize:pipeline_schedule",
        "//shardy/dialect/mpmd/transforms/sharding_propagation:passes",
        "//third_party/absl/log",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
