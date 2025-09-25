# The MPMD optimize passes and pipeline.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")
# load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(default_visibility = ["//visibility:public"])

td_library(
    name = "passes_td_files",
    srcs = [
        "passes.td",
    ],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "passes_inc",
    tbl_outs = {
        "passes.h.inc": [
            "-gen-pass-decls",
            "-name=MpmdOptimize",
        ],
        "g3doc/mpmd_optimize_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "optimize_pipeline.cc",
        "remat_fragment.cc",
        "rule_based_schedule.cc",
        "scheduler.cc",
    ],
    hdrs = [
        "passes.h",
        "scheduler.h",
    ],
    deps = [
        ":passes_inc",
        ":pipeline_schedule",
        ":utils",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/ir:fragment_execution_rules",
        "//shardy/dialect/mpmd/transforms/common:distributed_function_pass",
        "//shardy/dialect/mpmd/transforms/common:passes",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "pipeline_schedule",
    srcs = ["pipeline_schedule.cc"],
    hdrs = ["pipeline_schedule.h"],
    deps = [
        ":utils",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/ir:register",
        "//shardy/dialect/mpmd/transforms/common:testing_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)
