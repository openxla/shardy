# The MPMD common library and passes.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
            "-name=MpmdCommon",
        ],
        "g3doc/mpmd_common_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "absorb_inferred_fragments.cc",
        "call_rewrites.cc",
        "copy_constants.cc",
        "fragment_dce.cc",
        "fragment_dedup.cc",
        "merge_fragments.cc",
        "merge_transfers.cc",
        "remove_transfer_cycles.cc",
        "rule_based_merge.cc",
        "split_bwd_fragments.cc",
        "uniquify_function_inputs_outputs.cc",
        "unroll_for_loops.cc",
    ],
    hdrs = [
        "merge_fragments.h",
        "passes.h",
    ],
    deps = [
        ":distributed_function_pass",
        ":passes_inc",
        ":utils",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/ir:fragment_execution_rules",
        "//shardy/dialect/mpmd/transforms/optimize:utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "distributed_function_pass",
    srcs = ["distributed_function_pass.cc"],
    hdrs = ["distributed_function_pass.h"],
    deps = [
        "//shardy/dialect/mpmd/ir:dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "simplify_region_op_base",
    srcs = ["simplify_region_op_base.cc"],
    hdrs = ["simplify_region_op_base.h"],
    deps = [
        ":utils",
        "//shardy/common:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "testing_utils",
    hdrs = ["testing_utils.h"],
    deps = [
        "//shardy/common:logging",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Support",
    ],
)
