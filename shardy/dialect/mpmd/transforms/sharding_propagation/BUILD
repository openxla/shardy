# The MPMD sharding propagation passes and pipeline.

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
            "-name=MpmdShardingPropagation",
        ],
        "g3doc/mpmd_sharding_propagation_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "convert_sdy_constants.cc",
        "convert_sdy_shardings_to_mpmd_types.cc",
        "enforce_user_shardings.cc",
        "extract_reshards_from_inter_mesh_transfers.cc",
        "populate_unreduced_out_sharding.cc",
        "sharding_propagation_pipeline.cc",
        "simplify_program.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":passes_inc",
        ":utils",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/transforms/common:distributed_function_pass",
        "//shardy/dialect/mpmd/transforms/common:passes",
        "//shardy/dialect/mpmd/transforms/common:simplify_region_op_base",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/export:explicit_reshards_util",
        "//shardy/dialect/sdy/transforms/propagation:passes",
        "//shardy/dialect/sdy/transforms/propagation:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes_optimization",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
