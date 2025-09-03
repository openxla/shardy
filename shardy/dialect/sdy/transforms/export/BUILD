# The SDY export passes and pipeline.

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
            "-name=SdyExport",
        ],
        "g3doc/sdy_export_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "close_shardings.cc",
        "constant_or_scalar_merger.cc",
        "drop_sharding_rules.cc",
        "export_pipeline.cc",
        "insert_explicit_reshards.cc",
        "remove_propagation_debug_info.cc",
        "remove_sharding_groups.cc",
        "reshard_to_collectives.cc",
        "sharding_constraint_to_reshard.cc",
        "sink_data_flow_edges.cc",
        "update_non_divisible_input_output_shardings.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":explicit_reshards_util",
        ":passes_inc",
        "//shardy/common:file_utils",
        "//shardy/common:logging",
        "//shardy/dialect/sdy/ir:axis_list_ref",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:op_properties",
        "//shardy/dialect/sdy/transforms/common:sharding_walker",
        "//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_registry",
        "//shardy/dialect/sdy/transforms/propagation:sharding_projection",
        "//shardy/dialect/sdy/transforms/propagation:utils",
        "//shardy/dialect/sdy/transforms/propagation/debugging:source_sharding",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
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
    name = "explicit_reshards_util",
    srcs = ["explicit_reshards_util.cc"],
    hdrs = ["explicit_reshards_util.h"],
    deps = [
        "//shardy/common:file_utils",
        "//shardy/common:logging",
        "//shardy/dialect/sdy/ir:axis_list_ref",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:op_properties",
        "//shardy/dialect/sdy/transforms/common:sharding_walker",
        "//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_registry",
        "//shardy/dialect/sdy/transforms/propagation:sharding_projection",
        "//shardy/dialect/sdy/transforms/propagation:utils",
        "//shardy/dialect/sdy/transforms/propagation/debugging:source_sharding",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)
