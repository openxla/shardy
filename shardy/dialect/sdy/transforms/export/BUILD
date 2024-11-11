# The SDY export passes and pipeline.

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
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SdyExport",
            ],
            "passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "g3doc/sdy_export_passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "close_shardings.cc",
        "export_pipeline.cc",
        "insert_explicit_reshards.cc",
        "remove_sharding_groups.cc",
        "sharding_constraint_to_reshard.cc",
        "sink_data_flow_edges.cc",
        "update_non_divisible_input_output_shardings.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":passes_inc",
        "//shardy/common:file_utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:op_properties",
        "//shardy/dialect/sdy/transforms/common:sharding_walker",
        "//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_registry",
        "//shardy/dialect/sdy/transforms/propagation:sharding_projection",
        "//shardy/dialect/sdy/transforms/propagation:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)
