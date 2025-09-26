# The SDY import passes and pipeline.

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
            "-name=SdyImport",
        ],
        "passes.capi.h.inc": [
            "-gen-pass-capi-header",
        ],
        "g3doc/sdy_import_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "add_data_flow_edges.cc",
        "apply_sharding_constraints.cc",
        "constant_or_scalar_splitter.cc",
        "import_pipeline.cc",
        "inline_meshes.cc",
        "lift_inlined_meshes.cc",
        "manual_axes_cleanup.cc",
        "remove_size_one_axes.cc",
        "sharding_group_import.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":passes_inc",
        "//shardy/common:file_utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:op_properties",
        "//shardy/dialect/sdy/transforms/common:propagation_options",
        "//shardy/dialect/sdy/transforms/common:sharding_walker",
        "//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_registry",
        "//shardy/dialect/sdy/transforms/propagation:sharding_projection",
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
