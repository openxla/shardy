# The SDY sharding propagation system.

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
            "-name=SdyPropagation",
        ],
        "g3doc/sdy_propagation_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "aggressive_propagation.cc",
        "basic_propagation.cc",
        "op_priority_propagation.cc",
        "populate_op_sharding_rules.cc",
        "propagation_pipeline.cc",
        "user_priority_propagation.cc",
    ],
    hdrs = [
        "aggressive_propagation.h",
        "basic_propagation.h",
        "op_priority_propagation.h",
        "passes.h",
        "user_priority_propagation.h",
    ],
    deps = [
        ":aggressive_factor_propagation",
        ":auto_partitioner_registry",
        ":basic_factor_propagation",
        ":factor_propagation",
        ":op_sharding_rule_builder",
        ":op_sharding_rule_registry",
        ":passes_inc",
        ":sharding_group_map",
        ":sharding_projection",
        ":utils",
        "//shardy/common:file_utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:op_properties",
        "//shardy/dialect/sdy/transforms/common:sharding_walker",
        "//shardy/dialect/sdy/transforms/export:passes",
        "//shardy/dialect/sdy/transforms/import:passes",
        "//shardy/dialect/sdy/transforms/propagation/debugging:source_sharding",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
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

cc_library(
    name = "op_sharding_rule_builder",
    srcs = ["op_sharding_rule_builder.cc"],
    hdrs = ["op_sharding_rule_builder.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "op_sharding_rule_registry",
    srcs = ["op_sharding_rule_registry.cc"],
    hdrs = ["op_sharding_rule_registry.h"],
    deps = [
        ":op_sharding_rule_builder",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "sharding_group_map",
    srcs = ["sharding_group_map.cc"],
    hdrs = ["sharding_group_map.h"],
    deps = [
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "sharding_projection",
    srcs = ["sharding_projection.cc"],
    hdrs = ["sharding_projection.h"],
    deps = [
        ":utils",
        "//shardy/dialect/sdy/ir:axis_list_ref",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "sharding_projection_test",
    srcs = ["sharding_projection_test.cc"],
    deps = [
        ":op_sharding_rule_registry",
        ":sharding_projection",
        ":testing_utils",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/ir:testing_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "auto_partitioner_registry",
    srcs = ["auto_partitioner_registry.cc"],
    hdrs = ["auto_partitioner_registry.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_test(
    name = "auto_partitioner_registry_test",
    srcs = ["auto_partitioner_registry_test.cc"],
    deps = [
        ":auto_partitioner_registry",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "testing_utils",
    hdrs = ["testing_utils.h"],
    deps = [
        ":sharding_projection",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "factor_propagation",
    hdrs = ["factor_propagation.h"],
    deps = [
        ":sharding_projection",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:macros",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "basic_factor_propagation",
    srcs = ["basic_factor_propagation.cc"],
    hdrs = ["basic_factor_propagation.h"],
    deps = [
        ":factor_propagation",
        ":sharding_projection",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:macros",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "basic_factor_propagation_test",
    srcs = ["basic_factor_propagation_test.cc"],
    deps = [
        ":basic_factor_propagation",
        ":factor_propagation",
        ":sharding_projection",
        ":testing_utils",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/ir:testing_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "aggressive_factor_propagation",
    srcs = ["aggressive_factor_propagation.cc"],
    hdrs = ["aggressive_factor_propagation.h"],
    deps = [
        ":basic_factor_propagation",
        ":factor_propagation",
        ":sharding_projection",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:macros",
        "//shardy/dialect/sdy/transforms/common:op_properties",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "aggressive_factor_propagation_test",
    srcs = ["aggressive_factor_propagation_test.cc"],
    deps = [
        ":aggressive_factor_propagation",
        ":factor_propagation",
        ":sharding_projection",
        ":testing_utils",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/ir:testing_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)
