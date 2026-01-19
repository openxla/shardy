# The MPMD MLIR dialect.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")
# load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(default_visibility = ["//visibility:public"])

td_library(
    name = "mpmd_td_files",
    srcs = [
        "attrs.td",
        "dialect.td",
        "enums.td",
        "ops.td",
        "types.td",
    ],
    deps = [
        "//shardy/dialect/sdy/ir:sdy_td_files",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@stablehlo//:base_td_files",
    ],
)

gentbl_cc_library(
    name = "dialect_inc",
    tbl_outs = {
        "dialect.h.inc": ["-gen-dialect-decls"],
        "dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [
        ":mpmd_td_files",
    ],
)

gentbl_cc_library(
    name = "canonicalization_inc",
    tbl_outs = {"canonicalization.cc.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "canonicalization.td",
    deps = [":mpmd_td_files"],
)

gentbl_cc_library(
    name = "ops_inc",
    tbl_outs = {
        "ops.h.inc": ["-gen-op-decls"],
        "ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [":mpmd_td_files"],
)

gentbl_cc_library(
    name = "types_inc",
    tbl_outs = {
        "types.h.inc": ["-gen-typedef-decls"],
        "types.cc.inc": ["-gen-typedef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "types.td",
    deps = [
        ":mpmd_td_files",
    ],
)

gentbl_cc_library(
    name = "attrs_inc",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attrs.td",
    deps = [":mpmd_td_files"],
)

gentbl_cc_library(
    name = "enums_inc",
    tbl_outs = {
        "enums.h.inc": ["-gen-enum-decls"],
        "enums.cc.inc": ["-gen-enum-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "enums.td",
    deps = [":mpmd_td_files"],
)

gentbl_filegroup(
    name = "dialect_doc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-doc",
                "-dialect=mpmd",
            ],
            "g3doc/mpmd_dialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [":mpmd_td_files"],
)

cc_library(
    name = "dialect",
    srcs = [
        "dialect.cc",
        "utils.cc",
    ],
    hdrs = [
        "dialect.h",
        "utils.h",
    ],
    deps = [
        ":attrs_inc",
        ":canonicalization_inc",
        ":dialect_inc",
        ":enums_inc",
        ":fragment_arg_res_attrs",
        ":ops_inc",
        ":types_inc",
        "//shardy/common:logging",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
    ],
)

cc_library(
    name = "register",
    srcs = ["register.cc"],
    hdrs = ["register.h"],
    deps = [
        ":dialect",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/ir:register",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "fragment_arg_res_attrs",
    hdrs = ["fragment_arg_res_attrs.h"],
    deps = [
        "//shardy/common:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "fragment_execution_rules",
    srcs = ["fragment_execution_rules.cc"],
    hdrs = ["fragment_execution_rules.h"],
    deps = [
        ":dialect",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "fragment_execution_rules_test",
    srcs = ["fragment_execution_rules_test.cc"],
    deps = [
        ":dialect",
        ":fragment_execution_rules",
        ":register",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":dialect",
        ":register",
        "//shardy/common:logging",
        "//shardy/dialect/sdy/ir:dialect",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)
