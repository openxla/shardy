# The SDY MLIR dialect.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(default_visibility = ["//visibility:public"])

td_library(
    name = "sdy_td_files",
    srcs = [
        "attrs.td",
        "canonicalization.td",
        "dialect.td",
        "enums.td",
        "ops.td",
    ],
    deps = [
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:BytecodeOpInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [
        ":sdy_td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [
        ":sdy_td_files",
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
    deps = [":sdy_td_files"],
)

gentbl_cc_library(
    name = "enums_inc",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "enums.td",
    deps = [":sdy_td_files"],
)

gentbl_cc_library(
    name = "canonicalization_inc",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "canonicalization.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "canonicalization.td",
    deps = [":sdy_td_files"],
)

gentbl_filegroup(
    name = "dialect_doc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-doc",
                "-dialect=sdy",
            ],
            "g3doc/sdy_dialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [":sdy_td_files"],
)

cc_library(
    name = "dialect",
    srcs = [
        "canonicalization.cc",
        "data_flow_utils.cc",
        "dialect.cc",
        "parsers.cc",
        "printers.cc",
        "utils.cc",
        "verifiers.cc",
    ],
    hdrs = [
        "constants.h",
        "data_flow_utils.h",
        "dialect.h",
        "parsers.h",
        "printers.h",
        "utils.h",
    ],
    deps = [
        ":attrs_inc",
        ":canonicalization_inc",
        ":dialect_inc",
        ":enums_inc",
        ":ops_inc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_assembly_format",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_type_inference",
    ],
)

cc_test(
    name = "dialect_test",
    srcs = ["dialect_test.cc"],
    deps = [
        ":dialect",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
