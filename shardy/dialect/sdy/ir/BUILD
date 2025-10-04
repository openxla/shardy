# The SDY MLIR dialect.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")
# load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

td_library(
    name = "sdy_td_files",
    srcs = [
        "attrs.td",
        "canonicalization.td",
        "dialect.td",
        "enums.td",
        "op_interface.td",
        "ops.td",
    ],
    deps = [
        "@llvm-project//mlir:AttrTdFiles",
        "@llvm-project//mlir:BuiltinDialectBytecodeTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:BytecodeOpInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "op_interface_inc",
    tbl_outs = {
        "op_interface.h.inc": ["-gen-op-interface-decls"],
        "op_interface.cc.inc": ["-gen-op-interface-defs"],
        "g3doc/sdy_op_interfaces.md": ["-gen-op-interface-docs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "op_interface.td",
    deps = [
        ":sdy_td_files",
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
        ":sdy_td_files",
    ],
)

gentbl_cc_library(
    name = "ops_inc",
    tbl_outs = {
        "ops.h.inc": ["-gen-op-decls"],
        "ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [
        ":sdy_td_files",
    ],
)

gentbl_cc_library(
    name = "attrs_inc",
    tbl_outs = {
        "attrs.h.inc": ["-gen-attrdef-decls"],
        "attrs.cc.inc": ["-gen-attrdef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attrs.td",
    deps = [":sdy_td_files"],
)

gentbl_cc_library(
    name = "enums_inc",
    tbl_outs = {
        "enums.h.inc": ["-gen-enum-decls"],
        "enums.cc.inc": ["-gen-enum-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "enums.td",
    deps = [":sdy_td_files"],
)

gentbl_cc_library(
    name = "canonicalization_inc",
    tbl_outs = {"canonicalization.cc.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "canonicalization.td",
    deps = [":sdy_td_files"],
)

gentbl_cc_library(
    name = "bytecode_inc",
    tbl_outs = {"bytecode.cc.inc": [
        "-gen-bytecode",
        "-bytecode-dialect=sdy",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "bytecode.td",
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
        "bytecode.cc",
        "canonicalization.cc",
        "compatibility.cc",
        "dialect.cc",
        "extensions/stablehlo_extensions.cc",
        "parsers.cc",
        "printers.cc",
        "utils.cc",
        "verifiers.cc",
    ],
    hdrs = [
        "bytecode.h",
        "compatibility.h",
        "constants.h",
        "dialect.h",
        "enums.h",
        "extensions/stablehlo_extensions.h",
        "parsers.h",
        "printers.h",
        "utils.h",
        "verifiers.h",
    ],
    deps = [
        ":attrs_inc",
        ":bytecode_inc",
        ":canonicalization_inc",
        ":dialect_inc",
        ":enums_inc",
        ":macros",
        ":op_interface_inc",
        ":ops_inc",
        "//shardy/common:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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
        ":testing_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "register",
    srcs = ["register.cc"],
    hdrs = ["register.h"],
    deps = [
        ":dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "axis_list_ref",
    srcs = ["axis_list_ref.cc"],
    hdrs = ["axis_list_ref.h"],
    deps = [
        ":dialect",
        ":macros",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "testing_utils",
    hdrs = ["testing_utils.h"],
    deps = [
        ":dialect",
        ":register",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "axis_list_ref_test",
    srcs = ["axis_list_ref_test.cc"],
    deps = [
        ":axis_list_ref",
        ":dialect",
        ":register",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":dialect",
        ":testing_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)
