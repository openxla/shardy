# Python IR bindings for the SDY dialect.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup", "td_library")

package(default_visibility = ["//visibility:public"])

exports_files(["sdy_module.cc"])

td_library(
    name = "sdy_enums_td_files",
    srcs = [
        "sdy_enums.td",
    ],
    deps = [
        "//shardy/dialect/sdy/ir:sdy_td_files",
    ],
)

gentbl_filegroup(
    name = "sdy_enums_py_gen",
    tbl_outs = [
        (
            [
                "-gen-python-enum-bindings",
                "-bind-dialect=sdy",
            ],
            "_sdy_enums_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "sdy_enums.td",
    deps = [
        ":sdy_enums_td_files",
    ],
)

td_library(
    name = "sdy_ops_td_files",
    srcs = [
        "sdy_ops.td",
    ],
    deps = [
        "//shardy/dialect/sdy/ir:sdy_td_files",
    ],
)

gentbl_filegroup(
    name = "sdy_ops_py_gen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=sdy",
            ],
            "_sdy_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "sdy_ops.td",
    deps = [
        ":sdy_ops_td_files",
    ],
)

filegroup(
    name = "sdy_ops_py_files",
    srcs = [
        "sdy.py",
        ":sdy_enums_py_gen",
        ":sdy_ops_py_gen",
    ],
)
