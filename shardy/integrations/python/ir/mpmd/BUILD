# Python IR bindings for the SDY dialect.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup", "td_library")

package(default_visibility = ["//visibility:public"])

exports_files(["mpmd_module.cc"])

td_library(
    name = "mpmd_ops_td_files",
    srcs = [
        "mpmd_ops.td",
    ],
    deps = [
        "//shardy/dialect/mpmd/ir:mpmd_td_files",
    ],
)

gentbl_filegroup(
    name = "mpmd_ops_py_gen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=mpmd",
            ],
            "_mpmd_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mpmd_ops.td",
    deps = [
        ":mpmd_ops_td_files",
    ],
)

filegroup(
    name = "mpmd_ops_py_files",
    srcs = [
        "mpmd.py",
        ":mpmd_ops_py_gen",
    ],
)
