# Python IR bindings for the SDY dialect.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup", "td_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("//third_party/py:py_extension.bzl", "py_extension")

package(default_visibility = ["//visibility:public"])

exports_files(["sdy_module.cc"])

td_library(
    name = "sdy_ops_td_files",
    srcs = [
        "sdy_ops.td",
    ],
    deps = [
        "//shardy/dialect/sdy/ir:sdy_td_files",
    ],
)

filegroup(
    name = "sdy_ops_py_files",
    srcs = [
        ":sdy_ops_py_gen",
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

py_library(
    name = "ir",
    srcs = [
        "__init__.py",
        "_ods_common.py",
        ":sdy_ops_py_gen",
    ],
    deps = [
        ":_sdy",
        "//third_party/py/mlir",
        "//third_party/py/mlir:core",
        "//third_party/py/mlir:ir",
    ],
)

py_extension(
    name = "_sdy",
    srcs = [
        "sdy_module.cc",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = [
        # Cannot use header_modules (parse_headers feature fails). Needed to get the extension
        # to build.
        "-use_header_modules",
    ],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/integrations/c:sdy_capi",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeadersAndDeps",
    ],
)
