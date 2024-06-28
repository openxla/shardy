load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("//third_party/py:symlink_files.bzl", "symlink_files", "symlink_inputs")

package(default_visibility = ["//visibility:public"])

symlink_inputs(
    name = "core",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "dialects": ["@llvm-project//mlir/python:DialectCorePyFiles"],
    }},
)

symlink_inputs(
    name = "ir",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        ".": ["@llvm-project//mlir/python:IRPyFiles"],
    }},
    deps = [
        ":mlir",
    ],
)

symlink_inputs(
    name = "mlir_libs",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "_mlir_libs/": ["@llvm-project//mlir/python:MlirLibsPyFiles"],
    }},
)

symlink_inputs(
    name = "runtime",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        "runtime": ["@llvm-project//mlir/python:RuntimePyFiles"],
    }},
    deps = [
        ":mlir",
    ],
)

py_library(
    name = "mlir",
    imports = [".."],
    deps = [
        ":mlir_libs",
        "//third_party/py/mlir/_mlir_libs:_mlir",
    ],
)
