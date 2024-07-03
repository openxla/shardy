# Copyright 2024 The Shardy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
