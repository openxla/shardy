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

load("//third_party/py:py_extension.bzl", "py_extension")
load("//third_party/py:symlink_files.bzl", "symlink_files")

package(
    default_visibility = ["//visibility:public"],
)

# These flags are needed for parse_headers feature.
COPTS = [
    "-fexceptions",
    "-frtti",
]

##---------------------------------------------------------------------------##
# Python extensions.
##---------------------------------------------------------------------------##

py_extension(
    name = "_mlir",
    srcs = [
        "@llvm-project//mlir:lib/Bindings/Python/MainModule.cpp",
    ],
    copts = COPTS,
    # pytype_srcs = [
    #     ":_mlirPyi",
    # ],
    deps = [
        "@llvm-project//mlir:MLIRBindingsPythonCore",
        "@llvm-project//mlir:MLIRBindingsPythonHeadersAndDeps",
    ],
)
