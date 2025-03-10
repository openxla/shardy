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
"""Workspace for Shardy."""

workspace(name = "shardy")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

###############################
# Initialize non-hermetic Python

rules_python_version = "0.30.0"

http_archive(
    name = "rules_python",
    strip_prefix = "rules_python-{}".format(rules_python_version),
    url = "https://github.com/bazelbuild/rules_python/releases/download/{}/rules_python-{}.tar.gz".format(rules_python_version, rules_python_version),
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_version = "3.11"

python_register_toolchains(
    name = "local_config_python",  #"python_{}".format(python_version_),
    # Available versions are listed in @rules_python//python:versions.bzl.
    # We recommend using the same version your team is already standardized on.
    python_version = python_version,
)

###############################
#TF workspace 4

###############################
#TF/TSL workspace 3

#These need to come in this specific order otherwise bazel complains of missing/circular dependencies.
load("//third_party/llvm:workspace.bzl", llvm = "repo")

llvm("llvm-raw")

load("//third_party/llvm:setup.bzl", "llvm_setup")

llvm_setup("llvm-project")

###############################
#TF/TSL workspace 2

#register_execution_platforms("@local_execution_config_platform//:platform")
#register_toolchains("@local_execution_config_python//:py_toolchain")

###############################

http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-2.12.0",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/v2.12.0.zip"],
)

# We still require the pybind library.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    strip_prefix = "pybind11-2.12",
    urls = ["https://github.com/pybind/pybind11/archive/v2.12.zip"],
)

###############################

load("//third_party/stablehlo:workspace.bzl", stablehlo = "repo")

stablehlo()

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-5ab508a01f9eb089207ee87fd547d290da39d015",
    urls = ["https://github.com/google/googletest/archive/5ab508a01f9eb089207ee87fd547d290da39d015.zip"],
)
