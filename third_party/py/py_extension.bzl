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
"""Supports writing Python modules in C++."""

# buildifier: disable=unused-variable
def py_extension(
        name = None,
        srcs = None,
        hdrs = None,
        data = None,
        features = None,
        visibility = None,
        deps = None,
        copts = None,
        linkflags = None):
    """Creates a Python module implemented in C++.

    Python modules can depend on a py_extension. Other py_extensions can depend
    on a generated C++ library named with "_cc" suffix.

    Args:
      name: Name for this target.
      srcs: C++ source files.
      hdrs: C++ header files, for other py_extensions which depend on this.
      data: Files needed at runtime. This may include Python libraries.
      features: Passed to cc_library.
      visibility: Controls which rules can depend on this.
      deps: Other C++ libraries that this library depends upon.
      copts: Not used.
      linkflags: Not used.
    """

    cc_library_name = name + "_cc"
    cc_binary_name = name + ".so"

    native.cc_library(
        name = cc_library_name,
        srcs = srcs,
        hdrs = hdrs,
        data = data,
        features = features,
        visibility = visibility,
        deps = deps,
        alwayslink = True,
    )

    native.cc_binary(
        name = cc_binary_name,
        linkshared = True,
        linkstatic = True,
        visibility = ["//visibility:private"],
        deps = [cc_library_name],
    )

    native.py_library(
        name = name,
        data = [cc_binary_name],
        visibility = visibility,
    )
