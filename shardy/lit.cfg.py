# Copyright 2024 The Shardy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lit configuration to drive test in this repo."""

# -*- Python -*-
# pylint: disable=undefined-variable

import os

import lit.formats
from lit.llvm import llvm_config

# Populate Lit configuration with the minimal required metadata.
# Some metadata is populated in lit.site.cfg.py.in.
config.name = 'SHARDY_TESTS_SUITE'
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

# Disallow reusing variables across CHECK-LABEL matches.
# A variable can eschew this (be made "global") by prefixing its name with $.
config.environment['FILECHECK_OPTS'] = '-enable-var-scope'

# Make LLVM and Shardy tools available in RUN directives
tools = [
    'FileCheck',
    'sdy_opt',
]
tool_dirs = [
    config.llvm_tools_dir,
    config.shardy_tools_dir,
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
