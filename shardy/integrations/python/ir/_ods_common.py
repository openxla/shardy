# Copyright 2024 The Shardy Authors.
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
# ==============================================================================
"""A trampoline to make the generated Python files run."""

from mlir.dialects._ods_common import _cext

_cext.globals.append_dialect_search_prefix('openxla')

# pylint: disable=unused-import, g-multiple-import
from mlir.dialects._ods_common import _cext, segmented_accessor, equally_sized_accessor, get_default_loc_context, get_op_result_or_value, get_op_results_or_values, get_op_result_or_op_results
