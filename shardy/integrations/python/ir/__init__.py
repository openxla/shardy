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
"""Python bindings for the SDY dialect."""

# pylint: disable=g-multiple-import,g-importing-member,unused-import,useless-import-alias
from ._sdy import (
    register_dialect as register_dialect,
    MeshAxisAttr as MeshAxisAttr,
    MeshAttr as MeshAttr,
    SubAxisInfoAttr as SubAxisInfoAttr,
    AxisRefAttr as AxisRefAttr,
    DimensionShardingAttr as DimensionShardingAttr,
    TensorShardingAttr as TensorShardingAttr,
    TensorShardingPerValueAttr as TensorShardingPerValueAttr,
)

from ._sdy_enums_gen import PropagationDirection as PropagationDirection

from ._sdy_ops_gen import (
    ConstantOp as ConstantOp,
    ManualComputationOp as ManualComputationOp,
    MeshOp as MeshOp,
    ReshardOp as ReshardOp,
    ReturnOp as ReturnOp,
    ShardingConstraintOp as ShardingConstraintOp,
)
