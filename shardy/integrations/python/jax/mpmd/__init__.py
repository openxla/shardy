# Copyright 2025 The MPMD Authors.
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
"""Shardy MPMD JAX API."""


from shardy.integrations.python.jax.mpmd.jit import jit
from shardy.integrations.python.jax.mpmd.jit import MpmdWrapped
from shardy.integrations.python.jax.mpmd.ops import broadcast
from shardy.integrations.python.jax.mpmd.ops import call
from shardy.integrations.python.jax.mpmd.ops import enumerated_for
from shardy.integrations.python.jax.mpmd.ops import fori_loop
from shardy.integrations.python.jax.mpmd.ops import named_computation
from shardy.integrations.python.jax.mpmd.ops import named_tensor
from shardy.integrations.python.jax.mpmd.ops import reduce
from shardy.integrations.python.jax.mpmd.pipeline import FragmentInfo
from shardy.integrations.python.jax.mpmd.pipeline import FragmentMergeRule
from shardy.integrations.python.jax.mpmd.pipeline import FragmentMergeRules
from shardy.integrations.python.jax.mpmd.pipeline import FragmentOrigin
from shardy.integrations.python.jax.mpmd.stages import MpmdCompiled as Compiled
from shardy.integrations.python.jax.mpmd.stages import MpmdExecutable as Executable
from shardy.integrations.python.jax.mpmd.stages import MpmdJitShardingInfo
from shardy.integrations.python.jax.mpmd.stages import MpmdLowered as Lowered
from shardy.integrations.python.jax.mpmd.types import FunctionIOMeshAssignment
from shardy.integrations.python.jax.mpmd.types import make_config
from shardy.integrations.python.jax.mpmd.types import mesh_names
from shardy.integrations.python.jax.mpmd.types import MpmdConfig as Config
from shardy.integrations.python.jax.mpmd.types import NameToMeshAssignment
from shardy.integrations.python.jax.mpmd.types import PartitioningOptions
from shardy.integrations.python.jax.mpmd.types import Topology
from shardy.integrations.python.jax.mpmd.utils import FunctionNamedShardings


__all__ = (
    "Compiled",
    "Config",
    "Executable",
    "FragmentInfo",
    "FragmentMergeRule",
    "FragmentMergeRules",
    "FragmentOrigin",
    "FunctionIOMeshAssignment",
    "FunctionNamedShardings",
    "Lowered",
    "MpmdWrapped",
    "MpmdJitShardingInfo",
    "NameToMeshAssignment",
    "PartitioningOptions",
    "Topology",
    "broadcast",
    "call",
    "enumerated_for",
    "fori_loop",
    "jit",
    "make_config",
    "mesh_names",
    "named_computation",
    "named_tensor",
    "reduce",
)
