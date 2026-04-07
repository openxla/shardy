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
"""Utils for MPMD partitioning."""

from collections.abc import Callable, Mapping, Sequence, Set
import dataclasses
import functools
import logging
from typing import Any, TypeVar

import jax
from jaxlib import _sdy_mpmd as mpmd_utils
from jax.typing import ArrayLike
import jaxtyping

PyTree = jaxtyping.PyTree

X = TypeVar('X')
Y = TypeVar('Y')
T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class FunctionNamedShardings:
  """Stores trees of NamedShardings for inputs and outputs."""

  input_specs: PyTree[jax.sharding.NamedSharding]
  output_specs: PyTree[jax.sharding.NamedSharding]


def _sdy_spec_to_named_sharding(
    sdy_sharding: Sequence[Sequence[str]],
    mesh: jax.sharding.Mesh,
    unreduced_axes: Set[str] | None = None,
    memory_kind: str | None = None,
) -> jax.sharding.NamedSharding:
  """Converts a SDY spec to a NamedSharding."""

  def get_pspec(sdy_sharding):
    cloned_sharding = list(sdy_sharding)
    while cloned_sharding:
      dim_sharding = cloned_sharding[-1]
      if dim_sharding:
        break
      else:
        # Remove trailing replicated dimensions so we don't get a PartitionSpec
        # with trailing Nones.
        cloned_sharding.pop()

    unreduced = unreduced_axes or set()
    if not cloned_sharding:
      return jax.sharding.PartitionSpec(unreduced=unreduced)
    return jax.sharding.PartitionSpec(*cloned_sharding, unreduced=unreduced)

  return jax.sharding.NamedSharding(
      mesh, get_pspec(sdy_sharding), memory_kind=memory_kind
  )


def meshes_and_sdy_specs_to_named_shardings(
    nr_const_args: int,
    meshes_and_specs: mpmd_utils.FunctionIOShardingSpecsAndMeshes,
    input_tree_def: jax.tree_util.PyTreeDef,
    output_tree_def: jax.tree_util.PyTreeDef,
    topology: Mapping[str, jax.sharding.Mesh],
) -> FunctionNamedShardings:
  """Builds a FunctionNamedShardings object."""
  flat_input_named_shardings = [
      _sdy_spec_to_named_sharding(
          input_spec.tensor_spec,
          topology[input_spec.mesh_name],
          unreduced_axes=set(input_spec.unreduced_axes),
          memory_kind=input_spec.memory_kind,
      )
      for input_spec in meshes_and_specs.input_specs[nr_const_args:]
  ]

  flat_output_named_shardings = [
      _sdy_spec_to_named_sharding(
          output_spec.tensor_spec,
          topology[output_spec.mesh_name],
          unreduced_axes=set(output_spec.unreduced_axes),
          memory_kind=output_spec.memory_kind,
      )
      for output_spec in meshes_and_specs.output_specs
  ]
  return FunctionNamedShardings(
      jax.tree.unflatten(input_tree_def, flat_input_named_shardings),
      jax.tree.unflatten(output_tree_def, flat_output_named_shardings),
  )


class InvalidUnusedArgsInfoError(Exception):
  """Raised when unused args info is not consistent with the function specs."""


@dataclasses.dataclass(frozen=True)
class JaxFunctionInfo:
  """Information about the Jax function that is being partitioned.

  Attributes:
    func_name: the name of the function being partitioned.
    global_flat_input_abstract_values: a list of abstract values (one for each
      input) that the user passed to partir.jit, which includes unused args.
    global_flat_output_abstract_values: a list of abstract values (one for each
      output).
    input_tree: the structure of input tree that the user passed to partir.jit,
      which includes unused args.
    output_tree: the structure of output tree.
    kept_inputs_indices: Indices of the kept inputs of the Jax function after
      removing unused args.
    const_args: The closed-over constants in the Jax function.
      In presence of closed-over constants, `global_flat_input_abstract_values`
      does not include the constants, but `kept_inputs_indices` does.
      This is empty unless JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS is True.
      See https://docs.jax.dev/en/latest/internals/constants.html.
  """

  func_name: str
  global_flat_input_abstract_values: list[jax.core.ShapedArray]
  global_flat_output_abstract_values: list[jax.core.ShapedArray]
  input_tree: jax.tree_util.PyTreeDef
  output_tree: jax.tree_util.PyTreeDef
  kept_inputs_indices: Set[int]
  const_args: list[ArrayLike]

  def with_placeholder_for_removed_inputs(
      self,
      data_with_unused_removed: Sequence[mpmd_utils.NamedSpmdShardingSpec],
      placeholder: mpmd_utils.NamedSpmdShardingSpec,
  ) -> list[mpmd_utils.NamedSpmdShardingSpec]:
    """Returns the data with placeholders in the removed args positions.

    Args:
      data_with_unused_removed: Sequence of data where the unused arguments are
        removed. It has length `len(self.kept_inputs_indices)` and it includes
        closed-over constants if any.
      placeholder: Object to be inserted into the unused arguments positions.

    Raises:
      InvalidUnusedArgsInfoError: if data_with_unused_removed is inconsistent
        with self.kept_inputs_indices.
    """
    nr_all_inputs = (
        len(self.const_args) + len(self.global_flat_input_abstract_values)
    )
    data_with_all_inputs = [placeholder for _ in range(nr_all_inputs)]
    if len(self.kept_inputs_indices) != len(data_with_unused_removed):
      raise InvalidUnusedArgsInfoError(
          'Invalid unused args info. Cannot map '
          f'{len(self.kept_inputs_indices)} `kept indices` to '
          f'{len(data_with_unused_removed)} available function arguments.'
      )
    for i, kept_input_idx in enumerate(sorted(self.kept_inputs_indices)):
      data_with_all_inputs[kept_input_idx] = data_with_unused_removed[i]

    return data_with_all_inputs


def get_func_name(func: Callable[..., Any], prefix: str = 'mpmd_') -> str:
  """Attempts to determine a name for func."""
  if hasattr(func, '__name__'):
    return f'{prefix}{func.__name__}'
  elif isinstance(func, functools.partial) and hasattr(func.func, '__name__'):
    return f'{prefix}{func.func.__name__}'
  else:
    return f'{prefix}fn'
