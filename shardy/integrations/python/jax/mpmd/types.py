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

"""Common types used by PartIR:MPMD."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import jax
import jaxtyping

from shardy.integrations.python.jax.mpmd import partitioning_options as part_options


PyTree = jaxtyping.PyTree

# A topology is a dictionary between mesh names and SPMD Meshes.
Topology = dict[str, jax.sharding.Mesh]
# A user-defined mapping between named computations and tensors and mesh names.
NameToMeshAssignment = Mapping[str, str]
# A user-defined mapping between named_computations and stage ids.
NameToStageAssignment = Mapping[str, int]
# A user-defined mapping between mesh names and CompilerOptions overrides. The
# dictionary does not need to contain entries for all the meshes in a topology
# because users can provide compile options overrides only for some meshes.
MeshToCompileOptions = Mapping[str, jax.stages.CompilerOptions]
PartitioningOptions = dict[str, bool | str]


@dataclasses.dataclass(frozen=True)
class FragmentOrigin:
  """The origin of a fragment."""

  computation_name: str
  transpose_count: int = 0


@dataclasses.dataclass(frozen=True)
class FragmentInfo:
  """A fragment of a computation."""

  origins: Sequence[FragmentOrigin]
  stage_id: int | None = None
  call_counter: int | None = None
  is_weight_gradient: bool = False


@dataclasses.dataclass(frozen=True)
class FragmentMergeRule:
  """A rule for merging fragments of a computation."""

  sources: Sequence[FragmentInfo]
  target: FragmentInfo


FragmentMergeRules = Sequence[FragmentMergeRule]


@dataclasses.dataclass(frozen=True)
class MpmdConfig:
  """Config for constructing an MPMD program with PartIR.

  Attributes:
    topology: A topology of SPMD Meshes. A topology is a mapping between mesh
      names and SPMD Meshes. Mesh names cannot have the `@cpu` suffix.
    name_to_mesh_assignment: A mapping between named computations and tensors to
      mesh names. Each mesh name is expected to be found in the topology.
    name_to_stage_assignment: A mapping between named computations and stage
      ids. Each stage id must be a positive integer. Two named_computations will
      be merged into the same computation iff they have the same stage id,
      transpose count (e.g., both are forward computations) and call_counter, if
      defined (c.f. mpmd.call docs).
    input_mesh_assignment: An assignment of inputs to mesh names (or None if
      unassigned), which must be a pytree prefix (e.g. one value in place of a
      whole subtree) of the input tree, in which case the leaves of the
      assignment are broadcasted to all values in the respective input subtree.
    output_mesh_assignment: An assignment of outputs to mesh names (or None if
      unassigned), which must be a pytree prefix (e.g. one value in place of a
      whole subtree) of the input tree, in which case the leaves of the
      assignment are broadcasted to all values in the respective input subtree.
    partitioning_options: The partitioning options to use for the MPMD
      partitioning. If left undefined, PartIR will read these options from absl
      flags. See `PartitioningOptions` for more details.
    spmd_mesh: A mesh representative of any mesh in the topology for SPMD
      partitioning purposes.
    read_input_output_mesh_from_shardings: If set, we will ignore the
      input_mesh_assignment and output_mesh_assignment and read the mesh
      assignment from the mpmd.jit in_shardings and out_shardings. Currently
      reading from arg shardings is not supported. TODO: b/377706756 - Read from
      arg shardings too, and migrate users to this and remove this option once
      stabilized.
    fragment_merge_rules: A sequence of fragment merge rules. Each merge rule
      contains a sequence of fragment metadata objects that should be merged
      into a single fragment, together with metadata for the resulting fragment.
  """

  topology: Topology
  name_to_mesh_assignment: NameToMeshAssignment
  name_to_stage_assignment: NameToStageAssignment
  input_mesh_assignment: PyTree[str | None]
  output_mesh_assignment: PyTree[str | None]
  partitioning_options: PartitioningOptions | None
  read_input_output_mesh_from_shardings: bool
  fragment_merge_rules: FragmentMergeRules | None

  @property
  def _spmd_mesh(self) -> jax.sharding.Mesh:
    return list(self.topology.values())[0]

  @property
  def sharding_mesh(self) -> jax.sharding.AbstractMesh:
    """Returns an abstract mesh used for sharding annotations.

    This allows us to differentiate between the mesh used for SPMD and the mesh
    used for MPMD.
    """
    return self._spmd_mesh.abstract_mesh

  @property
  def mesh_and_stage_assignment(self) -> dict[str, str | tuple[str, int]]:
    """Returns the merged the mesh and stage assignment into a single dict."""
    assignment = {}
    for name, mesh in self.name_to_mesh_assignment.items():
      if name in self.name_to_stage_assignment:
        assignment[name] = (mesh, self.name_to_stage_assignment[name])
      else:
        assignment[name] = mesh
    return assignment


def _check_no_reserved_mesh_name(topology: Topology) -> None:
  """Checks that the topology does not contain reserved mesh names."""
  for mesh_name in topology:
    if mesh_name.endswith('#pinned_host') or mesh_name.endswith('#device'):
      raise ValueError(
          f'Mesh name {mesh_name} (or any name suffixed with #pinned_host or'
          ' #device) is reserved for memory_kinds. Please rename it.'
      )


def check_no_reserved_name(topology: Topology) -> None:
  """Checks that the topology does not contain reserved names or characters."""
  reserved = {'@'}
  for mesh_name in topology:
    if any(r in mesh_name for r in reserved):
      raise ValueError(
          f'{mesh_name} uses one of the reserved substrings ({reserved}).'
          ' Please rename the it.'
      )


def _get_assignment_named_without_memory_kind(
    assignment_struct: PyTree[str],
) -> set[str]:
  """Returns the mesh names in `assignment_struct`, without memory kind."""
  return set(
      jax.tree.leaves(
          jax.tree.map(lambda x: x.split('#')[0], assignment_struct)
      )
  )


def make_config(
    topology: Topology,
    name_to_mesh_assignment: NameToMeshAssignment,
    *,
    name_to_stage_assignment: NameToStageAssignment | None = None,
    input_mesh_assignment: PyTree[str | None] = (),
    output_mesh_assignment: PyTree[str | None] = (),
    partitioning_options: PartitioningOptions | None = None,
    read_input_output_mesh_from_shardings: bool = False,
    fragment_merge_rules: FragmentMergeRules | None = None,
) -> MpmdConfig:
  """Creates a `MpmdConfig`, inferring the tpu topology if not provided.

  We assume for now that all meshes are identical. And we pick an arbitrary
  mesh from the mesh topology.

  Args:
    topology: See `MpmdConfig`.
    name_to_mesh_assignment: See `MpmdConfig`.
    name_to_stage_assignment: See `MpmdConfig`.
    input_mesh_assignment: See `MpmdConfig`.
    output_mesh_assignment: See `MpmdConfig`.
    partitioning_options: See `MpmdConfig`.
    read_input_output_mesh_from_shardings: see `MpmdConfig`.
    fragment_merge_rules: See `MpmdConfig`.

  Returns:
    An `MpmdConfig` object.
  """

  if not topology:
    raise ValueError('`topology` must have at least one mesh.')

  check_no_reserved_name(topology)
  _check_no_reserved_mesh_name(topology)
  _validate_partitioning_options(partitioning_options)

  # TODO: b/328231527 - Validate there are no overlapping device ids.

  validate_meshes_in_assignments(
      name_to_mesh_assignment,
      input_mesh_assignment,
      output_mesh_assignment,
      topology,
  )
  validate_stage_assignments(name_to_stage_assignment)
  validate_input_output_mesh_assignments(
      read_input_output_mesh_from_shardings,
      input_mesh_assignment,
      output_mesh_assignment,
  )
  if fragment_merge_rules is None:
    fragment_merge_rules = []
  validate_fragment_merge_rules(fragment_merge_rules)

  return MpmdConfig(
      topology,
      name_to_mesh_assignment,
      name_to_stage_assignment if name_to_stage_assignment else {},
      input_mesh_assignment,
      output_mesh_assignment,
      partitioning_options,
      read_input_output_mesh_from_shardings,
      fragment_merge_rules,
  )


def validate_meshes_in_assignments(
    name_to_mesh_assignment: NameToMeshAssignment,
    input_mesh_assignment: NameToMeshAssignment | None,
    output_mesh_assignment: NameToMeshAssignment | None,
    topology: Topology,
) -> None:
  """Validates that every mesh name in the assignments is actually in the topology."""
  # Check that every mesh name in `name_to_mesh_assignment` is actually in the
  # topology.
  if undefined_mesh_name := _get_assignment_named_without_memory_kind(
      name_to_mesh_assignment
  ) - set(topology):
    raise ValueError(
        f'Invalid mesh name(s): {undefined_mesh_name} not in {topology.keys()}.'
    )

  # Check that every mesh name in input_mesh_assignment and
  # output_mesh_assignment is actually in the topology.
  if undefined_mesh_name := _get_assignment_named_without_memory_kind(
      input_mesh_assignment
  ) - set(topology):
    raise ValueError(
        'Invalid mesh name(s) in input_mesh_assignment:'
        f' {undefined_mesh_name} not in {topology.keys()}.'
    )
  if undefined_mesh_name := _get_assignment_named_without_memory_kind(
      output_mesh_assignment
  ) - set(topology):
    raise ValueError(
        'Invalid mesh name(s) in output_mesh_assignment:'
        f' {undefined_mesh_name} not in {topology.keys()}.'
    )


def validate_stage_assignments(
    name_to_stage_assignment: NameToStageAssignment | None,
) -> None:
  if name_to_stage_assignment and any(
      stage < 0 for stage in name_to_stage_assignment.values()
  ):
    invalid_stages_str = ','.join(
        str(stage) for stage in name_to_stage_assignment.values() if stage < 0
    )
    raise ValueError(
        f'Stage ids must be positive integers, but got: {invalid_stages_str}.'
    )


def validate_input_output_mesh_assignments(
    read_input_output_mesh_from_shardings: bool,
    input_mesh_assignment: NameToMeshAssignment | None,
    output_mesh_assignment: NameToMeshAssignment | None,
) -> None:
  """Validates the input and output mesh assignments."""
  if read_input_output_mesh_from_shardings:
    if input_mesh_assignment:
      raise ValueError(
          'input_mesh_assignment must not be specified when'
          ' read_input_output_mesh_from_shardings is True.'
      )
    if output_mesh_assignment:
      raise ValueError(
          'output_mesh_assignment must not be specified when'
          ' read_input_output_mesh_from_shardings is True.'
      )


def validate_fragment_merge_rules(
    fragment_merge_rules: FragmentMergeRules,
) -> None:
  """Validates the fragment merge rules."""

  for rule in fragment_merge_rules:
    if len(rule.sources) < 2:
      raise ValueError(
          'Fragment merge rule must contain at least two source fragments, but'
          f' got {rule}.'
      )
    for fragment in rule.sources:
      if not fragment.origins:
        raise ValueError(
            'Each source fragment must have at least one origin, but got'
            f' {rule}.'
        )

    if not rule.target.origins:
      raise ValueError(
          f'Target fragment must have at least one origin, but got {rule}.'
      )


def mesh_names(
    pytree: PyTree[
        jax.Array
        | jax.ShapeDtypeStruct
        | jax.sharding.NamedSharding
        | jax.sharding.Mesh
    ],
    topology: Topology,
) -> PyTree[str | None]:
  """Returns the mesh name for the Mesh of each element in `pytree`."""
  mesh_to_name = {mesh: name for name, mesh in topology.items()}

  def _get_name(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh):
    if isinstance(mesh, jax.sharding.AbstractMesh):
      # Abstract meshes have no names, and are only used for SPMD shardings but
      # not MPMD mesh assignments.
      return None
    if mesh in mesh_to_name:
      return mesh_to_name[mesh]
    raise ValueError(
        f'Mesh {mesh} with devices {mesh.devices} not found in topology'
        f' {topology}.'
    )

  def _get_mesh_name(x):
    if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)):
      return _get_mesh_name(x.sharding)
    if isinstance(x, jax.sharding.NamedSharding):
      return _get_name(x.mesh)
    if isinstance(x, (jax.sharding.Mesh, jax.sharding.AbstractMesh)):
      return _get_name(x)
    raise ValueError(f'Unexpected type: {type(x)} ')

  return jax.tree.map(_get_mesh_name, pytree)


@dataclasses.dataclass(frozen=True)
class FunctionIOMeshAssignment:
  """The mesh names to which all function inputs and outputs are assigned to.

  Attributes:
    input_meshes: Mesh assignment for the inputs, retrieved from the MLIR
      module. Has placeholders for removed unused inputs.
    output_meshes: Mesh assignment for the outputs, retrieved from the MLIR
      module.
  """

  input_meshes: PyTree[str]
  output_meshes: PyTree[str]


def _validate_partitioning_options(
    partitioning_options: Mapping[str, bool | str] | None,
):
  """Validates the partitioning options."""
  if partitioning_options is None:
    return

  if not part_options.MPMD_OPTIONS.issuperset(partitioning_options):
    raise ValueError(
        '`mpmd_options` contains unsupported options '
        f'{partitioning_options.keys() - part_options.MPMD_OPTIONS}'
    )

  for k, v in partitioning_options.items():
    if k in part_options.MPMD_BOOLEAN_OPTIONS:
      if not isinstance(v, bool):
        raise ValueError(f'Option {k} has value {v}, which is not a boolean.')
    elif k == part_options.MPMD_PIPELINE_SCHEDULE_OPTION:
      if v not in part_options.MPMD_PIPELINE_SCHEDULE_VALUES:
        raise ValueError(
            f'Option {k} has value {v}, which is not a valid pipeline schedule.'
        )
    else:
      # Raise exception to guard against adding new PartIR MPMD options without
      # extending this function to handle them.
      raise NotImplementedError(f'PartIR MPMD option {k} is not supported.')
