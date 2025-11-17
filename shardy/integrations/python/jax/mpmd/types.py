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
import enum
from typing import Callable

from absl import logging
import jax
from jax.jaxlib import _sdy_mpmd as jaxlib_mpmd
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

## Type aliases for custom scheduling and merging rules

# Sequence of fragment merge rules defining how fragments can be combined.
FragmentMergeRules = Sequence['FragmentMergeRule']
# Sequence of fragment schedule rules defining execution order of fragments.
FragmentScheduleRules = Sequence['FragmentScheduleRule']

# Function that builds schedule rules from fragments and pipeline context.
ScheduleRuleBuilder = Callable[
    [Sequence['FragmentInfo'], 'PipelineContext'], FragmentScheduleRules
]

# Function that constructs a target FragmentInfo from a sequence of source
# fragments that will be merged together into the target.
TargetInfoBuilder = Callable[[Sequence['FragmentInfo']], 'FragmentInfo']
# Function that builds merge rules from fragments and pipeline context.
MergeRuleBuilder = Callable[
    [Sequence['FragmentInfo'], 'PipelineContext'], FragmentMergeRules
]
# Function that builds both schedule and merge rules from fragments and pipeline
# context.
ScheduleMergeRuleBuilder = Callable[
    [Sequence['FragmentInfo'], 'PipelineContext'],
    tuple[FragmentScheduleRules, FragmentMergeRules],
]

# Binary predicate determining if two fragments should be merged or scheduled
# together.
RuleGeneratorPredicate = Callable[
    ['FragmentInfo', 'FragmentInfo', 'PipelineContext'], bool
]


@dataclasses.dataclass(frozen=True)
class FragmentOrigin:
  """The origin of a fragment."""

  computation_name: str
  transpose_count: int = 0


@enum.unique
class SplitFragmentType(enum.Enum):
  """Fragment split behavior for transferred data.

  These values indicate how fragment portions handle transferred data from
  the original fragment if the fragment is split during compilation:
  - KEEP_TRANSFERRED: Fragment portion retains transferred data
  - DROP_TRANSFERRED: Fragment portion drops transferred data
  """

  KEEP_TRANSFERRED = enum.auto()
  DROP_TRANSFERRED = enum.auto()


@dataclasses.dataclass(frozen=True)
class FragmentInfo:
  """A fragment of a computation."""

  origins: Sequence[FragmentOrigin]
  stage_id: int | None = None
  call_counter: int | None = None
  split_type: SplitFragmentType | None = None
  mesh_name: str = ''


@dataclasses.dataclass(frozen=True)
class FragmentMergeRule:
  """A rule for merging fragments of a computation.

  Attributes:
    sources: The source fragments to be merged. The order does not affect the
      final position of the merged fragment, which is determined by topological
      sorting based on data dependencies.
    target: The target fragment metadata that results from merging the sources.
  """

  sources: Sequence[FragmentInfo]
  target: FragmentInfo


@dataclasses.dataclass(frozen=True)
class FragmentScheduleRule:
  """A rule for scheduling fragments in a specific execution order.

  Attributes:
    ordered_fragments: Fragments in the relative order they should execute. Must
      contain at least 2 fragments, and all fragments must be on the same mesh.
  """

  ordered_fragments: Sequence[FragmentInfo]


@dataclasses.dataclass(frozen=True)
class PipelineContext:
  """Context for pipeline scheduling and merging predicates."""

  num_meshes: int


# LINT.IfChange
CPU_MESH_SUFFIX = '/cpu'
# LINT.ThenChange(
#   https://github.com/openxla/shardy/blob/main/shardy/dialect/mpmd/ir/utils.h
# )


def mesh_is_on_cpu(mesh_name: str) -> bool:
  """Returns whether the mesh name is for a cpu mesh."""
  return mesh_name.endswith(CPU_MESH_SUFFIX)


def get_schedulable_meshes(topology: Topology) -> list[str]:
  """Returns the schedulable meshes in the topology."""
  return [name for name in topology if not mesh_is_on_cpu(name)]


# Type conversion functions for FragmentInfo and related types
def _to_jaxlib_mpmd_split_type(
    split_type: SplitFragmentType | None,
) -> jaxlib_mpmd.SplitFragmentType | None:
  """Convert native Python enum to pybinded enum."""
  if split_type is None:
    return None
  if split_type == SplitFragmentType.KEEP_TRANSFERRED:
    return jaxlib_mpmd.SplitFragmentType.KEEP_TRANSFERRED
  elif split_type == SplitFragmentType.DROP_TRANSFERRED:
    return jaxlib_mpmd.SplitFragmentType.DROP_TRANSFERRED
  else:
    raise ValueError(f'Unknown SplitFragmentType: {split_type}')


def _from_jaxlib_mpmd_split_type(
    split_type: jaxlib_mpmd.SplitFragmentType | None,
) -> SplitFragmentType | None:
  """Convert pybinded enum to native Python enum."""
  if split_type is None:
    return None
  if split_type == jaxlib_mpmd.SplitFragmentType.KEEP_TRANSFERRED:
    return SplitFragmentType.KEEP_TRANSFERRED
  elif split_type == jaxlib_mpmd.SplitFragmentType.DROP_TRANSFERRED:
    return SplitFragmentType.DROP_TRANSFERRED
  else:
    raise ValueError(f'Unknown jaxlib_mpmd.SplitFragmentType: {split_type}')


def convert_fragment_info_to_pybind(
    fragment: FragmentInfo,
) -> jaxlib_mpmd.FragmentInfo:
  """Converts FragmentInfo to jaxlib_mpmd.FragmentInfo."""
  return jaxlib_mpmd.FragmentInfo(
      origins=[
          jaxlib_mpmd.FragmentOrigin(
              origin.computation_name, origin.transpose_count
          )
          for origin in fragment.origins
      ],
      stage_id=fragment.stage_id,
      call_counter=fragment.call_counter,
      split_type=_to_jaxlib_mpmd_split_type(fragment.split_type),
      mesh_name=fragment.mesh_name,
  )


def convert_pybind_fragment_info_to_types(
    fragment: jaxlib_mpmd.FragmentInfo,
) -> FragmentInfo:
  """Converts jaxlib_mpmd.FragmentInfo to FragmentInfo."""
  return FragmentInfo(
      origins=[
          FragmentOrigin(origin.computation_name, origin.transpose_count)
          for origin in fragment.origins
      ],
      stage_id=fragment.stage_id,
      call_counter=fragment.call_counter,
      split_type=_from_jaxlib_mpmd_split_type(fragment.split_type),
      mesh_name=fragment.mesh_name,
  )


def convert_fragment_merge_rules_to_pybind(
    fragment_merge_rules: FragmentMergeRules,
) -> list[jaxlib_mpmd.FragmentMergeRule]:
  """Converts fragment merge rules to jaxlib_mpmd.FragmentMergeRules."""
  pybind_fragment_merge_rules = []
  for rule in fragment_merge_rules:
    fragments = [
        convert_fragment_info_to_pybind(fragment) for fragment in rule.sources
    ]
    pybind_fragment_merge_rules.append(
        jaxlib_mpmd.FragmentMergeRule(
            sources=fragments,
            target=convert_fragment_info_to_pybind(rule.target),
        )
    )
  return pybind_fragment_merge_rules


def convert_fragment_schedule_rules_to_pybind(
    fragment_schedule_rules: FragmentScheduleRules,
) -> list[jaxlib_mpmd.FragmentScheduleRule]:
  """Converts fragment schedule rules to jaxlib_mpmd.FragmentScheduleRules."""
  pybind_fragment_schedule_rules = []
  for rule in fragment_schedule_rules:
    fragments = [
        convert_fragment_info_to_pybind(fragment)
        for fragment in rule.ordered_fragments
    ]
    pybind_fragment_schedule_rules.append(
        jaxlib_mpmd.FragmentScheduleRule(ordered_fragments=fragments)
    )
  return pybind_fragment_schedule_rules


@dataclasses.dataclass(frozen=True)
class PipelineSchedule:
  """A set of rules and options which define an MPMD pipeline.

  Attributes:
    merge_rule_builders: A sequence of functions that build merge rules for
      fragments.
    schedule_rule_builders: A sequence of functions that build schedule rules
      for fragments.
    schedule_merge_rule_builders: A sequence of functions that build both
      schedule and merge rules for fragments.
    required_mpmd_options: A mapping of PartitioningEnvironment flags that are
      required for this schedule to function correctly. See
      partitioning_options.py for available options.
  """

  merge_rule_builders: Sequence[MergeRuleBuilder] | None = None
  schedule_rule_builders: Sequence[ScheduleRuleBuilder] | None = None
  schedule_merge_rule_builders: Sequence[ScheduleMergeRuleBuilder] | None = None
  required_mpmd_options: Mapping[str, bool | str] | None = None


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
    pipeline_schedule: A PipelineSchedule object used to generate merge and/or
      schedule rules for partitioning, as well as set any required MPMD options
      for the pipeline.
  """

  topology: Topology
  name_to_mesh_assignment: NameToMeshAssignment
  name_to_stage_assignment: NameToStageAssignment
  input_mesh_assignment: PyTree[str | None]
  output_mesh_assignment: PyTree[str | None]
  partitioning_options: PartitioningOptions | None
  read_input_output_mesh_from_shardings: bool
  pipeline_schedule: PipelineSchedule | None

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
    pipeline_schedule: PipelineSchedule | None = None,
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
    pipeline_schedule: See `MpmdConfig`.

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

  return MpmdConfig(
      topology,
      name_to_mesh_assignment,
      name_to_stage_assignment if name_to_stage_assignment else {},
      input_mesh_assignment,
      output_mesh_assignment,
      partitioning_options,
      read_input_output_mesh_from_shardings,
      pipeline_schedule,
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


def validate_fragment_rule_origins(
    fragment_sequence: Sequence[FragmentInfo],
) -> None:
  for fragment in fragment_sequence:
    if not fragment.origins:
      raise ValueError(
          f'Each fragment must have at least one origin, but got {fragment} in'
          f' {fragment_sequence}.'
      )


def validate_fragment_rule_meshes(
    fragment_sequence: Sequence[FragmentInfo],
) -> None:
  first_mesh = fragment_sequence[0].mesh_name
  if not all(
      fragment.mesh_name == first_mesh for fragment in fragment_sequence
  ):
    raise ValueError(
        'Fragments being merged/scheduled must be on the same mesh, but got'
        f' {fragment_sequence}.'
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
    validate_fragment_rule_origins(rule.sources)
    validate_fragment_rule_meshes(rule.sources)

    if not rule.target.origins:
      raise ValueError(
          f'Target fragment must have at least one origin, but got {rule}.'
      )


def validate_fragment_schedule_rules(
    fragment_schedule_rules: FragmentScheduleRules,
) -> None:
  """Validates the fragment schedule rules."""
  for rule in fragment_schedule_rules:
    if len(rule.ordered_fragments) < 2:
      raise ValueError(
          'Fragment schedule rule must contain at least two fragments, but'
          f' got {rule}.'
      )

    validate_fragment_rule_origins(rule.ordered_fragments)
    validate_fragment_rule_meshes(rule.ordered_fragments)


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


def override_partitioning_options(
    mpmd_options: Mapping[str, bool | str] | None,
    base_options_to_override: PartitioningOptions | None = None,
) -> PartitioningOptions | None:
  """Overrides the base partitioning options with the given MPMD options."""
  if mpmd_options is None:
    return base_options_to_override

  _validate_partitioning_options(mpmd_options)

  options = {}
  if base_options_to_override is not None:
    options.update(base_options_to_override)

  options.update(mpmd_options)
  return options


def check_partitioning_option_conflicts(
    pipeline_required_options: Mapping[str, bool | str],
    user_options_dict: Mapping[str, bool | str],
) -> list[str]:
  """Checks for conflicts between pipeline requirements and user options.

  Args:
    pipeline_required_options: Options required by the pipeline schedule.
    user_options_dict: Options explicitly set by the user.

  Returns:
    List of conflict error messages. Empty if no conflicts.
    Logs warnings for options that will be set automatically.
  """
  conflicts = []
  for k, required_value in pipeline_required_options.items():
    if k in user_options_dict:
      user_value = user_options_dict[k]
      if user_value != required_value:
        conflicts.append(
            f"  - '{k}': pipeline schedule requires {required_value}, "
            f'but user specified {user_value}'
        )
    else:
      # Option not set by user, will be set automatically
      logging.warning(
          'Setting partitioning option %r to %s (required by pipeline'
          ' schedule)',
          k,
          required_value,
      )
  return conflicts


def validate_and_merge_partitioning_options(
    pipeline_required_options: Mapping[str, bool | str] | None,
    user_provided_options: PartitioningOptions | None,
) -> PartitioningOptions | None:
  """Validates and merges user options with pipeline requirements.

  Ensures that user-provided partitioning options don't conflict with options
  required by the pipeline schedule. If conflicts are found, it raises an error.
  If the user hasn't set all required options, it logs warnings for the options
  that still need to be set and will set them automatically.

  Args:
    pipeline_required_options: Options required by the pipeline schedule.
    user_provided_options: Options provided by the user via MpmdConfig. This is
      expected to be a dict or None.

  Returns:
    Merged partitioning options with pipeline requirements taking precedence
    where the user hasn't specified them.

  Raises:
    ValueError: If user options conflict with pipeline required options.
  """
  if pipeline_required_options is None:
    return user_provided_options

  user_options_dict = user_provided_options if user_provided_options else {}

  # Check for conflicts and log warnings
  conflicts = check_partitioning_option_conflicts(
      pipeline_required_options, user_options_dict
  )

  if conflicts:
    conflict_msg = '\n'.join(conflicts)
    raise ValueError(
        f'Conflicting partitioning options detected:\n{conflict_msg}\n'
        'Please remove these options from your MpmdConfig or ensure they '
        'match the pipeline schedule requirements.'
    )

  # Merge the options
  return override_partitioning_options(
      mpmd_options=pipeline_required_options,
      base_options_to_override=user_provided_options,
  )


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
