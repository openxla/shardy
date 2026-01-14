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

"""Core data structures and helper functions for MPMD pipeline scheduling.

The primary entry point for defining a schedule is the `PipelineSchedule`
object, which uses rule builders to determine the execution order and
merging of these fragments. Rule builders take lists of fragments and build
concrete scheduling/merging rules.

There are two main approaches to defining pipeline schedules:

1. Predicate-based approach (recommended for simple patterns):
   Use binary predicates with helper functions to automatically generate rules.
   `schedule_impl.py` contains implementations of common schedules using these
   predicates.

2. Direct construction (for complex custom schedules):
   Explicitly build execution order and merge rules for full control.

This is best shown through example: see `pipeline_test.py` for a concrete
PipelineSchedule definitions using both approaches.
"""

import collections
from collections.abc import Collection, Mapping, Sequence, Set
import dataclasses
import enum
from typing import Callable

FragmentMergeRules = Sequence['FragmentMergeRule']
FragmentScheduleRules = Sequence['FragmentScheduleRule']

# Function that constructs a target FragmentInfo from a sequence of source
# fragments that will be merged together into the target.
TargetInfoBuilder = Callable[[Sequence['FragmentInfo']], 'FragmentInfo']

# Function that builds schedule and/or merge rules from fragments and pipeline
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

  origins: tuple[FragmentOrigin, ...]
  stage_id: int | None = None
  call_counter: int | None = None
  split_type: SplitFragmentType | None = None
  mesh_name: str = ''


def validate_fragment_rule_origins(
    fragment_collection: Collection[FragmentInfo],
) -> None:
  """Validates that all fragments have at least one origin."""
  for fragment in fragment_collection:
    if not fragment.origins:
      raise ValueError(
          f'Each fragment must have at least one origin, but got {fragment} in'
          f' {fragment_collection}.'
      )


def validate_fragment_rule_meshes(
    fragment_collection: Collection[FragmentInfo],
) -> None:
  """Validates that all fragments are on the same mesh."""
  first_fragment = next(iter(fragment_collection))
  first_mesh = first_fragment.mesh_name
  if not all(
      fragment.mesh_name == first_mesh for fragment in fragment_collection
  ):
    raise ValueError(
        'Fragments being merged/scheduled must be on the same mesh, but got'
        f' {fragment_collection}.'
    )


@dataclasses.dataclass(frozen=True)
class FragmentMergeRule:
  """A rule for merging fragments of a computation.

  Attributes:
    sources: The source fragments to be merged. The order does not affect the
      final position of the merged fragment.
    target: The target fragment metadata that results from merging the sources.
  """

  sources: Set[FragmentInfo]
  target: FragmentInfo

  def __post_init__(self):
    # Validate the fragment merge rule.
    if len(self.sources) < 2:
      raise ValueError(
          'FragmentMergeRule must contain at least 2 source fragments, but got'
          f' {self}.'
      )
    validate_fragment_rule_origins(self.sources)
    validate_fragment_rule_meshes(self.sources)

    if not self.target.origins:
      raise ValueError(
          f'Target fragment must have at least one origin, but got {self}.'
      )


@dataclasses.dataclass(frozen=True)
class FragmentScheduleRule:
  """A rule for scheduling fragments in a specific execution order.

  Attributes:
    ordered_fragments: Fragments in the order they should execute. Must contain
      at least 2 fragments, and all fragments must be on the same mesh.
  """

  ordered_fragments: Sequence[FragmentInfo]

  def __post_init__(self):
    # Validate the fragment schedule rule.
    if len(self.ordered_fragments) < 2:
      raise ValueError(
          'FragmentScheduleRule must contain at least 2 fragments, but got'
          f' {self}.'
      )
    validate_fragment_rule_origins(self.ordered_fragments)
    validate_fragment_rule_meshes(self.ordered_fragments)


@dataclasses.dataclass(frozen=True)
class PipelineContext:
  """Context for pipeline scheduling and merging predicates."""

  num_meshes: int


@dataclasses.dataclass(frozen=True)
class PipelineSchedule:
  """A set of rules and options which define an MPMD pipeline.

  Attributes:
    schedule_merge_rule_builders: A sequence of functions that build schedule
      and/or merge rules for fragments.
    required_mpmd_options: A mapping of PartitioningEnvironment flags that are
      required for this schedule to function correctly. See
      `partitioning_options.py` for available options.  Relevant options
      include:

  `mpmd_split_bwd_fragments`: Set to True to split backward fragments into
    separate weight gradient and activation gradient fragments. This enables
    independent scheduling of weight and activation gradients.

  `mpmd_merge_inferred_after_scheduling`: Set to True to defer merging of
    inferred fragments until after scheduling. If False (default), inferred
    fragments are merged before scheduling, which may create unintended data
    dependencies that constrain your scheduling order.
  """

  schedule_merge_rule_builders: Sequence[ScheduleMergeRuleBuilder] | None = None
  required_mpmd_options: Mapping[str, bool | str] | None = None


def fragment_origins_contain(fragment: FragmentInfo, substring: str) -> bool:
  """Checks if any computation name in fragment origins contains the substring."""
  return any(
      substring in origin.computation_name for origin in fragment.origins
  )


def build_schedule_rules_from_predicate(
    fragment_infos: Sequence[FragmentInfo],
    context: PipelineContext,
    *,
    before_pred: RuleGeneratorPredicate,
) -> tuple[FragmentScheduleRules, FragmentMergeRules]:
  """Builds a list of scheduling rules using a binary predicate function."""
  res = []
  for i, a in enumerate(fragment_infos):
    for j, b in enumerate(fragment_infos):
      if i == j:
        continue
      if a.mesh_name != b.mesh_name:
        continue

      if before_pred(a, b, context):
        res.append(FragmentScheduleRule(ordered_fragments=[a, b]))
  return res, []


def union_fragment_origins(
    source_fragments: Sequence[FragmentInfo],
) -> tuple[FragmentOrigin, ...]:
  """Union all origins from a sequence of fragment infos."""
  merged_origins = []
  seen_origins = set()
  for fragment in source_fragments:
    for origin in fragment.origins:
      origin_key = (origin.computation_name, origin.transpose_count)
      if origin_key not in seen_origins:
        merged_origins.append(origin)
        seen_origins.add(origin_key)
  return tuple(merged_origins)


def _minimal_create_target_info(
    source_fragments: Sequence[FragmentInfo],
) -> FragmentInfo:
  """Creates a target FragmentInfo based on a sequence of source FragmentInfos.

  FragmentMergeRule takes in a FragmentInfo which describes the final fragment
  metadata after all sources have been merged. This functions creates a target
  info with the minimal amount of information needed to create this target
  FragmentInfo.

  Args:
    source_fragments: List of source fragment infos to create target info from.

  Returns:
    FragmentInfo object representing the target fragment info.

  Raises:
    ValueError: If `source_fragments` is empty or fragments have inconsistent
      `mesh_name` values.
  """
  if not source_fragments:
    raise ValueError(
        'Cannot create target info from empty source fragments sequence'
    )

  mesh_name = source_fragments[0].mesh_name
  for fragment in source_fragments:
    if fragment.mesh_name != mesh_name:
      raise ValueError(
          f'Inconsistent mesh_name values: {mesh_name} vs {fragment.mesh_name}'
      )

  return FragmentInfo(
      origins=union_fragment_origins(source_fragments),
      stage_id=None,
      call_counter=None,
      split_type=None,
      mesh_name=mesh_name,
  )


def build_merge_rules_from_predicate(
    fragment_infos: Sequence[FragmentInfo],
    context: PipelineContext,
    target_info_builder: TargetInfoBuilder = _minimal_create_target_info,
    *,
    pred: RuleGeneratorPredicate,
) -> tuple[FragmentScheduleRules, FragmentMergeRules]:
  """Creates a list of fragment merge rules based on a binary predicate.

  Args:
    fragment_infos: List of fragments to create merge rules for.
    context: PipelineContext object containing additional context for the
      scheduling and merging process.
    target_info_builder: Function that creates a target fragment info based on
      on a list of source fragment infos. Defaults to create_target_info.
    pred: Binary predicate function that determines if fragments should be
      merged.

  Returns:
    Tuple of (schedule_rules, merge_rules) where schedule_rules is empty.
  """
  merge_rules = []
  for i, fragment_a in enumerate(fragment_infos):
    # Order of fragments should not matter for merge rules, so we can skip
    # checking pairs of fragments that have already been checked.
    for fragment_b in fragment_infos[i + 1 :]:
      if fragment_a.mesh_name != fragment_b.mesh_name:
        continue

      if pred(fragment_a, fragment_b, context):
        merge_rules.append(
            FragmentMergeRule(
                sources={fragment_a, fragment_b},
                target=target_info_builder([fragment_a, fragment_b]),
            )
        )
  return [], merge_rules


def build_rules_from_pipeline(
    fragment_infos: Sequence[FragmentInfo],
    pipeline: PipelineSchedule,
    context: PipelineContext,
) -> tuple[FragmentScheduleRules, FragmentMergeRules]:
  """Builds scheduling and merging rules from a PipelineSchedule.

  Args:
    fragment_infos: List of fragments to build rules for.
    pipeline: PipelineSchedule containing rule generators and options.
    context: PipelineContext with pipeline configuration.

  Returns:
    Tuple of (schedule_rules, merge_rules) built from rule builders.
  """
  # Create a list of fragments for each mesh once
  mesh_fragments = collections.defaultdict(list)
  for fragment in fragment_infos:
    mesh_fragments[fragment.mesh_name].append(fragment)

  all_schedule_rules = []
  all_merge_rules = []

  if pipeline.schedule_merge_rule_builders:
    for builder in pipeline.schedule_merge_rule_builders:
      # Run each builder on fragments from each mesh separately
      for _, single_mesh_fragments in mesh_fragments.items():
        schedule_rules, merge_rules = builder(single_mesh_fragments, context)
        all_schedule_rules.extend(schedule_rules)
        all_merge_rules.extend(merge_rules)

  return all_schedule_rules, all_merge_rules


def maybe_unique_transpose_count(
    fragment: FragmentInfo,
) -> int | None:
  """Returns transpose count if all fragment origins have the same value."""
  if not fragment.origins:
    return None

  # Check if all origins have the same transpose count.
  transpose_counts = {origin.transpose_count for origin in fragment.origins}
  if len(transpose_counts) == 1:
    return transpose_counts.pop()

  return None


def get_scheduling_unit_info(fragment: FragmentInfo) -> tuple[int, int] | None:
  """Returns (call_counter, transpose_count) if fragment is a valid scheduling unit.

  A fragment is a valid scheduling unit if it meets all of the following
  conditions:
  - It is a user fragment (has origins)
  - It has a call_counter
  - It has a single transpose_count which is 0 or 1

  Args:
    fragment: Fragment to check scheduling unit for.

  Returns:
    A tuple of (call_counter, transpose_count) if valid, None otherwise.
  """
  if not fragment.origins:
    return None

  if fragment.call_counter is None:
    return None

  transpose_count = maybe_unique_transpose_count(fragment)
  if transpose_count is not None and (
      transpose_count == 0 or transpose_count == 1
  ):
    return (fragment.call_counter, transpose_count)

  return None


def get_staged_scheduling_info(
    f1: FragmentInfo, f2: FragmentInfo, error_context: str
) -> tuple[int, int, int, int] | None:
  """Validates two fragments for scheduling and returns their info.

  Args:
    f1: First fragment to validate.
    f2: Second fragment to validate.
    error_context: Context for the error message if stage_id validation fails,
      e.g., "1F1B scheduling".

  Returns:
    Tuple of (call_counter_f1, transpose_count_f1, call_counter_f2,
    transpose_count_f2) if both fragments are valid scheduling units with
    stages, None otherwise.

  Raises:
    ValueError: If `stage_id` is not set on either of the fragments.
  """
  f1_info = get_scheduling_unit_info(f1)
  f2_info = get_scheduling_unit_info(f2)
  if f1_info is None or f2_info is None:
    return None

  if f1.stage_id is None or f2.stage_id is None:
    raise ValueError(f'All fragments must have a stage id for {error_context}.')

  call_counter_f1, transpose_count_f1 = f1_info
  call_counter_f2, transpose_count_f2 = f2_info
  return (
      call_counter_f1,
      transpose_count_f1,
      call_counter_f2,
      transpose_count_f2,
  )
