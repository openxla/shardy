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

"""Pipeline scheduling and merging for MPMD.

This module provides rule builders that determine fragment execution order and
merging in pipeline parallelism, as well as associated helper functions. Rule
builders take a list of fragments and build concrete scheduling/merging rules.
These rule builders are used to define PipelineSchedule objects which determine
fragment execution order and merging.

There are two main approaches to defining pipeline schedules:

1. Predicate-based approach (recommended for simple patterns):
   Use binary predicates with helper functions to automatically generate rules.
   See pipeline_registry.py for more predicate-based examples.

   Example:
       def my_schedule_predicate(f1, f2, context):
           # Return True if f1 should execute before f2
           return f1.call_counter < f2.call_counter

       def my_merge_predicate(f1, f2, context):
           # Return True if f1 and f2 should be merged
           return f1.stage_id == f2.stage_id and f1.call_counter ==
           f2.call_counter

       schedule = PipelineSchedule(
           schedule_rule_builders=[
               functools.partial(build_schedule_rules_from_predicate,
                               before_pred=my_schedule_predicate)
           ],
           merge_rule_builders=[
               functools.partial(build_merge_rules_from_predicate,
                               pred=my_merge_predicate)
           ],
           required_mpmd_options={...},
       )

2. Direct construction (for complex custom schedules):
   Explicitly build execution order and merge rules for full control.
   The example below implements the same logic as the predicate-based example
   above.

   Example:
       def custom_schedule_builder(fragment_infos, context):
           # Separate forward and backward fragments, sorted by call_counter
           forward = sorted(
               [f for f in fragment_infos if f.transpose_count == 0],
               key=lambda f: f.call_counter
           )
           backward = sorted(
               [f for f in fragment_infos if f.transpose_count == 1],
               key=lambda f: f.call_counter
           )

           # Interleave: fwd, bwd, fwd, bwd, ...
           execution_order = []
           for fwd, bwd in zip(forward, backward):
               execution_order.extend([fwd, bwd])

           # Merge fragments with same stage_id
           merge_rules = [
               FragmentMergeRule(
                   sources=(fwd, bwd),
                   target=minimal_create_target_info((fwd, bwd))
               )
               for fwd, bwd in zip(forward, backward)
               if fwd.stage_id == bwd.stage_id
           ]

           return [FragmentScheduleRule(ordered_fragments=execution_order)],
           merge_rules

       schedule = PipelineSchedule(
           schedule_merge_rule_builders=[custom_schedule_builder],
           required_mpmd_options={...},
       )

Useful MPMD Options:

The `required_mpmd_options` field allows you to force the MPMD compiler to use
the specified options, useful if your pipeline schedule would not work without
them. Relevant options include:

- mpmd_split_bwd_fragments: Set to True to split backward fragments into
  separate weight gradient and activation gradient fragments. This enables
  independent scheduling of weight and activation gradients.

- mpmd_merge_inferred_after_scheduling: Set to True to defer merging of inferred
  fragments until after scheduling. If False (default), inferred fragments are
  merged before scheduling, which may create unintended data dependencies that
  constrain your scheduling order.
"""

import collections
from collections.abc import Sequence
from typing import Callable

from shardy.integrations.python.jax.mpmd import types as shardy_mpmd_types

# Type aliases for clarity
FragmentOrigin = shardy_mpmd_types.FragmentOrigin
FragmentInfo = shardy_mpmd_types.FragmentInfo
FragmentMergeRule = shardy_mpmd_types.FragmentMergeRule
FragmentScheduleRule = shardy_mpmd_types.FragmentScheduleRule
FragmentMergeRules = shardy_mpmd_types.FragmentMergeRules
FragmentScheduleRules = shardy_mpmd_types.FragmentScheduleRules
PipelineSchedule = shardy_mpmd_types.PipelineSchedule
PipelineContext = shardy_mpmd_types.PipelineContext
RuleGeneratorPredicate = shardy_mpmd_types.RuleGeneratorPredicate
SplitFragmentType = shardy_mpmd_types.SplitFragmentType
ScheduleRuleBuilder = shardy_mpmd_types.ScheduleRuleBuilder
MergeRuleBuilder = shardy_mpmd_types.MergeRuleBuilder
TargetInfoBuilder = shardy_mpmd_types.TargetInfoBuilder


def fragment_origins_contain(fragment: FragmentInfo, substring: str) -> bool:
  """Checks if any computation name in fragment origins contains the substring.

  Args:
    fragment: FragmentInfo to check.
    substring: String to search for in computation names.

  Returns:
    True if any origin's computation_name contains the substring.
  """
  return any(
      substring in origin.computation_name for origin in fragment.origins
  )


def build_schedule_rules_from_predicate(
    fragment_infos: Sequence[FragmentInfo],
    context: PipelineContext,
    *,
    before_pred: RuleGeneratorPredicate,
) -> FragmentScheduleRules:
  """Builds a list of scheduling rules using a binary predicate function.

  Args:
    fragment_infos: List of fragments to create scheduling rules for.
    context: PipelineContext object containing additional context for the
      scheduling and merging process.
    before_pred: Binary predicate function that determines if fragment A should
      be scheduled before fragment B.

  Returns:
    List of FragmentScheduleRule objects.
  """
  res = []
  for i, a in enumerate(fragment_infos):
    for j, b in enumerate(fragment_infos):
      if i == j:
        continue
      if a.mesh_name != b.mesh_name:
        continue

      if before_pred(a, b, context):
        res.append(FragmentScheduleRule(ordered_fragments=[a, b]))
  return res


def union_fragment_origins(
    source_fragments: Sequence[FragmentInfo],
) -> list[FragmentOrigin]:
  """Union all origins from a sequence of fragment infos."""
  merged_origins = []
  seen_origins = set()
  for fragment in source_fragments:
    for origin in fragment.origins:
      origin_key = (origin.computation_name, origin.transpose_count)
      if origin_key not in seen_origins:
        merged_origins.append(origin)
        seen_origins.add(origin_key)
  return merged_origins


def minimal_create_target_info(
    source_fragments: Sequence[FragmentInfo],
) -> FragmentInfo:
  """Creates a target fragment info based on a sequence of source fragment infos.

  This version only creates a target info with the minimal amount of
  information needed to create a valid target fragment info, which generally
  allows for greater flexibility later on in the compiler.

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
        "Cannot create target info from empty source fragments sequence"
    )

  mesh_name = source_fragments[0].mesh_name
  for fragment in source_fragments:
    if fragment.mesh_name != mesh_name:
      raise ValueError(
          f"Inconsistent mesh_name values: {mesh_name} vs {fragment.mesh_name}"
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
    target_info_builder: TargetInfoBuilder = minimal_create_target_info,
    *,
    pred: RuleGeneratorPredicate,
) -> list[FragmentMergeRule]:
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
    List of FragmentMergeRule objects.
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
                sources=[fragment_a, fragment_b],
                target=target_info_builder([fragment_a, fragment_b]),
            )
        )
  return merge_rules


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
  if pipeline.schedule_rule_builders:
    for builder in pipeline.schedule_rule_builders:
      # Run each builder on fragments from each mesh separately
      for _, single_mesh_fragments in mesh_fragments.items():
        all_schedule_rules.extend(builder(single_mesh_fragments, context))

  all_merge_rules = []
  if pipeline.merge_rule_builders:
    for builder in pipeline.merge_rule_builders:
      # Run each builder on fragments from each mesh separately
      for _, single_mesh_fragments in mesh_fragments.items():
        all_merge_rules.extend(builder(single_mesh_fragments, context))

  if pipeline.schedule_merge_rule_builders:
    for builder in pipeline.schedule_merge_rule_builders:
      for _, single_mesh_fragments in mesh_fragments.items():
        schedule_rules, merge_rules = builder(single_mesh_fragments, context)
        all_schedule_rules.extend(schedule_rules)
        all_merge_rules.extend(merge_rules)

  return all_schedule_rules, all_merge_rules


def try_to_find_single_transpose_count(
    fragment: FragmentInfo,
) -> int | None:
  """Returns transpose count if all fragment origins have the same value.

  Args:
    fragment: Fragment to check transpose count for.

  Returns:
    Transpose count if consistent across all origins, None otherwise.
  """
  if not fragment.origins:
    return None

  # Check if all origins have the same transpose count.
  transpose_counts = {origin.transpose_count for origin in fragment.origins}
  if len(transpose_counts) == 1:
    return transpose_counts.pop()

  return None


def get_scheduling_unit_info(fragment: FragmentInfo) -> tuple[int, int] | None:
  """Returns (call_counter, transpose_count) if fragment is a valid scheduling unit.

  A fragment is a scheduling unit if it meets all of the following conditions:
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

  transpose_count = try_to_find_single_transpose_count(fragment)
  if transpose_count is not None and (
      transpose_count == 0 or transpose_count == 1
  ):
    return (fragment.call_counter, transpose_count)

  return None


def _validate_staged_scheduling_info(
    f1: FragmentInfo, f2: FragmentInfo, error_msg: str
) -> tuple[int, int, int, int] | None:
  """Validates two fragments for scheduling and returns their info.

  Args:
    f1: First fragment to validate
    f2: Second fragment to validate
    error_msg: Error message for stage_id validation

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
    raise ValueError(error_msg)

  call_counter_f1, transpose_count_f1 = f1_info
  call_counter_f2, transpose_count_f2 = f2_info
  return (
      call_counter_f1,
      transpose_count_f1,
      call_counter_f2,
      transpose_count_f2,
  )


def gpipe_schedule_predicate(
    f1: FragmentInfo,
    f2: FragmentInfo,
    _: PipelineContext,
) -> bool:
  """Returns true if `f1` must happen before `f2` in a GPipe schedule."""
  transpose_count_f1 = try_to_find_single_transpose_count(f1)
  transpose_count_f2 = try_to_find_single_transpose_count(f2)
  if (
      transpose_count_f1 is None
      or transpose_count_f2 is None
      or f1.call_counter is None
      or f2.call_counter is None
  ):
    return False

  return (transpose_count_f1, f1.call_counter) < (
      transpose_count_f2,
      f2.call_counter,
  )


def one_fwd_one_bwd_schedule_predicate(
    f1: FragmentInfo,
    f2: FragmentInfo,
    context: PipelineContext,
) -> bool:
  """Returns true if f1 must happen before f2 in a 1F1B schedule."""
  result = _validate_staged_scheduling_info(
      f1, f2, "All fragments must have a stage id for 1F1B scheduling."
  )
  if result is None:
    return False
  call_counter_f1, transpose_count_f1, call_counter_f2, transpose_count_f2 = (
      result
  )

  # The following two conditions guarantee the forward and backward fragments
  # are interleaved in the steady state of the pipeline.

  # Example: in mesh/stage 0 of pipeline of depth 4, the backward computation
  # of microbatch 0 must be scheduled before the forward computation of
  # microbatch 4: 0 == 4 - 4 + 0.
  if transpose_count_f1 == 1 and transpose_count_f2 == 0:
    return call_counter_f1 == call_counter_f2 - context.num_meshes + f1.stage_id

  # Example: in mesh/stage 0 of pipeline of depth 4, the forward computation of
  # microbatch 5 must be scheduled before the backward computation of
  # microbatch 2: 5 == 2 + 4 - (0 + 1).
  if transpose_count_f1 == 0 and transpose_count_f2 == 1:
    return call_counter_f1 == call_counter_f2 + context.num_meshes - (
        f1.stage_id + 1
    )

  # If the fragments have the same transpose count, guarantee that the
  # call_counter ordering is preserved.
  if transpose_count_f1 == transpose_count_f2:
    return call_counter_f1 < call_counter_f2

  return False


def gpipe_with_1f1b_on_last_mesh_schedule_predicate(
    f1: FragmentInfo,
    f2: FragmentInfo,
    context: PipelineContext,
) -> bool:
  """Returns true if f1 must happen before f2 in a GPipe schedule with 1F1B on the last mesh."""
  result = _validate_staged_scheduling_info(
      f1,
      f2,
      "All fragments must have a stage id for GPipe with 1F1B on the last mesh"
      " scheduling.",
  )
  if result is None:
    return False
  # Validation successful - delegate to other functions
  _ = result

  if f1.stage_id == context.num_meshes - 1:
    return one_fwd_one_bwd_schedule_predicate(f1, f2, context)
  return gpipe_schedule_predicate(f1, f2, context)


def circular_schedule_predicate_base(
    f1: FragmentInfo,
    f2: FragmentInfo,
    context: PipelineContext,
    reverse_backward: bool,
) -> bool:
  """Returns true if f1 must happen before f2 in circular schedule."""
  # Check that both fragments are scheduling units
  result = _validate_staged_scheduling_info(
      f1, f2, "Cannot schedule for circular pipelining without stages."
  )
  if result is None:
    return False
  call_counter_f1, transpose_count_f1, call_counter_f2, transpose_count_f2 = (
      result
  )

  if transpose_count_f1 != transpose_count_f2:
    # Forward fragments always happen before backward fragments
    return transpose_count_f1 < transpose_count_f2

  # Both forward or both backward - use phase-based ordering
  phase_f1 = call_counter_f1 // context.num_meshes
  phase_f2 = call_counter_f2 // context.num_meshes

  f1_list = [phase_f1, f1.stage_id, call_counter_f1]
  f2_list = [phase_f2, f2.stage_id, call_counter_f2]

  # Forward fragments - ascending order
  if transpose_count_f1 == 0:
    return f1_list < f2_list

  # Backward fragments
  if reverse_backward:
    # Descending order
    return f1_list > f2_list

  # Backward fragments with stage in descending order
  f1_list[1], f2_list[1] = f2_list[1], f1_list[1]  # Swap stage IDs
  return f1_list < f2_list


def zero_bubble_h1_schedule_predicate(
    f1: FragmentInfo,
    f2: FragmentInfo,
    context: PipelineContext,
) -> bool:
  """Returns true if f1 must happen before f2 in a ZeroBubbleH1 schedule."""
  result = _validate_staged_scheduling_info(
      f1, f2, "All fragments must have a stage id for ZeroBubbleH1 scheduling."
  )
  if result is None:
    return False
  call_counter_f1, transpose_count_f1, call_counter_f2, transpose_count_f2 = (
      result
  )

  is_wgrad_f1 = f1.split_type == SplitFragmentType.DROP_TRANSFERRED
  is_wgrad_f2 = f2.split_type == SplitFragmentType.DROP_TRANSFERRED

  # The following two conditions guarantee the forward and backward fragments
  # are interleaved in the steady state of the pipeline. They are just like
  # 1F1B but specialized to actual back-propagation fragments.

  # Clause 1: Ba(i) < F(i + num_meshes - stage_id)
  if transpose_count_f1 == 1 and not is_wgrad_f1 and transpose_count_f2 == 0:
    return call_counter_f1 == call_counter_f2 - context.num_meshes + f1.stage_id

  # Clause 2: F(i + num_meshes - stage_id - 1) < Ba(i)
  if transpose_count_f1 == 0 and transpose_count_f2 == 1 and not is_wgrad_f2:
    return call_counter_f1 == call_counter_f2 + context.num_meshes - (
        f1.stage_id + 1
    )

  # The rest of the conditions position the parameter gradient fragments.
  # Clause 3: Bw(i) < F(i + num_meshes)
  # e.g. Bw(0) < F(4) above.
  if (
      transpose_count_f1 == 1
      and (is_wgrad_f1 or f1.stage_id == 0)
      and transpose_count_f2 == 0
  ):
    return call_counter_f2 - call_counter_f1 == context.num_meshes

  # Clause 4: Ba(i + stage_id) < Bw(i)
  # e.g.
  # mesh0:  Ba(0) < Bw(0)
  # mesh1:  Ba(1) < Bw(0)
  # mesh2:  Ba(2) < Bw(0)
  # mesh3:  Ba(3) < Bw(0)
  if (
      transpose_count_f1 == 1
      and not is_wgrad_f1
      and transpose_count_f2 == 1
      and is_wgrad_f2
  ):
    return call_counter_f1 - call_counter_f2 == f1.stage_id

  # This is just needed for transitively completing Clauses 3 and 2, needed for
  # the final phase where there may be no remaining forward to anchor to.
  # Bw(i) < Ba(i + stage_id + 1)
  if (
      transpose_count_f1 == 1
      and is_wgrad_f1
      and transpose_count_f2 == 1
      and not is_wgrad_f2
  ):
    return call_counter_f2 - call_counter_f1 == f1.stage_id + 1

  return False


def zero_bubble_h2_schedule_predicate(
    f1: FragmentInfo,
    f2: FragmentInfo,
    context: PipelineContext,
    init_fwd_per_stage_fn: Callable[[int], int],
) -> bool:
  """Returns true if f1 must happen before f2 in a ZeroBubbleH2 schedule."""
  result = _validate_staged_scheduling_info(
      f1, f2, "All fragments must have a stage id for ZeroBubbleH2 scheduling."
  )
  if result is None:
    return False
  _, transpose_count_f1, _, transpose_count_f2 = result

  is_wgrad_f1 = f1.split_type == SplitFragmentType.DROP_TRANSFERRED
  is_wgrad_f2 = f2.split_type == SplitFragmentType.DROP_TRANSFERRED

  # How many fwd we are allowed to stream before entering steady state
  init_fwd = init_fwd_per_stage_fn(f1.stage_id)
  # The ZeroBubbleH2 pipeline is diagonally symmetric
  complement_init_fwd = init_fwd_per_stage_fn(
      context.num_meshes - f1.stage_id - 1
  )

  # Initial phase
  # Clause 1: F(i) <= B(_) for i < init_fwd
  if (
      transpose_count_f1 == 0
      and transpose_count_f2 == 1
      and f1.call_counter < init_fwd
  ):
    return True

  # Clause 2: Ba(i) < F(i + init_fwd)
  if (
      transpose_count_f1 == 1
      and not is_wgrad_f1
      and transpose_count_f2 == 0
      and f2.call_counter >= init_fwd
  ):
    return f2.call_counter - f1.call_counter == init_fwd

  # Clause 3: F(i + init_fwd - 1) < Ba(i)
  if (
      transpose_count_f1 == 0
      and f1.call_counter >= init_fwd
      and transpose_count_f2 == 1
      and not is_wgrad_f2
  ):
    return f1.call_counter - f2.call_counter == init_fwd - 1

  # Clause 4: Ba(i + complement_init_fwd - 1) < Bw(i)
  if (
      transpose_count_f1 == 1
      and not is_wgrad_f1
      and transpose_count_f2 == 1
      and is_wgrad_f2
  ):
    return f1.call_counter - f2.call_counter == complement_init_fwd - 1

  # Clause 5: Bw(i) < Ba(i + complement_init_fwd)
  if (
      transpose_count_f1 == 1
      and is_wgrad_f1
      and transpose_count_f2 == 1
      and not is_wgrad_f2
  ):
    return f2.call_counter - f1.call_counter == complement_init_fwd

  return False


def latency_hiding_zero_bubble_h2_schedule_predicate(
    f1: FragmentInfo,
    f2: FragmentInfo,
    context: PipelineContext,
    latency_stage_fraction: float,
) -> bool:
  """Returns true if f1 must happen before f2 in a latency-hiding ZeroBubbleH2 schedule.

  Args:
    f1: First fragment to compare.
    f2: Second fragment to compare.
    context: Pipeline context with configuration.
    latency_stage_fraction: Float between 0.0 and 1.0 specifying how much time
      activation forwarding transfers take compared to a stage compute time.
  """
  if not (0.0 <= latency_stage_fraction <= 1.0):
    raise ValueError("latency_stage_fraction must be between 0.0 and 1.0")

  def init_fwds_per_stage(stage_id: int) -> int:
    """Calculate number of forward microbatches before first backward."""
    # Number of transfers from beginning until first backward can execute
    num_init_transfers = 2.0 * (context.num_meshes - stage_id - 1)
    # Compute that has happened in initial first microbatch path
    num_init_compute = 2.0 * (context.num_meshes - stage_id) - 1.0
    return int(num_init_compute + num_init_transfers * latency_stage_fraction)

  return zero_bubble_h2_schedule_predicate(f1, f2, context, init_fwds_per_stage)


def parallel_pipelines_with_wraparound_schedule_predicate(
    f1: FragmentInfo,
    f2: FragmentInfo,
    _: PipelineContext,
) -> bool:
  """Returns true if f1 must happen before f2 in parallel pipelines with wraparound.

  Only supports forward fragments. The entrypoint for mesh{i} is call_counter
  {i}.
  For each mesh, the order is [F{n}, F{n-1}, ..., F{1}] rotated such that
  the leading fragment is F{mesh_index}.

  Args:
    f1: First fragment to compare.
    f2: Second fragment to compare.
  """
  result = _validate_staged_scheduling_info(
      f1,
      f2,
      "All fragments must have a stage id for parallel pipelines scheduling.",
  )
  if result is None:
    return False
  call_counter_f1, transpose_count_f1, call_counter_f2, transpose_count_f2 = (
      result
  )

  # Only forward fragments supported
  if transpose_count_f1 != 0 or transpose_count_f2 != 0:
    raise ValueError("Only forward fragments supported for parallel pipelines")

  if call_counter_f1 == call_counter_f2:
    raise ValueError(
        "Should not have duplicate call counter in parallel pipelines"
    )

  # The entrypoint to stage{i} is call_counter {i}, so this always happens
  # before
  if call_counter_f1 == f1.stage_id or call_counter_f2 == f1.stage_id:
    return call_counter_f1 == f1.stage_id

  # stage_id is the pivot. If both call_counters are on the same side of
  # the pivot, we flip the order. But if they are on different
  # sides, then we take the order as per normal.
  if (call_counter_f1 > f1.stage_id and call_counter_f2 > f1.stage_id) or (
      call_counter_f1 < f1.stage_id and call_counter_f2 < f1.stage_id
  ):
    return call_counter_f1 > call_counter_f2

  return call_counter_f1 < call_counter_f2
