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

"""Pipeline scheduling and merging predicates for MPMD.

Binary predicates that determine fragment relationships for pipeline
parallelism. Predicates generate concrete scheduling and merging rules during
compilation.

Usage:
    def my_schedule_predicate(f1, f2, context):
        # Return True if f1 should execute before f2
        return f1.call_counter < f2.call_counter

    def my_merge_predicate(f1, f2, context):
        # Return True if f1 and f2 should be merged
        return f1.stage_id == f2.stage_id and f1.call_counter == f2.call_counter

    schedule = PipelineSchedule(
        schedule_predicates=[my_schedule_predicate],
        merge_predicates=[my_merge_predicate],
        required_mpmd_options={...}
    )

Predicate signature: (FragmentInfo, FragmentInfo, PipelineContext) → bool
"""

from typing import Callable, List, Sequence, Tuple

from shardy.integrations.python.jax.mpmd import types as shardy_mpmd_types

# Type aliases for clarity
FragmentInfo = shardy_mpmd_types.FragmentInfo
FragmentMergeRule = shardy_mpmd_types.FragmentMergeRule
FragmentScheduleRule = shardy_mpmd_types.FragmentScheduleRule
FragmentMergeRules = shardy_mpmd_types.FragmentMergeRules
FragmentScheduleRules = shardy_mpmd_types.FragmentScheduleRules
PipelineSchedule = shardy_mpmd_types.PipelineSchedule
PipelineContext = shardy_mpmd_types.PipelineContext
RuleGeneratorPredicate = shardy_mpmd_types.RuleGeneratorPredicate
SplitFragmentType = shardy_mpmd_types.SplitFragmentType


def generate_schedule_rules(
    fragment_infos: Sequence[FragmentInfo],
    before_pred: RuleGeneratorPredicate,
    context: PipelineContext,
) -> FragmentScheduleRules:
  """Creates a list of scheduling rules based on a binary predicate.

  Args:
    fragment_infos: List of fragments to create scheduling rules for.
    before_pred: Binary predicate function that determines if one fragment
      should execute before another.
    context: PipelineContext object containing additional context for the
      scheduling and merging process.

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


def generate_target_info(source_fragments: List[FragmentInfo]) -> FragmentInfo:
  """Generates a target fragment info based on a list of source fragment infos."""
  # TODO: b/443920649 - Determine best way to automatically generate target
  # info from source fragment infos.
  raise NotImplementedError


def generate_merge_rules(
    fragment_infos: List[FragmentInfo],
    pred: RuleGeneratorPredicate,
    context: PipelineContext,
    target_info_generator: Callable[
        [List[FragmentInfo]], FragmentInfo
    ] = generate_target_info,
) -> List[FragmentMergeRule]:
  """Creates a list of fragment merge rules based on a binary predicate.

  Args:
    fragment_infos: List of fragments to create merge rules for.
    pred: Binary predicate function that determines if fragments should be
      merged.
    context: PipelineContext object containing additional context for the
      scheduling and merging process.
    target_info_generator: Function that generates a target fragment info based
      on a list of source fragment infos. Defaults to generate_target_info.

  Returns:
    List of FragmentMergeRule objects.
  """
  # TODO: b/443920649 - Implement once generate_target_info is implemented.
  raise NotImplementedError


def generate_rules_from_pipeline(
    fragment_infos: List[FragmentInfo],
    pipeline: PipelineSchedule,
    context: PipelineContext,
) -> Tuple[FragmentScheduleRules, FragmentMergeRules]:
  """Generates scheduling and merging rules from a PipelineSchedule.

  Args:
    fragment_infos: List of fragments to generate rules for.
    pipeline: PipelineSchedule containing predicates and options.
    context: PipelineContext with pipeline configuration.

  Returns:
    Tuple of (schedule_rules, merge_rules) generated from predicates.
  """
  all_schedule_rules = []
  if pipeline.schedule_predicates:
    for pred in pipeline.schedule_predicates:
      all_schedule_rules.extend(
          generate_schedule_rules(fragment_infos, pred, context)
      )

  all_merge_rules = []
  if pipeline.merge_predicates:
    for pred in pipeline.merge_predicates:
      all_merge_rules.extend(
          generate_merge_rules(fragment_infos, pred, context)
      )

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

  first_count = fragment.origins[0].transpose_count
  # Check if all origins have the same transpose count.
  if all(
      origin.transpose_count == first_count for origin in fragment.origins[1:]
  ):
    return first_count

  return None


def get_scheduling_unit_info(fragment: FragmentInfo) -> tuple[int, int] | None:
  """Returns (call_counter, transpose_count) if fragment is a valid scheduling unit.

  A fragment is a scheduling unit if:
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


def _validate_scheduling_fragments(
    f1: FragmentInfo, f2: FragmentInfo, error_msg: str
) -> tuple[int, int, int, int] | None:
  """Validates two fragments for scheduling and returns their info.

  Args:
    f1: First fragment to validate
    f2: Second fragment to validate
    error_msg: Error message for stage_id validation

  Returns:
    Tuple of (call_counter_f1, transpose_count_f1, call_counter_f2,
    transpose_count_f2) if both fragments are valid scheduling units, None
    otherwise.
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
  result = _validate_scheduling_fragments(
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
  result = _validate_scheduling_fragments(
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
  result = _validate_scheduling_fragments(
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
  result = _validate_scheduling_fragments(
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
  result = _validate_scheduling_fragments(
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
  result = _validate_scheduling_fragments(
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
