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

"""Tests for MPMD pipeline functions."""

from collections.abc import Sequence
import functools
import unittest
from absl.testing import parameterized
from shardy.integrations.python.jax.mpmd import pipeline


def _make_fragment(
    mesh_name: str = "mesh1",
    origins: Sequence[pipeline.FragmentOrigin] | None = None,
    **kwargs,
) -> pipeline.FragmentInfo:
  """Helper to create FragmentInfo with common defaults."""
  # Use None instead of [] to avoid shared mutable default argument
  if origins is None:
    origins = ()
  return pipeline.FragmentInfo(origins=origins, mesh_name=mesh_name, **kwargs)


class BasicScheduleBuildTest(unittest.TestCase):

  def test_predicate_based_schedule_example(self):
    def my_schedule_predicate(f1, f2, _):
      return f1.call_counter < f2.call_counter

    def my_merge_predicate(f1, f2, _):
      return f1.stage_id == f2.stage_id and f1.call_counter == f2.call_counter

    schedule = pipeline.PipelineSchedule(
        schedule_merge_rule_builders=[
            functools.partial(
                pipeline.build_schedule_rules_from_predicate,
                before_pred=my_schedule_predicate,
            ),
            functools.partial(
                pipeline.build_merge_rules_from_predicate,
                pred=my_merge_predicate,
            ),
        ],
        required_mpmd_options={},
    )

    self.assertIsNotNone(schedule)
    self.assertEqual(len(schedule.schedule_merge_rule_builders), 2)

  def test_direct_construction_schedule_example(self):
    # This test implements the same logic as the predicate-based example above.
    def custom_schedule_builder(fragment_infos, _):
      forward = sorted(
          [
              f
              for f in fragment_infos
              if f.origins
              and pipeline.maybe_unique_transpose_count(f) == 0
          ],
          key=lambda f: f.call_counter or 0,
      )
      backward = sorted(
          [
              f
              for f in fragment_infos
              if f.origins
              and pipeline.maybe_unique_transpose_count(f) == 1
          ],
          key=lambda f: f.call_counter or 0,
      )

      execution_order = []
      for fwd, bwd in zip(forward, backward):
        execution_order.extend([fwd, bwd])

      merge_rules = [
          pipeline.FragmentMergeRule(
              sources={fwd, bwd},
              target=pipeline._minimal_create_target_info([fwd, bwd]),
          )
          for fwd, bwd in zip(forward, backward)
          if fwd.stage_id == bwd.stage_id
      ]

      return [
          pipeline.FragmentScheduleRule(ordered_fragments=execution_order)
      ], merge_rules

    schedule = pipeline.PipelineSchedule(
        schedule_merge_rule_builders=[custom_schedule_builder],
        required_mpmd_options={},
    )

    self.assertIsNotNone(schedule)
    self.assertEqual(len(schedule.schedule_merge_rule_builders), 1)


class MinimalCreateTargetInfoTest(parameterized.TestCase):

  def test_empty_source_fragments_raises_error(self):
    with self.assertRaises(ValueError):
      pipeline._minimal_create_target_info([])

  def test_single_fragment(self):
    origin = pipeline.FragmentOrigin("comp1", transpose_count=1)
    fragment = _make_fragment(
        origins=(origin,),
        stage_id=5,
        call_counter=10,
        split_type=pipeline.SplitFragmentType.KEEP_TRANSFERRED,
    )

    result = pipeline._minimal_create_target_info([fragment])

    self.assertEqual(result.origins, (origin,))
    # minimal_create_target_info always sets these to None
    self.assertIsNone(result.stage_id)
    self.assertIsNone(result.call_counter)
    self.assertIsNone(result.split_type)
    self.assertEqual(result.mesh_name, "mesh1")

  def test_origins_union_preserves_all_transpose_counts(self):
    origin1 = pipeline.FragmentOrigin("comp1", transpose_count=0)
    origin2 = pipeline.FragmentOrigin("comp2", transpose_count=1)
    origin3 = pipeline.FragmentOrigin(
        "comp1", transpose_count=1
    )  # Different transpose_count

    fragment1 = pipeline.FragmentInfo(
        origins=(origin1, origin2), mesh_name="mesh1"
    )
    fragment2 = pipeline.FragmentInfo(origins=(origin3,), mesh_name="mesh1")

    result = pipeline._minimal_create_target_info([fragment1, fragment2])

    self.assertCountEqual(result.origins, (origin1, origin2, origin3))

  def test_origins_union_removes_duplicates(self):
    origin1 = pipeline.FragmentOrigin("comp1", transpose_count=0)
    origin2 = pipeline.FragmentOrigin("comp2", transpose_count=1)

    fragment1 = pipeline.FragmentInfo(
        origins=(origin1, origin2), mesh_name="mesh1"
    )
    # `origin1` also exists in fragment1 origins
    fragment2 = pipeline.FragmentInfo(origins=(origin1,), mesh_name="mesh1")

    result = pipeline._minimal_create_target_info([fragment1, fragment2])
    # Verify that the duplicate `origin1` does not remain
    self.assertCountEqual(result.origins, (origin1, origin2))

  def test_mesh_name_inconsistency_raises_error(self):
    """Test that inconsistent mesh_name values raise ValueError."""
    fragment1 = _make_fragment(mesh_name="mesh1")
    fragment2 = _make_fragment(mesh_name="mesh2")

    with self.assertRaises(ValueError) as cm:
      pipeline._minimal_create_target_info([fragment1, fragment2])
    self.assertIn(
        "Inconsistent mesh_name values: mesh1 vs mesh2", str(cm.exception)
    )

  def test_mesh_name_from_first_fragment(self):
    fragment1 = _make_fragment(mesh_name="mesh1")
    fragment2 = _make_fragment(mesh_name="mesh1")

    result = pipeline._minimal_create_target_info((fragment1, fragment2))

    self.assertEqual(result.mesh_name, "mesh1")

  def test_always_returns_none_for_optional_fields(self):
    """Test that stage_id, call_counter, and split_type are always None."""
    fragment1 = pipeline.FragmentInfo(
        origins=(pipeline.FragmentOrigin("comp1", transpose_count=0),),
        stage_id=5,
        call_counter=10,
        split_type=pipeline.SplitFragmentType.KEEP_TRANSFERRED,
        mesh_name="mesh1",
    )
    fragment2 = pipeline.FragmentInfo(
        origins=(pipeline.FragmentOrigin("comp2", transpose_count=1),),
        stage_id=5,
        call_counter=10,
        split_type=pipeline.SplitFragmentType.KEEP_TRANSFERRED,
        mesh_name="mesh1",
    )

    result = pipeline._minimal_create_target_info([fragment1, fragment2])

    # Regardless of input values, these should always be None
    self.assertIsNone(result.stage_id)
    self.assertIsNone(result.call_counter)
    self.assertIsNone(result.split_type)


if __name__ == "__main__":
  unittest.main()
