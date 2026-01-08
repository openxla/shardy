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
"""Tests for jaxlib conversion utilities."""

from absl.testing import absltest
from absl.testing import parameterized
from jaxlib import _sdy_mpmd as jaxlib_mpmd

from shardy.integrations.python.jax.mpmd import jaxlib_utils
from shardy.integrations.python.jax.mpmd import types


class SplitTypeConversionTest(parameterized.TestCase):
  """Tests for SplitFragmentType conversion functions."""

  @parameterized.named_parameters(
      (
          'keep_transferred',
          types.SplitFragmentType.KEEP_TRANSFERRED,
          jaxlib_mpmd.SplitFragmentType.KEEP_TRANSFERRED,
      ),
      (
          'drop_transferred',
          types.SplitFragmentType.DROP_TRANSFERRED,
          jaxlib_mpmd.SplitFragmentType.DROP_TRANSFERRED,
      ),
      ('none', None, None),
  )
  def test_bidirectional_conversion(self, python_val, pybind_val):
    """Test SplitFragmentType conversion in both directions."""
    self.assertEqual(jaxlib_utils._to_jaxlib_split_type(python_val), pybind_val)
    self.assertEqual(
        jaxlib_utils._from_jaxlib_split_type(pybind_val), python_val
    )


class FragmentInfoConversionTest(parameterized.TestCase):
  """Tests for FragmentInfo conversion functions."""

  @parameterized.named_parameters(
      (
          'single_origin',
          types.FragmentInfo(
              origins=(types.FragmentOrigin('comp1', 0),), mesh_name='mesh1'
          ),
      ),
      (
          'multiple_origins',
          types.FragmentInfo(
              origins=(
                  types.FragmentOrigin('comp1', 0),
                  types.FragmentOrigin('comp2', 1),
              ),
              mesh_name='mesh1',
          ),
      ),
      (
          'all_fields',
          types.FragmentInfo(
              origins=(types.FragmentOrigin('comp1', 2),),
              stage_id=5,
              call_counter=3,
              split_type=types.SplitFragmentType.KEEP_TRANSFERRED,
              mesh_name='mesh2',
          ),
      ),
  )
  def test_roundtrip(self, fragment):
    """Test Python → pybind → Python roundtrip preserves data."""
    pybind_fragment = jaxlib_utils.convert_fragment_info_to_pybind(fragment)
    result = jaxlib_utils.convert_pybind_fragment_info_to_types(pybind_fragment)
    self.assertEqual(result, fragment)


class FragmentMergeRulesConversionTest(absltest.TestCase):
  """Tests for FragmentMergeRule conversion functions."""

  def test_single_rule(self):
    """Test converting single merge rule."""
    f1 = types.FragmentInfo(
        origins=(types.FragmentOrigin('f1', 0),), mesh_name='m1'
    )
    f2 = types.FragmentInfo(
        origins=(types.FragmentOrigin('f2', 0),), mesh_name='m1'
    )
    target = types.FragmentInfo(
        origins=(types.FragmentOrigin('f1', 0), types.FragmentOrigin('f2', 0)),
        mesh_name='m1',
    )

    rule = types.FragmentMergeRule(sources={f1, f2}, target=target)
    result = jaxlib_utils.convert_fragment_merge_rules_to_pybind([rule])

    self.assertLen(result, 1)
    self.assertLen(result[0].sources, 2)
    self.assertLen(result[0].target.origins, 2)


class FragmentScheduleRulesConversionTest(absltest.TestCase):
  """Tests for FragmentScheduleRule conversion functions."""

  def test_preserves_order(self):
    """Test that ordered_fragments order is preserved."""
    frags = [
        types.FragmentInfo(
            origins=(types.FragmentOrigin('first', 0),), mesh_name='m1'
        ),
        types.FragmentInfo(
            origins=(types.FragmentOrigin('second', 0),), mesh_name='m1'
        ),
        types.FragmentInfo(
            origins=(types.FragmentOrigin('third', 0),), mesh_name='m1'
        ),
    ]

    rule = types.FragmentScheduleRule(ordered_fragments=frags)
    result = jaxlib_utils.convert_fragment_schedule_rules_to_pybind([rule])

    self.assertEqual(
        result[0].ordered_fragments[0].origins[0].computation_name, 'first'
    )
    self.assertEqual(
        result[0].ordered_fragments[1].origins[0].computation_name, 'second'
    )
    self.assertEqual(
        result[0].ordered_fragments[2].origins[0].computation_name, 'third'
    )


if __name__ == '__main__':
  absltest.main()
