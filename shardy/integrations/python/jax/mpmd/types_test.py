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

"""Tests for the `types` user API."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np

from shardy.integrations.python.jax.mpmd import types


class MakeConfigTest(parameterized.TestCase):

  def setUp(self):
    chex.set_n_cpu_devices(8)
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_assignment',
          name_to_mesh_assignment={},
      ),
      dict(
          testcase_name='valid_assignment',
          name_to_mesh_assignment={'a': 'mesh1', 'b': 'mesh1'},
      ),
      dict(
          testcase_name='valid_assignment_with_memory_kinds',
          name_to_mesh_assignment={
              'a': 'mesh1#device',
              'b': 'mesh1#pinned_host',
          },
      ),
  )
  def test_valid_config_is_ok(self, name_to_mesh_assignment):
    topology = {'mesh1': jax.sharding.Mesh(jax.devices(), 'x')}
    config = types.make_config(topology, name_to_mesh_assignment)

    with self.subTest('test topology'):
      self.assertEqual(config.topology, topology)
    with self.subTest('test assignment'):
      self.assertEqual(
          config.name_to_mesh_assignment,
          name_to_mesh_assignment,
      )
    with self.subTest('test sharding_mesh'):
      self.assertEqual(config.sharding_mesh, topology['mesh1'].abstract_mesh)

  def test_empty_topology_raises_error(self):
    topology = {}
    assignment = {}

    with self.assertRaisesRegex(
        ValueError, '`topology` must have at least one mesh.'
    ):
      types.make_config(topology, assignment)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_invalid_assignment',
          name_to_mesh_assignment={'a': 'not_mesh', 'b': 'mesh1'},
          expected_error=r"{'not_mesh'} not in.*\['mesh1'\]",
      ),
      dict(
          testcase_name='multiple_invalid_assignment',
          name_to_mesh_assignment={'a': 'not_mesh', 'b': 'mesh1', 'c': 'mesh2'},
          expected_error=(
              r"{('mesh2', 'not_mesh'|'not_mesh', 'mesh2')} not in.*\['mesh1'\]"
          ),
      ),
      dict(
          testcase_name='duplicate_invalid_assignment',
          name_to_mesh_assignment={'a': 'not_mesh', 'b': 'not_mesh'},
          expected_error=r"{'not_mesh'} not in.*\['mesh1'\]",
      ),
      dict(
          testcase_name='invalid_mesh_in_input_assignment',
          name_to_mesh_assignment={},
          input_mesh_assignment=('non_mesh',),
          expected_error=(
              r"input_mesh_assignment: {'non_mesh'} not in.*\['mesh1'\]"
          ),
      ),
      dict(
          testcase_name='invalid_mesh_in_output_assignment',
          name_to_mesh_assignment={},
          output_mesh_assignment=('non_mesh',),
          expected_error=(
              r"output_mesh_assignment: {'non_mesh'} not in.*\['mesh1'\]"
          ),
      ),
      dict(
          testcase_name='invalid_stage_assignment',
          name_to_mesh_assignment={},
          name_to_stage_assignment={'a': -1, 'b': -2},
          expected_error='Stage ids must be positive integers, but got: -1,-2.',
      ),
  )
  def test_assigned_mesh_not_in_topology_raises_error(
      self,
      name_to_mesh_assignment,
      name_to_stage_assignment=None,
      input_mesh_assignment=(),
      output_mesh_assignment=(),
      expected_error=None,
  ):
    topology = {'mesh1': jax.sharding.Mesh(jax.devices(), 'x')}

    with self.assertRaisesRegex(ValueError, expected_error):
      types.make_config(
          topology,
          name_to_mesh_assignment,
          name_to_stage_assignment=name_to_stage_assignment,
          input_mesh_assignment=input_mesh_assignment,
          output_mesh_assignment=output_mesh_assignment,
      )

  def test_invalid_mesh_name_raises_error(self):
    with self.subTest('use@'):
      with self.assertRaisesRegex(
          ValueError, 'mesh@ uses one of the reserved substrings'
      ):
        types.make_config({'mesh@': jax.sharding.Mesh(jax.devices(), 'x')}, {})

  def test_memory_kind_in_mesh_name_raises_error(self):
    with self.subTest('pinned_host'):
      with self.assertRaisesRegex(ValueError, 'reserved for memory_kinds'):
        types.make_config(
            {'mesh#pinned_host': jax.sharding.Mesh(jax.devices(), 'x')}, {}
        )
    with self.subTest('device'):
      with self.assertRaisesRegex(ValueError, 'reserved for memory_kinds'):
        types.make_config(
            {'mesh#device': jax.sharding.Mesh(jax.devices(), 'x')}, {}
        )


class UtilsTest(absltest.TestCase):

  def setUp(self):
    chex.set_n_cpu_devices(8)
    super().setUp()

  def test_mesh_names_returns_expected(self):
    devices = jax.devices()
    devices_per_mesh = len(devices) // 2
    m1 = jax.sharding.Mesh(devices[devices_per_mesh:], 'x')
    m2 = jax.sharding.Mesh(devices[:devices_per_mesh], 'x')
    topology = {'mesh1': m1, 'mesh2': m2}

    sharding = jax.sharding.NamedSharding(m1, jax.sharding.PartitionSpec())
    x = jax.ShapeDtypeStruct((4, 4), np.float32, sharding=sharding)
    x_no_sharding = jax.ShapeDtypeStruct((4, 4), np.float32, sharding=None)

    pytree = (x, (m2, sharding), None, m1.abstract_mesh, x_no_sharding)

    result = types.mesh_names(pytree=pytree, topology=topology)
    self.assertEqual(result, ('mesh1', ('mesh2', 'mesh1'), None, None, None))

  def test_mesh_names_raises_error_when_mesh_not_in_topology(self):
    pytree = jax.sharding.Mesh(jax.devices(), 'x')

    with self.assertRaisesRegex(ValueError, 'Mesh .* with devices '):
      types.mesh_names(pytree=pytree, topology={})

  def test_mesh_names_raises_error_when_invalid_type(self):
    pytree = 'x'

    with self.assertRaisesRegex(ValueError, 'Unexpected type: '):
      types.mesh_names(pytree=pytree, topology={})


class MakePartitioningOptionsTest(parameterized.TestCase):

  @parameterized.product(
      option=[
          'mpmd_infer_transfers',
          'mpmd_infer_cross_mesh_reductions',
          'mpmd_merge_inferred_with_cloning_during_import',
          'mpmd_fragment_remat',
          'mpmd_merge_remat_fragments',
          'mpmd_split_bwd_fragments',
          'mpmd_assume_homogeneous_devices',
          'mpmd_absorb_inferred_fragments_on_entry_point_function',
          'mpmd_copy_constant_creation_from_producer_to_consumer',
          'mpmd_apply_merge_transfers_pass',
      ],
      value=[True, False],
  )
  def test_boolean_option(self, option, value):
    types._validate_partitioning_options({option: value})

  @parameterized.parameters(
      'None',
      '1F1B',
      'GPipe',
      'Circular',
      'CircularWithReversedBackward',
      'GPipeBut1F1BForLastMesh',
      'ZeroBubbleH1',
      'ZeroBubbleH2ZeroTxLatency',
      'ZeroBubbleH2HalfTxLatency',
      'ZeroBubbleH2FullTxLatency',
      'ParallelPipelinesWithWrapAround',
  )
  def test_mpmd_pipeline_schedule_option(self, schedule_as_str):
    types._validate_partitioning_options(
        {'mpmd_pipeline_schedule': schedule_as_str}
    )

  def test_unsupported_option_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        "`mpmd_options` contains unsupported options {'unsupported_option'}",
    ):
      types._validate_partitioning_options({'unsupported_option': ''})

  @parameterized.parameters(
      ('mpmd_infer_transfers', None),
      ('mpmd_infer_cross_mesh_reductions', 0),
      ('mpmd_merge_inferred_with_cloning_during_import', ''),
      ('mpmd_fragment_remat', None),
      ('mpmd_merge_remat_fragments', 0.0),
      ('mpmd_split_bwd_fragments', ''),
      ('mpmd_assume_homogeneous_devices', []),
      ('mpmd_absorb_inferred_fragments_on_entry_point_function', set()),
      ('mpmd_copy_constant_creation_from_producer_to_consumer', dict()),
      ('mpmd_apply_merge_transfers_pass', None),
  )
  def test_non_boolean_value_for_boolean_option_raises(self, option, value):
    with self.assertRaisesRegex(
        ValueError,
        f'Option {option} has value .*, which is not a boolean.',
    ):
      types._validate_partitioning_options({option: value})

  def test_invalid_pipeline_schedule_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        'Option mpmd_pipeline_schedule has value INVALID, which is not a'
        ' valid pipeline schedule.',
    ):
      types._validate_partitioning_options(
          {'mpmd_pipeline_schedule': 'INVALID'}
      )


class ValidateAndMergePartitioningOptionsTest(parameterized.TestCase):
  """Tests for validate_and_merge_partitioning_options function."""

  def test_no_pipeline_options_returns_user_options(self):
    user_options = {'mpmd_infer_transfers': True}
    result = types.validate_and_merge_partitioning_options(
        pipeline_required_options=None, user_provided_options=user_options
    )
    self.assertEqual(user_options, result)

  def test_no_user_options_returns_pipeline_options(self):
    result = types.validate_and_merge_partitioning_options(
        pipeline_required_options={'mpmd_infer_transfers': True},
        user_provided_options=None,
    )
    self.assertEqual({'mpmd_infer_transfers': True}, result)

  def test_no_conflict_with_compatible_options(self):
    result = types.validate_and_merge_partitioning_options(
        pipeline_required_options={
            'mpmd_infer_transfers': True,
            'mpmd_split_bwd_fragments': True,
        },
        user_provided_options={
            'mpmd_infer_transfers': True,
            'mpmd_fragment_remat': False,
        },
    )
    expected = {
        'mpmd_infer_transfers': True,
        'mpmd_fragment_remat': False,
        'mpmd_split_bwd_fragments': True,
    }
    self.assertEqual(expected, result)

  def test_conflict_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'(?s)Conflicting partitioning options detected:.*'
        r'mpmd_infer_transfers.*'
        r'pipeline schedule requires True.*'
        r'user specified False',
    ):
      types.validate_and_merge_partitioning_options(
          pipeline_required_options={'mpmd_infer_transfers': True},
          user_provided_options={'mpmd_infer_transfers': False},
      )

  def test_pipeline_schedule_conflict_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'(?s)Conflicting partitioning options detected:.*'
        r'mpmd_pipeline_schedule.*'
        r'pipeline schedule requires 1F1B.*'
        r'user specified GPipe',
    ):
      types.validate_and_merge_partitioning_options(
          pipeline_required_options={'mpmd_pipeline_schedule': '1F1B'},
          user_provided_options={'mpmd_pipeline_schedule': 'GPipe'},
      )

  def test_warnings_for_new_options(self):
    with self.assertLogs(level='WARNING') as logs:
      types.validate_and_merge_partitioning_options(
          pipeline_required_options={'mpmd_infer_transfers': True},
          user_provided_options=None,
      )
    self.assertLen(logs.output, 1)
    self.assertIn('mpmd_infer_transfers', logs.output[0])
    self.assertIn('required by pipeline schedule', logs.output[0])


if __name__ == '__main__':
  absltest.main()
