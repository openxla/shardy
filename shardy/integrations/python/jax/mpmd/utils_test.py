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
"""Utils tests."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jaxlib import _sdy_mpmd as mpmd_utils
import numpy as np
from shardy.integrations.python.jax.mpmd import utils


def setUpModule():
  chex.set_n_cpu_devices(8)


class UtilsTest(parameterized.TestCase):

  def test_meshes_and_sdy_specs_to_named_shardings(self):
    _, in_pytree = jax.tree_util.tree_flatten(((1, 2), 3, 4))
    _, out_pytree = jax.tree_util.tree_flatten((5, 6))

    devices = jax.devices()
    device_count = len(devices) // 2
    mesh1_devices = np.array(devices[:device_count]).reshape(
        (2, device_count // 2)
    )
    topology = {
        'm1': jax.sharding.Mesh(
            mesh1_devices, ('x', 'y'),
            axis_types=(jax.sharding.AxisType.Explicit,
                        jax.sharding.AxisType.Explicit)),
        'm2': jax.sharding.Mesh(np.array(devices[device_count:]), ('z',)),
    }

    host_named = (
        'unpinned_host' if jax.default_backend() == 'cpu' else 'pinned_host'
    )

    meshes_and_specs = mpmd_utils.FunctionIOShardingSpecsAndMeshes(
        [
            mpmd_utils.NamedSpmdShardingSpec(
                'm1', [['x'], ['y']], memory_kind=host_named  # m1 is a 2D mesh
            ),
            mpmd_utils.NamedSpmdShardingSpec(
                'm2',
                [[], []],  # m2 is a 1D mesh
            ),
            mpmd_utils.NamedSpmdShardingSpec(
                'm2', [['z']],  # m2 is a 1D mesh
            ),
            mpmd_utils.NamedSpmdShardingSpec(
                'm1', [['y'], []], unreduced_axes=['x']  # m1 is a 2D mesh
            ),
        ],
        [  # m1 is a 2D mesh
            mpmd_utils.NamedSpmdShardingSpec('m1', [['y'], ['x']]),
            mpmd_utils.NamedSpmdShardingSpec(
                'm1', [['x'], []], unreduced_axes=['y']
            ),
        ],
    )

    named_shardings = utils.meshes_and_sdy_specs_to_named_shardings(
        0, meshes_and_specs, in_pytree, out_pytree, topology
    )
    expected_input_specs = (
        (
            jax.sharding.NamedSharding(
                topology['m1'],
                jax.sharding.PartitionSpec(('x',), ('y',)),
                memory_kind=host_named,
            ),
            jax.sharding.NamedSharding(
                topology['m2'], jax.sharding.PartitionSpec()
            ),
        ),
        jax.sharding.NamedSharding(
            topology['m2'], jax.sharding.PartitionSpec(('z',))
        ),
        jax.sharding.NamedSharding(
            topology['m1'],
            jax.sharding.PartitionSpec('y', unreduced={'x'})
        ),
    )
    expected_output_specs = (
        jax.sharding.NamedSharding(
            topology['m1'], jax.sharding.PartitionSpec(('y',), ('x',))
        ),
        jax.sharding.NamedSharding(
            topology['m1'],
            jax.sharding.PartitionSpec('x', unreduced={'y'}),
        ),
    )
    self.assertEqual(
        named_shardings,
        utils.FunctionNamedShardings(
            expected_input_specs, expected_output_specs
        ),
    )

  @parameterized.parameters(
      dict(sdy_spec=[['x']], expected=jax.sharding.PartitionSpec('x')),
      dict(
          sdy_spec=[['x'], ['y']], expected=jax.sharding.PartitionSpec('x', 'y')
      ),
      dict(
          sdy_spec=[['x', 'y']], expected=jax.sharding.PartitionSpec(('x', 'y'))
      ),
      dict(
          sdy_spec=[[], ['y']], expected=jax.sharding.PartitionSpec(None, 'y')
      ),
      dict(sdy_spec=[['y'], []], expected=jax.sharding.PartitionSpec('y')),
      dict(
          sdy_spec=[['x']],
          expected=jax.sharding.PartitionSpec('x', unreduced={'y'}),
          unreduced={'y'},
          axis_types=(jax.sharding.AxisType.Auto,
                      jax.sharding.AxisType.Explicit)),
      dict(
          sdy_spec=[['y'], []],
          expected=jax.sharding.PartitionSpec('y', unreduced={'x'}),
          unreduced={'x'},
          axis_types=(jax.sharding.AxisType.Explicit,
                      jax.sharding.AxisType.Auto)),
  )
  def test_sdy_sharding_to_named_sharding(
      self, sdy_spec, expected, unreduced=None, axis_types=None):

    devices = jax.devices()
    device_count = len(devices) // 2
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((2, device_count)), ('x', 'y'),
        axis_types=axis_types,
    )

    self.assertEqual(
        utils._sdy_spec_to_named_sharding(sdy_spec, mesh, unreduced),
        jax.sharding.NamedSharding(mesh, expected),
    )


if __name__ == '__main__':
  absltest.main()
