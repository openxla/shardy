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
"""Tests for mpmd.jit user API that uses sdy propagation code path."""

from collections.abc import Sequence
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax.core import scope
import jax
from jax import sharding
import jax.numpy as jnp
import jaxtyping
import numpy as np

from shardy.integrations.python.jax import mpmd
from shardy.integrations.python.jax.mpmd import stages
from shardy.integrations.python.jax.mpmd import test_utils
from shardy.integrations.python.jax.mpmd import types as mpmd_types


PyTree = jaxtyping.PyTree
VariableDict = scope.VariableDict


def _replicated(
    mesh: sharding.Mesh | sharding.AbstractMesh,
) -> sharding.NamedSharding:
  """For readability. Makes it clear the tensor is fully replicated."""
  return test_utils.named_sharding(mesh)


def get_2_stage_mpmd_config(
    options: mpmd_types.PartitioningOptions | None = None,
):
  return mpmd_types.make_config(
      test_utils.get_two_mesh_topology(),
      {'stage1': 'mesh1', 'stage2': 'mesh2'},
      partitioning_options=options,
  )


def setUpModule():
  chex.set_n_cpu_devices(8)
  # We need `jax_legacy_prng_key='allow'`` because we create a PRNG key with
  # integer seed.
  jax.config.update('jax_legacy_prng_key', 'allow')
  jax.config.update('jax_numpy_dtype_promotion', 'standard')

class SdyPropagationTest(parameterized.TestCase):

  def test_mpmd_hello_world(self):
    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
      stage2 = mpmd.named_computation(lambda a: a @ w2, name='stage2')(stage1)
      return stage2

    mpmd_config = get_2_stage_mpmd_config()
    topology = mpmd_config.topology
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
        np.ones((3, 5), dtype=jnp.float32),
        np.ones((5, 7), dtype=jnp.float32),
        np.ones((10, 3), dtype=jnp.float32),
    )

    self.assertIn('mpmd.sdy_lowered', lowered.as_text('mpmd'))
    self.assertIn('ifrt.Call', lowered.as_text('ifrt'))
    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (
            _replicated(topology['mesh1']),
            _replicated(topology['mesh2']),
            _replicated(topology['mesh1']),
        ),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        _replicated(topology['mesh2']),
    )

  def test_mpmd_override_jit_func_name(self):
    def original_function_name(w1, w2, x):
      stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
      stage2 = mpmd.named_computation(lambda a: a @ w2, name='stage2')(stage1)
      return stage2

    mpmd_config = get_2_stage_mpmd_config()
    lowered: stages.MpmdLowered = mpmd.jit(
        original_function_name,
        mpmd_config,
        override_func_name='override_func_name',
    ).lower(
        np.ones((3, 5), dtype=jnp.float32),
        np.ones((5, 7), dtype=jnp.float32),
        np.ones((10, 3), dtype=jnp.float32),
    )
    self.assertIn('override_func_name', lowered.as_text('mpmd'))
    self.assertNotIn('original_function_name', lowered.as_text('mpmd'))

  def test_jit_twice(self):
    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
      stage2 = mpmd.named_computation(lambda a: a @ w2, name='stage2')(stage1)
      return stage2

    mpmd_config = get_2_stage_mpmd_config()
    lowered1: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
        np.ones((3, 5), dtype=jnp.float32),
        np.ones((5, 7), dtype=jnp.float32),
        np.ones((10, 3), dtype=jnp.float32),
    )

    lowered2: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
        np.ones((3, 5), dtype=jnp.float32),
        np.ones((5, 7), dtype=jnp.float32),
        np.ones((10, 3), dtype=jnp.float32),
    )

    self.assertEqual(
        lowered1.function_named_shardings.input_specs,
        lowered2.function_named_shardings.input_specs,
    )

    self.assertEqual(
        lowered2.function_named_shardings.output_specs,
        lowered1.function_named_shardings.output_specs,
    )

  def test_compile_fails_because_not_pathways_backend(self):

    lowered: stages.MpmdLowered = mpmd.jit(
        lambda x: x, get_2_stage_mpmd_config()
    ).lower(np.ones((3, 5), dtype=jnp.float32))
    with self.assertRaisesRegex(
        NotImplementedError,
        'MPMD functions can only be compiled through Pathways, but only the'
        r" following backends are available: \('cpu',\)",
    ):
      lowered.compile()

  def test_mpmd_simple_spmd_sharding(self):
    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1', 'stage2': 'mesh2'}

    def batch_sharded_matmul(batched_lhs, batched_rhs):
      sharded_lhs = test_utils.with_sharding(
          batched_lhs, partition_spec=('x',), mesh=topology['mesh1']
      )
      sharded_rhs = test_utils.with_sharding(
          batched_rhs, partition_spec=('x',), mesh=topology['mesh1']
      )
      return sharded_lhs @ sharded_rhs

    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(batch_sharded_matmul, name='stage1')(
          x, w1
      )
      stage2 = mpmd.named_computation(batch_sharded_matmul, name='stage2')(
          stage1, w2
      )
      return stage2

    config = mpmd_types.make_config(topology, assignment)
    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn,
        config,
        in_shardings=(
            test_utils.named_sharding(config.sharding_mesh, 'x'),
            test_utils.named_sharding(topology['mesh2'], 'x'),
            test_utils.named_sharding(config.sharding_mesh, 'x'),
        ),
    ).lower(
        np.ones((16, 3, 5), dtype=jnp.float32),
        np.ones((16, 5, 7), dtype=jnp.float32),
        np.ones((16, 10, 3), dtype=jnp.float32),
    )

    self.assertIn('mpmd.sdy_lowered', lowered.as_text('mpmd'))
    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (
            test_utils.named_sharding(topology['mesh1'], 'x'),
            test_utils.named_sharding(topology['mesh2'], 'x'),
            test_utils.named_sharding(topology['mesh1'], 'x'),
        ),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        test_utils.named_sharding(topology['mesh2'], 'x'),
    )

  @parameterized.parameters(
      dict(different_axis_sizes=True, different_axis_names=False),
      dict(different_axis_sizes=False, different_axis_names=True),
      dict(different_axis_sizes=True, different_axis_names=True),
  )
  def test_mpmd_invalid_sharding_errors_on_meshes_with_different_sizes(
      self, different_axis_sizes=False, different_axis_names=False
  ):
    topology = test_utils.get_two_mesh_topology()
    config = mpmd_types.make_config(topology, {})
    devices = (
        jax.devices() if different_axis_sizes else config._spmd_mesh.devices
    )
    axis_name = 'not_x' if different_axis_names else 'x'
    bad_mesh = sharding.Mesh(devices, (axis_name,))
    shardings = (
        test_utils.named_sharding(bad_mesh),
        test_utils.named_sharding(bad_mesh),
    )

    inputs = (
        np.ones((16, 3, 5), dtype=jnp.float32),
        np.ones((16, 5, 7), dtype=jnp.float32),
    )
    with self.subTest('in_shardings'):
      with self.assertRaisesRegex(ValueError, 'in_shardings'):
        mpmd.jit(lambda x, y: (x, y), config, in_shardings=shardings).lower(
            *inputs
        )

    with self.subTest('out_shardings'):
      with self.assertRaisesRegex(ValueError, 'out_shardings'):
        mpmd.jit(lambda x, y: (x, y), config, out_shardings=shardings).lower(
            *inputs
        )

    with self.subTest('lowering_args'):
      with self.assertRaisesRegex(ValueError, 'lowering_args'):
        mpmd.jit(lambda x, y: (x, y), config).lower(
            *jax.device_put(inputs, shardings)
        )

  def test_spmd_sharding_with_mesh_context_same_tpu_topology_passes(self):
    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1', 'stage2': 'mesh2'}
    batch_sharded_on_x = sharding.PartitionSpec('x')

    def batch_sharded_matmul(batched_lhs, batched_rhs):
      sharded_lhs = jax.lax.with_sharding_constraint(
          batched_lhs, batch_sharded_on_x
      )
      sharded_rhs = jax.lax.with_sharding_constraint(
          batched_rhs, batch_sharded_on_x
      )
      return sharded_lhs @ sharded_rhs

    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(batch_sharded_matmul, name='stage1')(
          x, w1
      )
      stage2 = mpmd.named_computation(batch_sharded_matmul, name='stage2')(
          stage1, w2
      )
      return stage2

    config = mpmd_types.make_config(topology, assignment)
    # This is ok since mesh1 and mesh2 have the same TPU topology.
    with topology['mesh1']:
      lowered: stages.MpmdLowered = mpmd.jit(
          mpmd_fn,
          config,
          in_shardings=(
              test_utils.named_sharding(config.sharding_mesh, 'x'),
              test_utils.named_sharding(config.sharding_mesh, 'x'),
              test_utils.named_sharding(config.sharding_mesh, 'x'),
          ),
      ).lower(
          np.ones((16, 3, 5), dtype=jnp.float32),
          np.ones((16, 5, 7), dtype=jnp.float32),
          np.ones((16, 10, 3), dtype=jnp.float32),
      )

    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (
            test_utils.named_sharding(topology['mesh1'], 'x'),
            test_utils.named_sharding(topology['mesh2'], 'x'),
            test_utils.named_sharding(topology['mesh1'], 'x'),
        ),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        test_utils.named_sharding(topology['mesh2'], 'x'),
    )

  def test_mixed_in_sharding_and_sharded_args_gets_correct_sharding(self):
    def f(x, y):
      return x + y

    config = get_2_stage_mpmd_config()
    # x and y are sharded across different meshes.
    pspec = jax.sharding.PartitionSpec('x')
    x_sharding = jax.sharding.NamedSharding(config.topology['mesh1'], pspec)
    y_sharding = jax.sharding.NamedSharding(config.topology['mesh2'], pspec)
    x = jnp.ones((8, 8), dtype=jnp.float32)
    y = jax.device_put(x, y_sharding)
    lowered = mpmd.jit(f, config, in_shardings=(x_sharding, None)).lower(x, y)

    self.assertEqual(
        tuple(x.spec for x in lowered.function_named_shardings.input_specs),
        (pspec, pspec),
    )

  def test_in_sharding_propagates_through_call_op_to_lowered_object(self):
    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1'}

    def call_fn(w1, x):
      stage1 = mpmd.named_computation(jnp.add, name='stage1')(x, w1)
      return stage1

    def mpmd_fn(w1, x):
      y = mpmd.call(call_fn, call_counter=0)(w1, x)
      z = mpmd.call(call_fn, call_counter=1)(w1, y)
      return z

    mesh1 = topology['mesh1']
    batch_sharded_on_x = sharding.PartitionSpec('x')
    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn,
        mpmd_types.make_config(topology, assignment),
        in_shardings=(
            sharding.NamedSharding(mesh1, batch_sharded_on_x),
            _replicated(mesh1),
        ),
    ).lower(
        np.ones((16, 10, 3), dtype=jnp.float32),
        np.ones((16, 10, 3), dtype=jnp.float32),
    )

    # Note that `x` gets sharded even though it starts replicated,
    # which is evidence that the sharding is propagated from `w1` to
    # `x` in the `mpmd.call`. And similarly we see the output gets sharded.
    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (
            sharding.NamedSharding(topology['mesh1'], batch_sharded_on_x),
            _replicated(topology['mesh1']),
        ),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        sharding.NamedSharding(topology['mesh1'], batch_sharded_on_x),
    )

  def test_in_sharding_propagates_to_lowered_object(self):
    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1', 'stage2': 'mesh2'}

    def matmul(lhs, rhs):
      return lhs @ rhs

    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(matmul, name='stage1')(x, w1)
      stage2 = mpmd.named_computation(matmul, name='stage2')(stage1, w2)
      return stage2

    config = mpmd_types.make_config(topology, assignment)
    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn,
        config,
        in_shardings=(
            test_utils.named_sharding(config.sharding_mesh, 'x'),
            test_utils.named_sharding(config.sharding_mesh, 'x'),
            test_utils.named_sharding(config.sharding_mesh, 'x'),
        ),
    ).lower(
        np.ones((16, 3, 5), dtype=jnp.float32),
        np.ones((16, 5, 7), dtype=jnp.float32),
        np.ones((16, 10, 3), dtype=jnp.float32),
    )

    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (
            test_utils.named_sharding(topology['mesh1'], 'x'),
            test_utils.named_sharding(topology['mesh2'], 'x'),
            test_utils.named_sharding(topology['mesh1'], 'x'),
        ),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        test_utils.named_sharding(topology['mesh2'], 'x'),
    )

  def test_out_sharding_propagates_to_lowered_object(self):
    mpmd_config = get_2_stage_mpmd_config()
    topology = mpmd_config.topology

    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(jnp.add, name='stage1')(x, w1)
      stage2 = mpmd.named_computation(jnp.add, name='stage2')(stage1, w2)
      return stage2

    batch_sharded_on_x = sharding.PartitionSpec('x')
    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn,
        mpmd_config,
        out_shardings=sharding.NamedSharding(
            mpmd_config.sharding_mesh, batch_sharded_on_x
        ),
    ).lower(
        np.ones((16, 3, 5), dtype=jnp.float32),
        np.ones((16, 3, 5), dtype=jnp.float32),
        np.ones((16, 3, 5), dtype=jnp.float32),
    )

    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        sharding.NamedSharding(topology['mesh2'], batch_sharded_on_x),
    )

  def test_sharded_args_propagates_to_lowered_object(self):
    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1', 'stage2': 'mesh2'}

    def matmul(lhs, rhs):
      return lhs @ rhs

    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(matmul, name='stage1')(x, w1)
      stage2 = mpmd.named_computation(matmul, name='stage2')(stage1, w2)
      return stage2

    mesh1 = topology['mesh1']
    args = (
        np.ones((16, 3, 5), dtype=jnp.float32),
        np.ones((16, 5, 7), dtype=jnp.float32),
        np.ones((16, 10, 3), dtype=jnp.float32),
    )
    batch_sharded_on_x = sharding.PartitionSpec('x')
    sharded_args = jax.device_put(
        args,
        (
            sharding.NamedSharding(mesh1, batch_sharded_on_x),
            sharding.NamedSharding(mesh1, batch_sharded_on_x),
            sharding.NamedSharding(mesh1, batch_sharded_on_x),
        ),
    )
    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn,
        mpmd_types.make_config(topology, assignment),
    ).lower(*sharded_args)

    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (
            sharding.NamedSharding(topology['mesh1'], batch_sharded_on_x),
            sharding.NamedSharding(topology['mesh2'], batch_sharded_on_x),
            sharding.NamedSharding(topology['mesh1'], batch_sharded_on_x),
        ),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        sharding.NamedSharding(topology['mesh2'], batch_sharded_on_x),
    )

  def test_in_sharding_propagates_through_transferred_inputs(self):
    def fn(x, y):
      x = mpmd.named_tensor(x, 'm1')
      return mpmd.named_computation(jnp.add, name='m2')(x, y)

    topology = test_utils.get_two_mesh_topology()
    assignment = {'m1': 'mesh1', 'm2': 'mesh2'}
    config = mpmd.make_config(topology, assignment)
    lowered = mpmd.jit(
        fn,
        config,
        in_shardings=(
            test_utils.named_sharding(config.sharding_mesh, 'x'),
            _replicated(config.sharding_mesh),
        ),
    ).lower(
        jax.ShapeDtypeStruct((8, 2), jnp.int32),
        jax.ShapeDtypeStruct((8, 2), jnp.int32),
    )

    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (
            test_utils.named_sharding(topology['mesh1'], 'x'),
            _replicated(topology['mesh2']),
        ),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        test_utils.named_sharding(topology['mesh2'], 'x'),
    )

  def test_out_sharding_is_always_respected(self):
    def fn(x, y):
      x = mpmd.named_computation(jnp.add, name='m1')(x, y)
      x = mpmd.named_tensor(x, 'm2')
      return x, x, y, y

    topology = test_utils.get_two_mesh_topology()
    assignment = {'m1': 'mesh1', 'm2': 'mesh2'}
    config = mpmd.make_config(topology, assignment)
    lowered = mpmd.jit(
        fn,
        config,
        in_shardings=(
            _replicated(config.sharding_mesh),
            _replicated(config.sharding_mesh),
        ),
        out_shardings=(
            test_utils.named_sharding(config.sharding_mesh, 'x'),
            test_utils.named_sharding(config.sharding_mesh, None, 'x'),
            test_utils.named_sharding(config.topology['mesh2'], None, 'x'),
            test_utils.named_sharding(config.sharding_mesh, 'x'),
        ),
    ).lower(
        jax.ShapeDtypeStruct((8, 16), jnp.int32),
        jax.ShapeDtypeStruct((8, 16), jnp.int32),
    )

    self.assertEqual(
        lowered.function_named_shardings.input_specs,
        (_replicated(topology['mesh1']), _replicated(topology['mesh1'])),
    )
    self.assertEqual(
        lowered.function_named_shardings.output_specs,
        (
            test_utils.named_sharding(topology['mesh2'], 'x'),
            test_utils.named_sharding(topology['mesh2'], None, 'x'),
            test_utils.named_sharding(topology['mesh1'], None, 'x'),
            test_utils.named_sharding(topology['mesh1'], 'x'),
        ),
    )

  def test_reshard_outside_fragment_fails(self):
    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1', 'stage2': 'mesh2', 'stage3': 'mesh1'}

    def batch_sharded_matmul(batched_lhs, batched_rhs, partition_spec):
      sharded_lhs = test_utils.with_sharding(
          batched_lhs, partition_spec=partition_spec, mesh=topology['mesh1']
      )
      sharded_rhs = test_utils.with_sharding(
          batched_rhs, partition_spec=partition_spec, mesh=topology['mesh1']
      )
      return sharded_lhs @ sharded_rhs

    def mpmd_fn(w1, w2, x):
      # This function does:
      # stage1 ----> stage3
      #    \-------> stage2
      # to force stage1 and stage3 not to merge (because stage2 blocks merging)
      # and hence resharding happens outside the named_computation, unless
      # we propagate shardings forward.
      stage1 = mpmd.named_computation(lambda x, y: x @ y, name='stage1')(x, w1)
      stage2 = mpmd.named_computation(
          lambda x, y: batch_sharded_matmul(x, y, (None, 'x')), name='stage2'
      )(stage1, w2)
      stage3 = mpmd.named_computation(
          lambda x, y: batch_sharded_matmul(x, y, (None, 'x')), name='stage3'
      )(stage1, w2)
      return stage2, stage3

    # Does not error.
    mpmd.jit(
        mpmd_fn,
        mpmd_types.make_config(topology, assignment),
    ).lower(
        np.ones((16, 16, 3, 5), dtype=jnp.float32),
        np.ones((16, 16, 5, 7), dtype=jnp.float32),
        np.ones((16, 16, 10, 3), dtype=jnp.float32),
    )

  def test_unused_func_args(self):
    def mpmd_fn(w1, w2, x):
      del w2
      stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
      return stage1

    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh2'}

    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn,
        mpmd_types.make_config(topology, assignment),
    ).lower(
        np.ones((3, 5), dtype=jnp.float32),
        np.ones((8, 7), dtype=jnp.float32),
        np.ones((10, 3), dtype=jnp.float32),
    )

    input_specs_meshes = lowered.function_named_shardings.input_specs
    output_specs_meshes = lowered.function_named_shardings.output_specs
    self.assertEqual(
        input_specs_meshes,
        (
            _replicated(topology['mesh2']),
            _replicated(topology['mesh1']),
            _replicated(topology['mesh2']),
        ),
    )
    self.assertEqual(
        output_specs_meshes,
        _replicated(topology['mesh2']),
    )

  def test_incomplete_assignment(self):
    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
      stage2 = mpmd.named_computation(lambda a: a @ w2, name='stage2')(stage1)
      return stage2

    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1'}

    with self.assertRaisesRegex(
        RuntimeError,
        (
            "Top-level NamedComputation 'stage2' is not assigned to a mesh in"
            ' the user-defined named-to-mesh assignment.'
        ),
    ):
      mpmd.jit(
          mpmd_fn,
          mpmd_types.make_config(topology, assignment),
      ).lower(
          np.ones((3, 5), dtype=jnp.float32),
          np.ones((5, 7), dtype=jnp.float32),
          np.ones((10, 3), dtype=jnp.float32),
      )

  def test_nested_assignment_fails_if_different_mesh(self):
    def mpmd_fn(w1, w2, x):
      inner = mpmd.named_computation(lambda a: a @ w2, name='inner')
      outer = mpmd.named_computation(lambda a: inner(a @ w1), name='outer')
      return outer(x)

    topology = test_utils.get_two_mesh_topology()
    assignment = {'outer': 'mesh1', 'inner': 'mesh2'}

    with self.assertRaisesRegex(
        RuntimeError,
        (
            "NamedComputation 'inner' is nested in a NamedComputation 'outer'"
            ' which has a different mesh or stage assignment.'
        ),
    ):
      mpmd.jit(
          mpmd_fn,
          mpmd_types.make_config(topology, assignment),
      ).lower(
          np.ones((3, 5), dtype=jnp.float32),
          np.ones((5, 7), dtype=jnp.float32),
          np.ones((10, 3), dtype=jnp.float32),
      )

  def test_nested_assignment_ok_if_same_mesh(self):
    def mpmd_fn(w1, w2, x):
      inner = mpmd.named_computation(lambda a: a @ w2, name='inner')
      outer = mpmd.named_computation(lambda a: inner(a @ w1), name='outer')
      return outer(x)

    topology = test_utils.get_two_mesh_topology()
    assignment = {'outer': 'mesh1', 'inner': 'mesh1'}

    lowered = mpmd.jit(
        mpmd_fn,
        mpmd_types.make_config(topology, assignment),
    ).lower(
        np.ones((3, 5), dtype=jnp.float32),
        np.ones((5, 7), dtype=jnp.float32),
        np.ones((10, 3), dtype=jnp.float32),
    )

    # There should only be one fragment and fragment_call.
    self.assertEqual(lowered.as_text('mpmd').count('fragment_call'), 1)

  def test_reserved_name(self):
    mesh1 = sharding.Mesh(jax.devices(), ('x',))
    topology = {'mesh1': mesh1}
    assignment = {'partir.foo': 'mesh1'}

    with self.assertRaisesRegex(
        ValueError,
        'computation names that start with `partir.` are reserved for the'
        ' compiler.',
    ):
      mpmd.jit(
          functools.partial(mpmd.named_computation, name='partir.foo')(jnp.add),
          mpmd_types.make_config(topology, assignment),
      ).lower(
          np.ones((3, 5), dtype=jnp.float32), np.ones((3, 5), dtype=jnp.float32)
      )

  def test_invalid_mesh_in_assignment(self):
    def mpmd_fn(w1, w2, x):
      stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
      stage2 = mpmd.named_computation(lambda a: a @ w2, name='stage2')(stage1)
      return stage2

    topology = test_utils.get_two_mesh_topology()
    assignment = {'stage1': 'mesh1', 'stage2': 'not_in_topology'}

    with self.assertRaisesRegex(
        ValueError,
        'Invalid mesh name',
    ):
      mpmd.jit(
          mpmd_fn,
          mpmd_types.make_config(topology, assignment),
      ).lower(
          np.ones((3, 5), dtype=jnp.float32),
          np.ones((5, 7), dtype=jnp.float32),
          np.ones((10, 3), dtype=jnp.float32),
      )

  def test_stress_mesh_assignment(self):
    def f(x):
      return mpmd.named_computation(
          lambda: [x + np.ones((8, 8))] * 24, name='f'
      )()

    topology = {'m': sharding.Mesh(jax.devices(), ('x',))}
    lowered = mpmd.jit(f, mpmd_types.make_config(topology, {'f': 'm'})).lower(
        np.ones((8, 8))
    )
    self.assertEqual(lowered.as_text('mpmd').count('fragment_call'), 1)

  def test_partitioned_functions(self):
    topology = test_utils.get_two_mesh_topology()
    assignment = {'name1': 'mesh1', 'name2': 'mesh2'}

    def f1():
      x = mpmd.named_tensor(np.ones((16, 16), dtype=np.float32), name='name1')
      y = mpmd.named_tensor(
          np.full((16, 16), 2, dtype=np.float32), name='name2'
      )
      return (
          test_utils.with_sharding(x, ('x',), topology['mesh1']),
          test_utils.with_sharding(y, ('x',), topology['mesh1']),
      )

    mpmd.jit(f1, mpmd_types.make_config(topology, assignment)).lower()

  def test_duplicate_outputs(self):
    def fn(x, y):
      x = mpmd.named_computation(lambda x, y: x + y, name='stage1')(x, y)
      return x, x

    mpmd.jit(fn, get_2_stage_mpmd_config(), donate_argnums=0).lower(0, 0)

  def test_donate_none(self):
    def fn(x, y):
      x = mpmd.named_computation(lambda x, y: x + y, name='stage1')(x, y)
      return x, x

    mpmd.jit(fn, get_2_stage_mpmd_config(), donate_argnums=None).lower(0, 0)

  def test_donate_tree(self):
    def fn(args):
      y = mpmd.named_computation(lambda x: x[0] + x[1], name='stage1')(args)
      return y

    mpmd_config = get_2_stage_mpmd_config()
    topology = mpmd_config.topology
    arr: list[jax.Array] = jax.device_put(
        [np.array([1]), np.array([2])],
        _replicated(topology['mesh1']),
    )
    mpmd.jit(fn, mpmd_config, donate_argnums=0).lower(
        [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in arr]
    )

  def test_mesh_inference_fails(self):
    mpmd_config = get_2_stage_mpmd_config()

    def f(x):
      x = mpmd.named_computation(lambda x: x + 1, name='stage1')(x)
      x = jax.lax.sqrt(x)
      y = mpmd.named_computation(lambda x: x + 1, name='stage2')(x)
      y = y + 1

      return y

    with self.assertRaisesRegex(
        RuntimeError,
        'Mesh assignment is not possible for op',
    ):
      mpmd.jit(f, mpmd_config).lower(np.ones((3, 5), dtype=jnp.float32))

  def test_mesh_inference_fails_because_operands_from_different_meshes(self):
    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        {'stage1': 'mesh1', 'stage2': 'mesh2'},
        input_mesh_assignment=('mesh1', None),
    )

    def f(x, y):
      x = x + x
      y = mpmd.named_computation(lambda y: y + 1, name='stage2')(y)

      return x / y

    def g(x, y):
      return mpmd.call(f)(x, y)

    with self.assertRaisesRegex(
        RuntimeError,
        'Mesh assignment is not possible for op as its operands are on'
        ' conflicting meshes',
    ):
      mpmd.jit(g, mpmd_config).lower(
          np.ones((3, 5), dtype=jnp.float32), np.ones((3, 5), dtype=jnp.float32)
      )

  # TODO: b/396601755 - Move these errors to a separate error-handling test
  # file.
  def test_mesh_inference_fails_because_caller_operands_from_different_meshes(
      self,
  ):
    mpmd_config = get_2_stage_mpmd_config()

    def f(x, y):

      def g(x):
        return x + 1

      mpmd_call = mpmd.call(g)

      # The two stages are on different meshes, so the mpmd.call will have
      # operands on different meshes.
      x = mpmd.named_computation(lambda x: x + 1, name='stage1')(x)
      y = mpmd.named_computation(lambda y: y + 1, name='stage2')(y)

      return mpmd_call(x) + mpmd_call(y)

    with self.assertRaisesRegex(
        RuntimeError,
        'Mesh assignment is not possible for arg0 of mpmd.call "g"',
    ):
      mpmd.jit(f, mpmd_config).lower(
          np.ones((3, 5), dtype=jnp.float32), np.ones((3, 5), dtype=jnp.float32)
      )

  def test_partitioning_options_are_passed_through_ok_test(self):
    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        {'stage1': 'mesh1', 'stage2': 'mesh2'},
        partitioning_options=mpmd.PartitioningOptions(
            mpmd_infer_transfers=True,
        ),
    )

    def f(x):
      x = mpmd.named_computation(lambda x: x + 1, name='stage1')(x)
      x = x + 1
      y = mpmd.named_computation(lambda x: x + 1, name='stage2')(x)
      return x + y

    lowered = mpmd.jit(f, mpmd_config).lower(np.ones((3, 5), dtype=jnp.float32))

    self.assertEqual(lowered.function_mesh_assignment.input_meshes, ('mesh1',))
    self.assertEqual(lowered.function_mesh_assignment.output_meshes, 'mesh1')

    # See explanation in test_create_transfers_during_inference_does_not_fail.
    self.assertEqual(lowered.as_text('mpmd').count('mpmd.transfer'), 2)

  def test_partitioning_options_are_passed_through_error_test(
      self,
  ):
    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        {'stage1': 'mesh1', 'stage2': 'mesh2'},
        partitioning_options=mpmd.PartitioningOptions(),
    )

    def f(x):
      x = mpmd.named_computation(lambda x: x + 1, name='stage1')(x)
      x = x + 1
      y = mpmd.named_computation(lambda x: x + 1, name='stage2')(x)
      return x + y

    try:
      mpmd.jit(f, mpmd_config).lower(np.ones((3, 5), dtype=jnp.float32))
    except RuntimeError:
      pass

  def test_create_transfers_during_inference_does_not_fail(self):
    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        {'stage1': 'mesh1', 'stage2': 'mesh2'},
        partitioning_options=mpmd.PartitioningOptions(
            mpmd_infer_transfers=True,
        ),
    )

    def f(x):
      x = mpmd.named_computation(lambda x: x + 1, name='stage1')(x)
      x = x + 1
      x = x * 2
      y = mpmd.named_computation(lambda x: x + 1, name='stage2')(x)
      y = y + 1

      return x + y

    lowered = mpmd.jit(f, mpmd_config).lower(np.ones((3, 5), dtype=jnp.float32))

    self.assertEqual(lowered.function_mesh_assignment.input_meshes, ('mesh1',))
    self.assertEqual(lowered.function_mesh_assignment.output_meshes, 'mesh1')

    # We expect two transfers: from mesh1 to mesh2 and then back to mesh1.
    # This is because the return value has empty src_set, and so defaults to
    # mesh1. But computation of y happens on mesh2, so we transfer to mesh2,
    # and then transfer back to mesh1. Observe that this is not optimal, but
    # easily fixed by adding output assignments.
    # Note: this behaviour is subject to change in the future, as we improve the
    # transfer creation logic.
    self.assertEqual(lowered.as_text('mpmd').count('mpmd.transfer'), 2)

  def test_fan_out_grad(self):
    r"""Mesh is inferred for the meshless add introduced in a fan-out pattern.

    In such a model with a fan-out pattern, the backward pass will have an add
    op not in any named_computation even if the forward pass is entirely
    annotated with names.

    E.g., in

              /-> stage2 -\
     stage1 -|             |--> stage4
              \-> stage3 -/

    we have a fan-out from stage1 to stage2 and stage3. And on the backward
    pass, an add will be introduced:
    `add(stage2_backward_result, stage3_backward_result)`
    and this add doesn't live in any of the stages.

    This test verifies that mesh inference succeeds, by checking that
    mpmd.jit succeeds
    """

    def model(w1, w2, w3, x):
      stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
      stage2 = mpmd.named_computation(lambda a: a @ w2, name='stage2')(stage1)
      stage3 = mpmd.named_computation(lambda a: a @ w3, name='stage3')(stage1)
      stage4 = mpmd.named_computation(
          lambda a, b: jnp.sum(a) + jnp.sum(b), name='stage4'
      )(stage2, stage3)
      return stage4

    result_and_param_grads = jax.value_and_grad(model, argnums=[0, 1, 2])
    topology = test_utils.get_four_mesh_topology()
    assignment = {
        'stage1': 'mesh1',
        'stage2': 'mesh2',
        'stage3': 'mesh3',
        'stage4': 'mesh4',
    }

    lowered = mpmd.jit(
        result_and_param_grads,
        mpmd_types.make_config(topology, assignment),
    ).lower(
        np.ones((3, 5), dtype=jnp.float32),
        np.ones((5, 7), dtype=jnp.float32),
        np.ones((5, 9), dtype=jnp.float32),
        np.ones((10, 3), dtype=jnp.float32),
    )
    input_specs_meshes = lowered.function_named_shardings.input_specs
    output_specs_meshes = lowered.function_named_shardings.output_specs
    self.assertEqual(
        input_specs_meshes,
        (
            _replicated(topology['mesh1']),
            _replicated(topology['mesh2']),
            _replicated(topology['mesh3']),
            _replicated(topology['mesh1']),
        ),
    )
    self.assertEqual(
        output_specs_meshes,
        (
            _replicated(topology['mesh4']),
            (
                _replicated(topology['mesh1']),
                _replicated(topology['mesh2']),
                _replicated(topology['mesh3']),
            ),
        ),
    )

  def test_input_mesh_assignment_input_on_different_meshes(self):
    def mpmd_fn(x, y, z):
      return x, y, z

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        input_mesh_assignment=(None, 'mesh1', 'mesh2'),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(1, 2, 3)
    actual_input_meshes = lowered.function_mesh_assignment.input_meshes

    self.assertEqual(actual_input_meshes[1], 'mesh1')
    self.assertEqual(actual_input_meshes[2], 'mesh2')

  def test_input_mesh_assignment_all_input_assigned_to_same_mesh(
      self,
  ):
    def mpmd_fn(x, y, z):
      return x, y, z

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        input_mesh_assignment=('mesh1', 'mesh1', 'mesh1'),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(1, 2, 3)

    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        ('mesh1', 'mesh1', 'mesh1'),
    )

  def test_input_mesh_assignment_as_pytree(
      self,
  ):
    def mpmd_fn(x, y):
      return x, y

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        input_mesh_assignment=('mesh1', 'mesh2'),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
        1,
        {'a': 2, 'b': 3},
    )

    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        ('mesh1', {'a': 'mesh2', 'b': 'mesh2'}),
    )

  def test_input_mesh_assignment_as_pytree_prefix(
      self,
  ):
    def mpmd_fn(x, y):
      return x, y

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        input_mesh_assignment=('mesh1', {'a': 'mesh2'}),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
        np.ones((3, 5), dtype=jnp.float32),
        {'a': {'c': 1, 'd': 2}},
    )
    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        ('mesh1', {'a': {'c': 'mesh2', 'd': 'mesh2'}}),
    )

  def test_input_mesh_assignment_from_in_shardings_as_pytree_prefix(
      self,
  ):
    def mpmd_fn(x, y):
      return x, y

    topology = test_utils.get_two_mesh_topology()

    mpmd_config = mpmd_types.make_config(
        topology,
        name_to_mesh_assignment={},
        read_input_output_mesh_from_shardings=True,
    )
    in_shardings = (
        _replicated(mpmd_config.sharding_mesh),
        {'a': _replicated(topology['mesh2'])},
    )
    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn, mpmd_config, in_shardings=in_shardings
    ).lower(
        np.ones((3, 5), dtype=jnp.float32),
        {'a': {'c': 1, 'd': 2}},
    )
    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        (
            'mesh1',  # Defaults to mesh1, since it uses abstract mesh.
            {'a': {'c': 'mesh2', 'd': 'mesh2'}},
        ),
    )

  def test_input_mesh_assignment_does_not_block_transfers(self):

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        input_mesh_assignment=('mesh1',),
        output_mesh_assignment=('mesh1', 'mesh2'),
    )
    lowered: stages.MpmdLowered = mpmd.jit(lambda x: (x, x), mpmd_config).lower(
        1
    )

    self.assertEqual(lowered.function_mesh_assignment.input_meshes, ('mesh1',))
    self.assertEqual(
        lowered.function_mesh_assignment.output_meshes, ('mesh1', 'mesh2')
    )
    self.assertEqual(lowered.as_text('mpmd').count('transfer'), 1)

  def test_removed_input_assigned_to_placeholder_mesh_when_no_user_assignment(
      self,
  ):
    def mpmd_fn(x, y, z):
      del z
      return x, y

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        input_mesh_assignment=('mesh1', 'mesh1', None),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(1, 2, 3)
    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        ('mesh1', 'mesh1', 'mesh1'),
    )

  def test_removed_input_assigned_to_user_given_mesh(self):
    x = np.ones((8, 8))
    xs = (x, (x, x))
    lowered = mpmd.jit(
        lambda x, y: y[0],
        mpmd_types.make_config(
            test_utils.get_two_mesh_topology(),
            {},
            input_mesh_assignment=('mesh2', ('mesh1', 'mesh2')),
        ),
    ).lower(*xs)
    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        ('mesh2', ('mesh1', 'mesh2')),
    )

  def test_unused_input_assigned_to_mesh_and_kind_in_input_specs(self):
    x = np.ones((8, 8))
    xs = (x, (x, x))
    lowered = mpmd.jit(
        lambda x, y: y[0],
        mpmd_types.make_config(
            test_utils.get_two_mesh_topology(),
            {},
            input_mesh_assignment=(
                'mesh2#pinned_host',
                ('mesh1#device', 'mesh2#pinned_host'),
            ),
        ),
    ).lower(*xs)
    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        ('mesh2', ('mesh1', 'mesh2')),
    )

  def test_output_mesh_assignment_output_on_different_meshes(self):
    def mpmd_fn(x, y, z):
      return x, y, z

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        output_mesh_assignment=(None, 'mesh1', 'mesh2'),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(1, 2, 3)
    actual_output_meshes = lowered.function_mesh_assignment.output_meshes

    self.assertEqual(actual_output_meshes[1], 'mesh1')
    self.assertEqual(actual_output_meshes[2], 'mesh2')

  def test_output_mesh_assignment_all_output_assigned_to_same_mesh(
      self,
  ):
    def mpmd_fn(x, y, z):
      return x, y, z

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        output_mesh_assignment=('mesh1', 'mesh1', 'mesh1'),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(1, 2, 3)

    self.assertEqual(
        lowered.function_mesh_assignment.input_meshes,
        ('mesh1', 'mesh1', 'mesh1'),
    )

  def test_output_mesh_assignment_as_pytree(
      self,
  ):
    def mpmd_fn(x, y):
      return x, y

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        output_mesh_assignment=('mesh1', 'mesh2'),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
        1,
        {'a': 2, 'b': 3},
    )

    self.assertEqual(
        lowered.function_mesh_assignment.output_meshes,
        ('mesh1', {'a': 'mesh2', 'b': 'mesh2'}),
    )

  def test_output_mesh_assignment_as_pytree_prefix(
      self,
  ):
    def mpmd_fn(x, y):
      return x, y

    mpmd_config = mpmd_types.make_config(
        test_utils.get_two_mesh_topology(),
        name_to_mesh_assignment={},
        output_mesh_assignment=('mesh1', {'a': 'mesh2'}),
    )
    lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
        np.ones((3, 5), dtype=jnp.float32),
        {'a': {'c': 1, 'd': 2}},
    )
    self.assertEqual(
        lowered.function_mesh_assignment.output_meshes,
        ('mesh1', {'a': {'c': 'mesh2', 'd': 'mesh2'}}),
    )

  def test_output_mesh_assignment_from_out_shardings_as_pytree_prefix(
      self,
  ):
    def mpmd_fn(x, y):
      return x, y

    topology = test_utils.get_two_mesh_topology()
    mpmd_config = mpmd_types.make_config(
        topology,
        name_to_mesh_assignment={},
        read_input_output_mesh_from_shardings=True,
    )

    out_shardings = (
        _replicated(mpmd_config.sharding_mesh),
        {'a': _replicated(topology['mesh2'])},
    )

    lowered: stages.MpmdLowered = mpmd.jit(
        mpmd_fn, mpmd_config, out_shardings=out_shardings
    ).lower(
        np.ones((3, 5), dtype=jnp.float32),
        {'a': {'c': 1, 'd': 2}},
    )
    self.assertEqual(
        lowered.function_mesh_assignment.output_meshes,
        (
            'mesh1',  # Defaults to mesh1, since it uses abstract mesh.
            {'a': {'c': 'mesh2', 'd': 'mesh2'}},
        ),
    )

  def test_shared_param_update(self):
    """Mesh is inferred for the gradient accumulation for shared params.

    When a parameter is used in different meshes and is updated using the
    gradient, the gradient from different meshes will be accumulated outside of
    the meshes. E.g. the embedding vector used for encoding and
    decoding in the first and last layers will have the gradients from the
    encoding and decoding layer accumulated outside of both meshes.

    To make inference succeed, we assign the gradient to a mesh, and inference
    will then deduce that the gradient accumulation should live in this mesh.

    When the new inference pass is complete, this parameter will be
    transferred to both meshes (e.g. mesh1 and mesh3 in this case). A
    gradient will exist on both of these meshes, and the gradient accumulation
    (i.e. the add of the two gradients) will be meshless, but be pushed in to
    the consuming mesh (i.e. mesh1/stage1 on the backward pass).
    """

    def model(shared_param, w1, x):
      stage1 = mpmd.named_computation(
          lambda a: a @ shared_param, name='stage1'
      )(x)
      stage2 = mpmd.named_computation(lambda a: a @ w1, name='stage2')(stage1)
      stage3 = mpmd.named_computation(
          lambda a: a @ shared_param, name='stage3'
      )(stage2)
      stage4 = mpmd.named_computation(jnp.sum, name='stage4')(stage3)
      return stage4

    def param_update(shared_param, w1, x):
      value, grad = jax.value_and_grad(model)(shared_param, w1, x)
      # We manually assign the grad to the first mesh, to force the param update
      # there. The gradient accumulation over mesh1 and mesh3 is meshless, but
      # is now consumed by mesh1 because of the named_tensor, and so the add
      # will be moved into mesh1.
      grad = mpmd.named_tensor(grad, name='stage1')
      return value, shared_param + grad

    topology = test_utils.get_four_mesh_topology()
    assignment = {
        'stage1': 'mesh1',
        'stage2': 'mesh2',
        'stage3': 'mesh3',
        'stage4': 'mesh4',
    }

    lowered = mpmd.jit(
        param_update,
        mpmd_types.make_config(topology, assignment),
    ).lower(
        np.ones((5, 5), dtype=jnp.float32),
        np.ones((5, 5), dtype=jnp.float32),
        np.ones((5, 5), dtype=jnp.float32),
    )
    input_specs_meshes = lowered.function_named_shardings.input_specs
    output_specs_meshes = lowered.function_named_shardings.output_specs
    self.assertEqual(
        input_specs_meshes,
        (
            _replicated(topology['mesh1']),
            _replicated(topology['mesh2']),
            _replicated(topology['mesh1']),
        ),
    )
    self.assertEqual(
        output_specs_meshes,
        (
            _replicated(topology['mesh4']),
            _replicated(topology['mesh1']),
        ),
    )

  def test_output_on_different_meshes(self):
    def fn():
      x = mpmd.named_tensor(np.ones((3, 5), dtype=np.float32), name='name1')
      y = mpmd.named_tensor(np.full((5, 7), 2, dtype=np.float32), name='name2')
      return x, y

    topology = test_utils.get_two_mesh_topology()
    assignment = {'name1': 'mesh1', 'name2': 'mesh2'}
    output_specs = (
        mpmd.jit(fn, mpmd_types.make_config(topology, assignment))
        .lower()
        .function_named_shardings.output_specs
    )

    self.assertEqual(
        output_specs,
        (
            _replicated(topology['mesh1']),
            _replicated(topology['mesh2']),
        ),
    )

  @parameterized.parameters(
      # Below, the ..1 represents the transpose of the stage.
      dict(
          num_microbatches=1,
          expected_stage_order=[
              *('stage1', 'stage2', 'stage3', 'stage4'),
              *('stage4..1', 'stage3..1', 'stage2..1', 'stage1..1'),
          ],
      ),
      dict(
          num_microbatches=2,
          expected_stage_order=[
              *('stage1', 'stage2', 'stage1', 'stage2'),
              *('stage3', 'stage4', 'stage3', 'stage4'),
              *('stage4..1', 'stage3..1', 'stage4..1', 'stage3..1'),
              *('stage2..1', 'stage1..1', 'stage2..1', 'stage1..1'),
          ],
      ),
      dict(
          num_microbatches=4,
          expected_stage_order=[
              *('stage1', 'stage2', 'stage1', 'stage2'),
              *('stage3', 'stage4', 'stage3', 'stage4'),
              *('stage1', 'stage2', 'stage1', 'stage2'),
              *('stage3', 'stage4', 'stage3', 'stage4'),
              *('stage4..1', 'stage3..1', 'stage4..1', 'stage3..1'),
              *('stage2..1', 'stage1..1', 'stage2..1', 'stage1..1'),
              *('stage4..1', 'stage3..1', 'stage4..1', 'stage3..1'),
              *('stage2..1', 'stage1..1', 'stage2..1', 'stage1..1'),
          ],
      ),
      dict(
          num_microbatches=1,
          compute_transpose=False,
          expected_stage_order=[
              *('stage1', 'stage2', 'stage3', 'stage4'),
          ],
      ),
      dict(
          num_microbatches=2,
          compute_transpose=False,
          expected_stage_order=[
              *('stage1', 'stage2', 'stage1', 'stage2'),
              *('stage3', 'stage4', 'stage3', 'stage4'),
          ],
      ),
      dict(
          num_microbatches=4,
          compute_transpose=False,
          expected_stage_order=[
              *('stage1', 'stage2', 'stage1', 'stage2'),
              *('stage3', 'stage4', 'stage3', 'stage4'),
              *('stage1', 'stage2', 'stage1', 'stage2'),
              *('stage3', 'stage4', 'stage3', 'stage4'),
          ],
      ),
  )
  def test_circular_pipeline(
      self,
      num_microbatches: int,
      compute_transpose: bool = True,
      expected_stage_order: Sequence[str] = (),
  ):
    def fwd(w, x):
      x = mpmd.named_computation(lambda a, b: a @ b, name='stage1')(w, x)
      x = mpmd.named_computation(lambda a: a @ a, name='stage2')(x)
      x = mpmd.named_computation(lambda a: a @ a, name='stage3')(x)
      x = mpmd.named_computation(lambda a: a @ a, name='stage4')(x)
      return x

    def loss_fn(w, xs):
      y = 0
      for i in range(num_microbatches):
        y += mpmd.call(fwd, call_counter=i)(w, xs[0])
      return jnp.sum(y * y)

    def train(w, xs):
      if compute_transpose:
        loss, w_grad = jax.value_and_grad(loss_fn)(w, xs)
      else:
        loss = loss_fn(w, xs)
        w_grad = None
      return loss, w_grad

    topology = test_utils.get_two_mesh_topology()
    mpmd_config = mpmd_types.make_config(
        topology,
        name_to_mesh_assignment={
            'stage1': 'mesh1',
            'stage2': 'mesh2',
            'stage3': 'mesh1',
            'stage4': 'mesh2',
        },
        name_to_stage_assignment={
            'stage1': 1,
            'stage2': 2,
            'stage3': 3,
            'stage4': 4,
        },
        partitioning_options=mpmd_types.PartitioningOptions(
            mpmd_pipeline_schedule='CircularWithReversedBackward'
        ),
    )

    w = jnp.ones((10, 10))
    xs = [jnp.ones((10, 10))] * 4
    l = mpmd.jit(train, mpmd_config).lower(w, xs)

    # Just get the fragment_call data, skip function name which always starts
    # with "@".
    func_calls = [
        line.split('@')[0]
        for line in l.as_text('mpmd').splitlines()
        if 'mpmd.fragment_call' in line
    ]

    for i, (func_call, expected_stage) in enumerate(
        zip(func_calls, expected_stage_order)
    ):
      self.assertRegex(
          func_call,
          expected_stage,
          f'Call {i} mismatched in\n'
          f'{"\n".join(f"Call {i}: {f}" for i, f in enumerate(func_calls))}',
      )


if __name__ == '__main__':
  absltest.main()
