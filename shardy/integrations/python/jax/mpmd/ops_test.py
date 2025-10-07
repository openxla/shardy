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

"""Tests that we registered the right rules for mpmd op primitives."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from shardy.integrations.python.jax.mpmd import ops


class NamedComputationTest(absltest.TestCase):

  def test_named_computation_abstract_eval(self):
    """Checks the abstract evaluation of the named_computation op."""

    def f(a):
      return ops.named_computation(jnp.transpose, name='f')(a)

    self.assertEqual(
        jax.eval_shape(f, np.ones((3, 5), dtype=jnp.float32)),
        jax.ShapeDtypeStruct((5, 3), jnp.float32),
    )

  def test_named_computation_eager_execution(self):
    """Checks the eager execution of the named_computation op."""

    def f(a):
      return ops.named_computation(lambda x: x, name='f')(a)

    ones = np.ones((3, 5), dtype=jnp.int32)
    np.testing.assert_equal(f(ones), ones)

  def test_named_computation_static_argnames_on_kwargs(self):
    x = jnp.ones(10)

    @ops.named_computation(name='x')
    def non_static_f(x, *, config):
      return x * config

    non_static_l = jax.jit(non_static_f, static_argnames='config').lower(
        x, config=10
    )

    # named_comp has two args.
    self.assertRegex(non_static_l.as_text(), r'func.*named_computation.*arg0')
    self.assertRegex(non_static_l.as_text(), r'func.*named_computation.*arg1')

    @ops.named_computation(name='x', static_argnames='config')
    def static_f(x, *, config):
      return x * config

    static_l = jax.jit(static_f, static_argnames='config').lower(x, config=10)

    # named_comp has only one arg, because second arg is static
    self.assertRegex(static_l.as_text(), r'func.*named_computation.*arg0')
    self.assertNotRegex(static_l.as_text(), r'func.*named_computation.*arg1')

  def test_named_computation_static_argnames_on_args(self):
    x = jnp.ones(10)

    @ops.named_computation(name='x')
    def non_static_f(x, config):
      return x * config

    non_static_l = jax.jit(non_static_f, static_argnames='config').lower(x, 10)

    # named_comp has two args.
    self.assertRegex(non_static_l.as_text(), r'func.*named_computation.*arg0')
    self.assertRegex(non_static_l.as_text(), r'func.*named_computation.*arg1')

    @ops.named_computation(name='x', static_argnames='config')
    def static_f(x, config):
      return x * config

    static_l = jax.jit(static_f, static_argnames='config').lower(x, 10)

    print(static_l.as_text())
    # named_comp has only one arg, because second arg is static
    self.assertRegex(static_l.as_text(), r'func.*named_computation.*arg0')
    self.assertNotRegex(static_l.as_text(), r'func.*named_computation.*arg1')


class NamedTensorTest(absltest.TestCase):

  def test_named_tensor_eager_execution(self):
    """Checks the eager execution of the named_tensor op."""

    def f(a):
      return ops.named_tensor(a, name='a_tensor')

    np.testing.assert_equal(
        f(np.ones((3, 5), dtype=jnp.float32)),
        np.ones((3, 5), dtype=jnp.float32),
    )

  def test_named_tensor_abstract_eval(self):
    """Checks the abstract evaluation of the named_tensor op."""

    def f(a):
      return ops.named_tensor(a, name='a_tensor')

    self.assertEqual(
        jax.eval_shape(f, np.ones((3, 5), dtype=jnp.float32)),
        jax.ShapeDtypeStruct((3, 5), jnp.float32),
    )

  def test_named_tensor_jax_jit_can_execute(self):
    fn = lambda x: ops.named_tensor(x, name='a_tensor')
    t = np.zeros((10, 10), dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(fn)(t), t)

  def test_gradient_of_named_tensor(self):
    t = np.zeros((10, 10), dtype=jnp.float32)
    grad = jax.grad(lambda x: jnp.sum(ops.named_tensor(x, name='a_tensor')))
    np.testing.assert_allclose(grad(t), jax.grad(jnp.sum)(t))

  def test_vmap_of_named_tensor(self):
    computation = lambda x: ops.named_tensor(x, name='a_tensor')
    result = jax.jit(jax.vmap(computation, in_axes=0))(
        np.zeros((3, 5, 7), dtype=jnp.float32)
    )
    self.assertEqual(result.shape, (3, 5, 7))


class CallOpTest(absltest.TestCase):

  def test_negative_call_counter(self):
    with self.assertRaisesRegex(
        ValueError,
        'call_counter must be non-negative, got call_counter=-1 instead.',
    ):
      ops.call(lambda x: x, call_counter=-1)(1.0)

  def test_abstract_eval(self):
    """Checks the abstract evaluation of the parallel loop op."""

    def f(a, b):
      return ops.call(lambda x: (x[1], (x[0], x[1])))((a, b))

    shape_a = jax.ShapeDtypeStruct((6, 7), jnp.float32)
    shape_b = jax.ShapeDtypeStruct((6, 8), jnp.float32)
    out_shape = jax.eval_shape(f, shape_a, shape_b)
    self.assertEqual(out_shape, (shape_b, (shape_a, shape_b)))

  def test_eager_execution(self):
    """Checks the eager execution of the parallel loop op."""

    def f(a, b):
      return ops.call(lambda x: (x[1], (x[0], x[1])))((a, b))

    ones = np.ones((6, 7), dtype=jnp.float32)
    zeros = np.zeros((6, 8), dtype=jnp.float32)
    np.testing.assert_equal(f(ones, zeros), (zeros, (ones, zeros)))

  def test_jax_jit(self):
    def f(a, b):
      return ops.call(lambda x: (x[1], (x[0], x[1])))((a, b))

    ones = np.ones((6, 7), dtype=jnp.float32)
    zeros = np.zeros((6, 8), dtype=jnp.float32)
    np.testing.assert_equal(jax.jit(f)(ones, zeros), (zeros, (ones, zeros)))

  def test_grad1(self):
    def f(a, b):
      return jnp.sum(ops.call(lambda x, y: x + y)(a, b))

    ones = np.ones((2, 4), dtype=jnp.float32)
    zeros = np.zeros((2, 4), dtype=jnp.float32)
    grads = jax.grad(f)(ones, zeros)
    np.testing.assert_equal(grads, ones)

  def test_grad2(self):
    def f(a, b):
      return ops.call(lambda x, y: jnp.sum(x + y), call_counter=42)(a, b)

    arg0 = jax.random.uniform(jax.random.PRNGKey(0))
    arg1 = jax.random.uniform(jax.random.PRNGKey(1))
    grads = jax.grad(f)(arg0, arg1)
    self.assertEqual(grads, 1.0)

  def test_grad3(self):
    def f(ws, x):
      def loss(ws, x):
        x = x @ ws[0]
        x = x @ ws[1]
        return jnp.sum(x)

      return ops.call(loss, call_counter=42)(ws, x)

    w0 = jax.random.uniform(jax.random.PRNGKey(0), (2, 5))
    w1 = jax.random.uniform(jax.random.PRNGKey(1), (5, 2))
    x = jax.random.uniform(jax.random.PRNGKey(2), (2,))
    # Just testing that we do not get any crash below.
    grads = jax.grad(f)([w0, w1], x)
    # Also, check that this works under jit.
    jit_grads = jax.jit(jax.grad(f))([w0, w1], x)
    # As a rudimentary test, check that the grads are the same, jitted or not.
    np.testing.assert_allclose(grads[0], jit_grads[0])
    np.testing.assert_allclose(grads[1], jit_grads[1])

  def test_vmap(self):
    def f(a, b):
      return ops.call(lambda x, y: x + y)(a, b)

    ones = np.ones((2, 4), dtype=jnp.float32)
    twos = np.ones((2, 4), dtype=jnp.float32) * 2
    xs = jax.vmap(f)(ones, twos)
    ys = f(ones, twos)
    np.testing.assert_array_equal(xs, ys)


class BroadcastTest(absltest.TestCase):

  def test_eager_execution(self):
    """Checks the eager execution of the broadcast op."""
    np.testing.assert_equal(
        ops.broadcast(np.ones((3, 5), dtype=jnp.float32)),
        np.ones((3, 5), dtype=jnp.float32),
    )

  def test_abstract_eval(self):
    """Checks the abstract evaluation of the broadcast op."""
    self.assertEqual(
        jax.eval_shape(ops.broadcast, np.ones((3, 5), dtype=jnp.float32)),
        jax.ShapeDtypeStruct((3, 5), jnp.float32),
    )

  def test_jax_jit_can_execute(self):
    """Checks that jax.jit can execute the broadcast op."""
    t = np.zeros((10, 10), dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(ops.broadcast)(t), t)

  def test_vmap(self):
    result = jax.jit(jax.vmap(ops.broadcast, in_axes=0))(
        np.zeros((3, 5, 7), dtype=jnp.float32)
    )
    self.assertEqual(result.shape, (3, 5, 7))


class ReduceTest(absltest.TestCase):

  def test_eager_execution(self):
    """Checks the eager execution of the reduce op."""
    x = jnp.array([1, 2, 3])
    result = ops.reduce(x)
    np.testing.assert_allclose(result, x)

  def test_abstract_eval(self):
    """Checks the abstract evaluation of the reduce op."""
    x = jnp.array([1, 2, 3])
    self.assertEqual(
        jax.eval_shape(ops.reduce, x),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )

  def test_jax_jit_can_execute(self):
    """Checks that jax.jit can execute the reduce op."""
    x = jnp.array([1, 2, 3])
    np.testing.assert_allclose(jax.jit(ops.reduce)(x), x)

  def test_vmap(self):
    """Checks that vmap can execute the reduce op."""
    x = jnp.array([1, 2, 3])
    result = jax.jit(jax.vmap(ops.reduce, in_axes=0))(x)
    np.testing.assert_allclose(result.shape, (3,))


class ForiLoopTest(absltest.TestCase):

  def test_eager_execution(self):
    """Checks the eager execution of the fori_loop op."""
    result = ops.fori_loop(
        3,
        lambda _, x: x + np.ones((1, 1), dtype=np.int32),
        np.zeros((1, 1), dtype=np.int32),
    )
    self.assertEqual(result, np.array([[3]], dtype=np.int32))

  def test_abstract_eval(self):
    """Checks the abstract evaluation of the fori_loop op."""

    def _f():
      return ops.fori_loop(
          3, lambda _, x: x + 1, np.zeros((1, 1), dtype=np.int32)
      )

    self.assertEqual(
        jax.eval_shape(_f), jax.ShapeDtypeStruct((1, 1), jnp.int32)
    )

  def test_jax_jit_can_execute(self):
    """Checks that jax.jit can execute the fori_loop op."""

    @jax.jit
    def _f():
      return ops.fori_loop(
          3, lambda _, x: x + 1, np.zeros((1, 1), dtype=np.int32)
      )

    self.assertEqual(_f(), np.array([[3]], dtype=np.int32))

  def test_input_output_type_mismatch(self):
    with self.assertRaisesRegex(
        ValueError, 'Input and output avals must match'
    ):
      ops.fori_loop(2, lambda i, x: i, np.zeros((1, 1), dtype=np.float32))

  def test_input_output_struct_mismatch(self):
    with self.assertRaisesRegex(
        ValueError,
        'Tree structs of init_val and the result of the fori_loop must match',
    ):
      ops.fori_loop(
          2,
          lambda i, x: (x, x),
          np.zeros((1, 1), dtype=np.float32),
      )


class EnumeratedForLoopTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_basic_input_tree(self, unroll: bool):
    def body(carry, _, mb):
      return carry + mb

    def f(init):
      inputs = [1.0, 2.0, 3.0]
      return ops.enumerated_for(body, init, inputs, unroll=unroll)

    with self.subTest('eager'):
      self.assertEqual(f(0.0), 6.0)

    with self.subTest('jax.jit'):
      self.assertEqual(jax.jit(f)(0.0), 6.0)

  @parameterized.parameters(True, False)
  def test_complex_input_tree_with_update_of_carry_dependent_on_index(
      self, unroll: bool
  ):
    def body(carry, i, mb):
      return {'c': carry['c'].at[i].set(mb['a'] + mb['b'])}

    def f(init, inputs):
      return ops.enumerated_for(body, init, inputs, unroll=unroll)

    carry = {'c': np.array([0.0, 0.0, 0.0])}
    inputs = {
        'a': [1.0, 2.0, 3.0],
        'b': [4.0, 5.0, 6.0],
    }
    np.testing.assert_equal(f(carry, inputs), {'c': np.array([5.0, 7.0, 9.0])})

  @parameterized.parameters(True, False)
  def test_not_all_tree_leaves_are_lists_but_are_sequences(self, unroll: bool):
    def body(carry, i, mb):
      return carry.at[i].set(mb['a'] + mb['b'])

    def f(init, inputs):
      return ops.enumerated_for(body, init, inputs, unroll=unroll)

    carry = np.array([0.0, 0.0, 0.0])
    inputs = {
        'a': (1.0, 2.0, 3.0),
        'b': [4.0, 5.0, 6.0],
    }
    # Using jit to check if our validation is good enough while tracing, etc.
    np.testing.assert_equal(
        jax.jit(f)(carry, inputs), np.array([5.0, 7.0, 9.0])
    )

  @parameterized.parameters(True, False)
  def test_nested_leaves(self, unroll: bool):
    def body(carry, i, mb):
      return carry.at[i].set(mb[0] + mb[1])

    def f(init, inputs):
      return ops.enumerated_for(body, init, inputs, unroll=unroll)

    carry = np.array([0.0, 0.0, 0.0])
    inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    np.testing.assert_equal(f(carry, inputs), np.array([5.0, 7.0, 9.0]))

  @parameterized.parameters(True, False)
  def test_leaf_is_not_a_sequence(self, unroll: bool):
    def f(init):
      inputs = 1.0
      return ops.enumerated_for(
          lambda c, _, x: c + x, init, inputs, unroll=unroll
      )

    with self.assertRaisesRegex(
        ValueError,
        'Expected all leaves of the `inputs` tree to be Sequences. Got leaf at'
        r' path \(\) of type .*float.*',
    ):
      f(0.0)

  @parameterized.parameters(True, False)
  def test_mismatch_in_length(self, unroll: bool):
    def body(carry, i, mb):
      return {'c': carry['c'].at[i].set(mb['a'] + mb['b'])}

    def f(init, inputs):
      return ops.enumerated_for(body, init, inputs, unroll=unroll)

    carry = {'c': np.array([0.0, 0.0, 0.0])}
    inputs = {
        'a': [1.0, 2.0, 3.0],
        'b': [4.0, 5.0, 6.0, 7.0],
    }
    with self.assertRaisesRegex(
        ValueError,
        'Expected all leaves of the `inputs` tree to be lists of the same'
        ' length. Got list lengths: {3, 4}',
    ):
      f(carry, inputs)

  @parameterized.parameters(True, False)
  def test_leaves_are_not_arrays(self, unroll: bool):
    def f(init, inputs):
      return ops.enumerated_for(
          lambda c, _, __: c, init, inputs, unroll=unroll
      )

    with self.assertRaisesRegex(
        ValueError,
        'Expected all leaves of the `inputs` tree to be sequences of arrays.'
        ' Got leaf at path',
    ):
      f(0.0, ['a'])

  @parameterized.parameters(True, False)
  def test_lowering_with_shapes(self, unroll: bool):
    def f(init, inputs):
      return ops.enumerated_for(
          lambda carry, __, mb: carry + mb[0] + mb[1],
          init,
          inputs,
          unroll=unroll,
      )

    carry = jax.ShapeDtypeStruct((1,), jnp.float32)
    list_of_shapes = [
        jax.ShapeDtypeStruct((1,), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.float32),
    ]
    inputs = [list_of_shapes, list_of_shapes]
    # We expect two loop iterations and therefore to call the body of the loop
    # function twice.
    self.assertEqual(
        str(jax.jit(f).lower(carry, inputs).compiler_ir()).count(
            'call @"shardy_mpmd<lambda>"'
        ),
        2,
    )

if __name__ == '__main__':
  absltest.main()
