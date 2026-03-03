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

"""Test that jax.named_scope location info is preserved in named_computation."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from shardy.integrations.python.jax.mpmd import ops


class NamedScopeLocationTest(absltest.TestCase):

  def test_named_scope_inside_named_computation(self):
    @ops.named_computation(name='my_comp')
    def f(x):
      with jax.named_scope('my_scope'):
        return x + x

    lowered = jax.jit(f).lower(np.ones((3, 5), dtype=jnp.float32))
    mlir_text = lowered.as_text(debug_info=True)

    self.assertIn('my_comp/my_scope/add', mlir_text)


if __name__ == '__main__':
  absltest.main()
