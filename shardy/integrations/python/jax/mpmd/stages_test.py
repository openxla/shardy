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
from unittest import mock

from absl.testing import absltest

from shardy.integrations.python.jax.mpmd import stages


class MpmdExecutableTest(absltest.TestCase):

  def test_input_shardings(self):
    in_shardings = {"a": (0, 1), "b": (2, 3, 4)}
    executable = stages.MpmdExecutable(
        executable=mock.MagicMock(),
        module_ir=mock.MagicMock(),
        func_name="mock_exexutable",
        flat_in_avals=[mock.MagicMock()] * 5,
        in_shardings=in_shardings,
        flat_out_shardings=[],
        out_avals=[],
        kept_inputs_indices=set(),
        donated_inputs_indices=set(),
        topology={},
    )

    self.assertEqual(executable.input_shardings_tree, in_shardings)
    self.assertEqual(executable.input_shardings(), [0, 1, 2, 3, 4])


if __name__ == "__main__":
  absltest.main()
