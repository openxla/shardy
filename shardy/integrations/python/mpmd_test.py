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
import unittest

from mlir import ir
from mlir.dialects import func
from openxla import sdy

mpmd = sdy.mpmd


class MpmdTest(unittest.TestCase):

  def test_user_origin_attr(self):
    with ir.Context() as ctx:
      mpmd.register_dialect(ctx)
      attr = mpmd.UserOriginAttr.get('test_user', 2)
      self.assertEqual(attr.user_name, 'test_user')
      self.assertEqual(attr.transpose_count, 2)
      self.assertEqual(str(attr), '#mpmd.user_origin<"test_user"(2)>')

  def test_named_computation_op(self):
    """Tests the `mpmd.named_computation` op in an MLIR function."""
    with ir.Context() as ctx, ir.Location.unknown() as loc:
      mpmd.register_dialect(ctx)
      # TODO: b/356626043 - Make the test work without this line.
      ctx.allow_unregistered_dialects = True
      module = ir.Module.create()
      f32_tensor_type = ir.RankedTensorType.get((4,), ir.F32Type.get())
      origin = mpmd.UserOriginAttr.get('f', 0, ctx)
      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(f32_tensor_type, results=[f32_tensor_type])
        def f(tensor):
          frag = mpmd.NamedComputationOp(
              [f32_tensor_type], [tensor], origin, loc=loc
          )
          block = frag.region.blocks.append(f32_tensor_type)
          with ir.InsertionPoint(block):
            mpmd.ReturnOp([block.arguments[0]])
          func.ReturnOp(frag.results_)

    self.assertRegex(
        str(module),
        r'mpmd.named_computation<"f">',
    )

  def test_named_tensor_op(self):
    """Tests the `mpmd.named_tensor` op being used in an MLIR func."""
    with ir.Context() as ctx, ir.Location.unknown() as loc:
      mpmd.register_dialect(ctx)
      # TODO: b/356626043 - Make the test work without this line.
      ctx.allow_unregistered_dialects = True
      module = ir.Module.create()
      f32_tensor_type = ir.RankedTensorType.get((4,), ir.F32Type.get())
      name = ir.StringAttr(ir.Attribute.parse('"some_name"'))
      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(f32_tensor_type, results=[f32_tensor_type])
        def f(tensor):
          tagged = mpmd.NamedTensorOp(tensor, name, loc=loc)
          func.ReturnOp([tagged.result])

    self.assertRegex(str(module), r'mpmd.named_tensor .* name="some_name"')


if __name__ == '__main__':
  unittest.main()
