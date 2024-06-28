# Copyright 2024 The Shardy Authors.
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
"""Tests for Shardy Attributes and Operations."""

import unittest

print("BEFOREIMPORT"*100)

from mlir import ir
from openxla import sdy


class SdyAttributeTest(unittest.TestCase):
  """Tests for Shardy attributes."""

  # TODO(b/339634165): define tests for ShardingRuleAttr.

  def run(self, result=None):
    with ir.Context() as ctx:
      print("HELLO"*100)
      sdy.ir.register_dialect(ctx)
      print("AFTERREGISTER"*100)
      super().run(result)

  def test_mesh_axis_attr(self):
    axis = sdy.ir.MeshAxisAttr.get('data', 42)
    self.assertEqual(str(axis), '#sdy<mesh_axis"data"=42>')
    self.assertEqual(axis.name, 'data')
    self.assertEqual(axis.size, 42)

  def test_mesh_attr(self):
    axes = [
        sdy.ir.MeshAxisAttr.get('data', 42),
        sdy.ir.MeshAxisAttr.get('model', 123),
    ]
    mesh = sdy.ir.MeshAttr.get(axes)
    self.assertEqual(str(mesh), '#sdy.mesh<"data"=42, "model"=123>')
    self.assertEqual(mesh.axes, axes)

  def test_sub_axis_info_attr(self):
    info = sdy.ir.SubAxisInfoAttr.get(1, 2)
    self.assertEqual(str(info), '#sdy<sub_axis_info(1)2>')
    self.assertEqual(info.pre_size, 1)
    self.assertEqual(info.size, 2)

  def test_axis_ref_attr(self):
    axis_ref = sdy.ir.AxisRefAttr.get('data')
    self.assertEqual(str(axis_ref), '#sdy<axis_ref"data">')
    self.assertEqual(axis_ref.name, 'data')
    self.assertIsNone(axis_ref.sub_axis_info)

  def test_axis_ref_attr_with_sub_axis_info(self):
    info = sdy.ir.SubAxisInfoAttr.get(4, 2)
    axis_ref = sdy.ir.AxisRefAttr.get('data', info)
    self.assertEqual(str(axis_ref), '#sdy<axis_ref"data":(4)2>')
    self.assertEqual(axis_ref.name, 'data')
    self.assertEqual(axis_ref.sub_axis_info, info)

  def test_dimension_sharding_attr_open_p0(self):
    axes = [
        sdy.ir.AxisRefAttr.get('model'),
    ]
    dimension_sharding = sdy.ir.DimensionShardingAttr.get(
        axes, is_closed=False, priority=0
    )
    self.assertEqual(
        str(dimension_sharding), '#sdy<dimension_sharding{"model", ?}p0>'
    )
    self.assertEqual(dimension_sharding.axes, axes)
    self.assertFalse(dimension_sharding.is_closed)
    self.assertEqual(dimension_sharding.priority, 0)

  def test_dimension_sharding_attr_open_no_priority(self):
    axes = [
        sdy.ir.AxisRefAttr.get('model'),
    ]
    dimension_sharding = sdy.ir.DimensionShardingAttr.get(axes, is_closed=False)
    self.assertEqual(
        str(dimension_sharding), '#sdy<dimension_sharding{"model", ?}>'
    )
    self.assertEqual(dimension_sharding.axes, axes)
    self.assertFalse(dimension_sharding.is_closed)
    self.assertIsNone(dimension_sharding.priority)

  def test_dimension_sharding_attr_closed_p1(self):
    axes = [
        sdy.ir.AxisRefAttr.get('data', sdy.ir.SubAxisInfoAttr.get(4, 2)),
        sdy.ir.AxisRefAttr.get('model'),
    ]
    dimension_sharding = sdy.ir.DimensionShardingAttr.get(
        axes, is_closed=True, priority=1
    )
    self.assertEqual(
        str(dimension_sharding),
        '#sdy<dimension_sharding{"data":(4)2, "model"}p1>',
    )
    self.assertEqual(dimension_sharding.axes, axes)
    self.assertTrue(dimension_sharding.is_closed)
    self.assertEqual(dimension_sharding.priority, 1)

  def test_dimension_sharding_attr_empty_closed(self):
    dimension_sharding = sdy.ir.DimensionShardingAttr.get([], is_closed=True)
    self.assertEqual(
        str(dimension_sharding),
        '#sdy<dimension_sharding{}>',
    )
    self.assertEqual(dimension_sharding.axes, [])
    self.assertTrue(dimension_sharding.is_closed)
    self.assertIsNone(dimension_sharding.priority)

  def test_dimension_sharding_attr_empty_open_p2(self):
    dimension_sharding = sdy.ir.DimensionShardingAttr.get(
        [], is_closed=False, priority=2
    )
    self.assertEqual(
        str(dimension_sharding),
        '#sdy<dimension_sharding{?}p2>',
    )
    self.assertEqual(dimension_sharding.axes, [])
    self.assertFalse(dimension_sharding.is_closed)
    self.assertEqual(dimension_sharding.priority, 2)

  def test_tensor_sharding_attr_only_dim_shardings(self):
    dimension_shardings = [
        sdy.ir.DimensionShardingAttr.get(
            [sdy.ir.AxisRefAttr.get('data', sdy.ir.SubAxisInfoAttr.get(4, 2))],
            is_closed=True,
            priority=1,
        ),
        sdy.ir.DimensionShardingAttr.get(
            [sdy.ir.AxisRefAttr.get('model')],
            is_closed=False,
            priority=0,
        ),
    ]
    tensor_sharding = sdy.ir.TensorShardingAttr.get(
        'my_mesh', dimension_shardings, replicated_axes=[]
    )
    self.assertEqual(
        str(tensor_sharding),
        '#sdy.sharding<@my_mesh, [{"data":(4)2}p1, {"model", ?}p0]>',
    )
    self.assertEqual(tensor_sharding.mesh_name, 'my_mesh')
    self.assertEqual(tensor_sharding.dimension_shardings, dimension_shardings)
    self.assertEqual(tensor_sharding.replicated_axes, [])

  def test_tensor_sharding_attr_only_replicated_axes(self):
    replicated_axes = [
        sdy.ir.AxisRefAttr.get('data', sdy.ir.SubAxisInfoAttr.get(4, 2)),
        sdy.ir.AxisRefAttr.get('model'),
    ]
    tensor_sharding = sdy.ir.TensorShardingAttr.get(
        'my_mesh', dimension_shardings=[], replicated_axes=replicated_axes
    )
    self.assertEqual(
        str(tensor_sharding),
        '#sdy.sharding<@my_mesh, [], replicated={"data":(4)2, "model"}>',
    )
    self.assertEqual(tensor_sharding.mesh_name, 'my_mesh')
    self.assertEqual(tensor_sharding.dimension_shardings, [])
    self.assertEqual(tensor_sharding.replicated_axes, replicated_axes)

  def test_tensor_sharding_attr_dim_sharding_and_replicated(self):
    dimension_shardings = [
        sdy.ir.DimensionShardingAttr.get([], is_closed=True),
        sdy.ir.DimensionShardingAttr.get([], is_closed=False),
        sdy.ir.DimensionShardingAttr.get(
            [sdy.ir.AxisRefAttr.get('model')], is_closed=True
        ),
    ]
    replicated_axes = [
        sdy.ir.AxisRefAttr.get('data', sdy.ir.SubAxisInfoAttr.get(4, 2)),
    ]
    tensor_sharding = sdy.ir.TensorShardingAttr.get(
        'my_mesh',
        dimension_shardings,
        replicated_axes,
    )
    self.assertEqual(
        str(tensor_sharding),
        '#sdy.sharding<@my_mesh, [{}, {?}, {"model"}],'
        ' replicated={"data":(4)2}>',
    )
    self.assertEqual(tensor_sharding.mesh_name, 'my_mesh')
    self.assertEqual(tensor_sharding.dimension_shardings, dimension_shardings)
    self.assertEqual(tensor_sharding.replicated_axes, replicated_axes)

  def test_tensor_sharding_per_value(self):
    tensor_sharding_1 = sdy.ir.TensorShardingAttr.get(
        'my_mesh',
        dimension_shardings=[
            sdy.ir.DimensionShardingAttr.get([], is_closed=True),
        ],
        replicated_axes=[
            sdy.ir.AxisRefAttr.get('data', sdy.ir.SubAxisInfoAttr.get(4, 2)),
        ],
    )
    tensor_sharding_2 = sdy.ir.TensorShardingAttr.get(
        'my_mesh',
        dimension_shardings=[
            sdy.ir.DimensionShardingAttr.get(
                [sdy.ir.AxisRefAttr.get('model')], is_closed=True
            ),
        ],
        replicated_axes=[],
    )
    tensor_sharding_per_value = sdy.ir.TensorShardingPerValueAttr.get(
        shardings=[tensor_sharding_1, tensor_sharding_2],
    )
    self.assertEqual(
        str(tensor_sharding_per_value),
        '#sdy.sharding_per_value<[<@my_mesh, [{}], replicated={"data":(4)2}>,'
        ' <@my_mesh, [{"model"}]>]>',
    )
    self.assertEqual(
        tensor_sharding_per_value.shardings,
        [tensor_sharding_1, tensor_sharding_2],
    )


class SdyOperationTest(unittest.TestCase):
  """Tests for Shardy operations."""

  # TODO(b/339634165): define tests for:
  # - ManualComputationOp
  # - ReshardOp
  # - ReturnOp
  # - ConstantOp
  # - IdentityOp

  def run(self, result=None):
    with ir.Context() as ctx, ir.Location.unknown(ctx):
      sdy.ir.register_dialect(ctx)
      module = ir.Module.create()
      with ir.InsertionPoint(module.body):
        super().run(result)

  def test_mesh(self):
    mesh_attr = sdy.ir.MeshAttr.get([
        sdy.ir.MeshAxisAttr.get('data', 42),
        sdy.ir.MeshAxisAttr.get('model', 123),
    ])
    mesh = sdy.ir.MeshOp('my_mesh', mesh_attr)
    self.assertEqual(str(mesh), 'sdy.mesh @my_mesh = <"data"=42, "model"=123>')
    self.assertEqual(mesh.sym_name.value, 'my_mesh')
    self.assertEqual(mesh.mesh, mesh_attr)

  def test_sharding_constraint(self):
    print("HERE"*100)
    custom_call = ir.Operation.create(
        'tensor.empty',
        results=[ir.RankedTensorType.get((16, 32), ir.F32Type.get())],
    )
    sharding = sdy.ir.TensorShardingAttr.get(
        'my_mesh',
        [
            sdy.ir.DimensionShardingAttr.get(
                [sdy.ir.AxisRefAttr.get('model')], is_closed=True
            ),
        ],
        [
            sdy.ir.AxisRefAttr.get('data', sdy.ir.SubAxisInfoAttr.get(4, 2)),
        ],
    )
    sharding_constraint = sdy.ir.ShardingConstraintOp(
        custom_call.result,
        sharding,
    )

    self.assertEqual(
        str(sharding_constraint),
        '%1 = "sdy.sharding_constraint"(%0) <{sharding ='
        ' #sdy.sharding<@my_mesh, [{"model"}], replicated={"data":(4)2}>}> :'
        ' (tensor<16x32xf32>) -> tensor<16x32xf32>',
    )
    self.assertEqual(sharding_constraint.input, custom_call.result)
    self.assertEqual(sharding_constraint.sharding, sharding)
    # Can't rely on Python to delete the ops, because Python may delete
    # `custom_call` first which means we'll get the error:
    # ```
    # error: 'stablehlo.custom_call' op operation destroyed but still has uses
    # ```
    # So need to delete sharding_constraint first then custom_call
    del sharding_constraint
    del custom_call

  def test_parsed_op_isinstance(self):
    # Make sure Python can figure out the type of the op if it's been parsed
    # from a string. This works due to registering openxla in the dialect path
    # to the MLIR python package resolver:
    # `_cext.globals.append_dialect_search_prefix('openxla')`
    m = ir.Operation.parse('sdy.mesh @mesh = <"c"=4>')
    self.assertIsInstance(m, sdy.ir.MeshOp)


if __name__ == '__main__':
  unittest.main()
