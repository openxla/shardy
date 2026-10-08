// RUN: sdy_opt %s -split-input-file -verify-diagnostics

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{'func.func' op arg 0 - non-shaped tensors can only have a sharding with rank 0 and no replicated or unreduced axes}}
func.func @token_sharding_rank_non_zero(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [{}]>}) -> !stablehlo.token {
  return %arg0 : !stablehlo.token
}

// -----

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{'func.func' op arg 0 - non-shaped tensors can only have a sharding with rank 0 and no replicated or unreduced axes}}
func.func @token_sharding_with_replicated_axes(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [], replicated={"a"}>}) -> !stablehlo.token {
  return %arg0 : !stablehlo.token
}

// -----

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{'func.func' op arg 0 - non-shaped tensors can only have a sharding with rank 0 and no replicated or unreduced axes}}
func.func @token_sharding_with_unreduced_axes(%arg0: !stablehlo.token {sdy.sharding=#sdy.sharding<@mesh, [], unreduced={"a"}>}) -> !stablehlo.token {
  return %arg0 : !stablehlo.token
}

// -----

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{'func.func' op arg 0 - only ranked tensors can have a sharding}}
func.func @unranked_tensor_with_sharding(%arg0: tensor<*xf32> {sdy.sharding=#sdy.sharding<@mesh, []>}) -> tensor<*xf32> {
  return %arg0 : tensor<*xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @dim_shardings_rank_mismatch(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{op result - sharding doesn't match tensor rank: 2 != 1}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"b"}], replicated={"a"}>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{op arg 0 - sharding doesn't match tensor rank: 1 != 2}}
func.func @dynamic_shaped_tensor_rank_mismatch(%arg0: tensor<?x?xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> tensor<?x?xf32> {
  return %arg0 : tensor<?x?xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{op arg 0 - dim 0 is empty and closed but has a priority}}
func.func @dynamic_shaped_tensor_empty_closed_priority(%arg0: tensor<?x?xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}p1, {}]>}) -> tensor<?x?xf32> {
  return %arg0 : tensor<?x?xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @op_with_unknown_mesh(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{op result - unknown mesh: @other_mesh}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@other_mesh, [{}, {"a"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{'func.func' op arg 0 - unknown mesh: @other_mesh}}
func.func @func_arg_unknown_mesh(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@other_mesh, [{}, {"a"}]>},
                                 %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1: tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @op_with_tensor_sharding_attr
func.func @op_with_tensor_sharding_attr(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{op should have a sharding attribute of type TensorShardingPerValueAttr}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding<@mesh, [{}, {"a"}]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// expected-error @+1 {{'func.func' op arg 1 - should have a sharding attribute of type TensorShardingAttr}}
func.func @func_arg_with_tensor_sharding_per_value_attr(
    %arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"a"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>})
    -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1: tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @num_shardings_does_not_match_num_results
func.func @num_shardings_does_not_match_num_results(%arg0: tensor<2x64x13xf32>, %arg1: tensor<2x64x13xf32>) -> (tensor<2x13xf32>, tensor<2x13xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
   // expected-error @+1 {{op result shardings don't match number of values: 3 shardings vs 2 values}}
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>,
                                           <@mesh, [{"b"}, {}]>,
                                           <@mesh, [{"a"}, {}]>]>} :
    (tensor<2x64x13xf32>, tensor<2x64x13xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x13xf32>, tensor<2x13xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<2x13xf32>, tensor<2x13xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// The purpose of this test is to check the error msg prefix for a func arg.
// expected-error @+1 {{'func.func' op arg 0 - unknown axis name: "c"}}
func.func @func_arg_failure(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"c"}]>},
                            %arg1: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"a"}, {}]>})
    -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1: tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// The purpose of this test is to check the error msg prefix for a func result.
// expected-error @+1 {{'func.func' op result 2 - unknown axis name: "c"}}
func.func @func_result_failure(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"a"}]>},
                               %arg1: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"c"}, {}]>}) {
  return %arg0, %arg1, %arg0: tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// The purpose of this test is to check the error msg prefix for a single-result
// op.
func.func @single_result_op_failure(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{op result - unknown axis name: "c"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"c"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// The purpose of this test is to check the error msg prefix for a multi-result
// op.
func.func @multi_result_op_failure(%arg0: tensor<2x64x13xf32>, %arg1: tensor<2x64x13xf32>) -> (tensor<2x13xf32>, tensor<2x13xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
   // expected-error @+1 {{op result 1 - unknown axis name: "c"}}
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{"c"}, {}]>]>} :
    (tensor<2x64x13xf32>, tensor<2x64x13xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x13xf32>, tensor<2x13xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<2x13xf32>, tensor<2x13xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @unknown_axis_mesh_ref(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{unknown axis name: "c"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"c"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

func.func @unknown_axis_inlined_mesh(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{unknown axis name: "c"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<mesh<["a"=2, "b"=2]>, [{}, {"c"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @unknown_replicated_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{unknown axis name: "c"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"c"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @unknown_unreduced_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{unknown axis name: "c"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"c"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @duplicate_axis_dim_sharding_and_replicated(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], replicated={"a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @duplicate_axis_dim_sharding_and_unreduced(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], unreduced={"a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @duplicate_axis_replicated_and_unreduced(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"a"}, unreduced={"a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @duplicate_replicated_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"a", "a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @duplicate_unreduced_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a", "a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @duplicate_sharded_axis_same_dim(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a", "a"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @duplicate_sharded_axis_different_dims(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a"}, {"a"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>

func.func @unordered_replicated_axes(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{replicated axes are not ordered w.r.t. mesh}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"a", "b", "c"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>

func.func @unordered_unreduced_axes(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{unreduced axes are not ordered w.r.t. mesh}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a", "b", "c"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2,"b"=4]>

func.func @empty_closed_dim_sharding_with_priority(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{dim 1 is empty and closed but has a priority}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}p3]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @unknown_sub_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{unknown axis name: "c"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"c":(2)2}], replicated={"a"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @duplicate_sub_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{duplicate axis ref: "a":(2)2}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)2}], replicated={"a":(2)2}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @both_full_axis_and_sub_axis_used_replicated(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{both sub-axis and full-axis are used for axis name: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], replicated={"a":(2)2}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @both_full_axis_and_sub_axis_used_unreduced(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{both sub-axis and full-axis are used for axis name: "a"}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}], unreduced={"a":(2)2}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @unordered_replicated_sub_axes(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{replicated axes are not ordered w.r.t. mesh}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"a":(4)2, "a":(1)2, "b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @unordered_unreduced_sub_axes(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{unreduced axes are not ordered w.r.t. mesh}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a":(4)2, "a":(1)2, "b"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @redundant_sub_axis_in_dim(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{two consecutive sub-axes can be merged: "a":(2)2, "a":(4)4}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)2, "a":(4)4}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @redundant_replicated_sub_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{two consecutive sub-axes can be merged: "a":(2)2, "a":(4)4}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], replicated={"a":(2)2, "a":(4)4}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @redundant_unreduced_sub_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{two consecutive sub-axes can be merged: "a":(2)2, "a":(4)4}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a":(2)2, "a":(4)4}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @sub_axis_non_positive_pre_size(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{sub-axis pre-size must be at least 1: "a":(-1)2}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(-1)2}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @sub_axis_size_one(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{sub-axis sizes must be greater than 1: "a":(2)1}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)1}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @sub_axis_pre_size_does_not_divide(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  // expected-error @+1 {{sub-axis next pre-size 6 doesn't divide the size of the full axis 8: "a":(3)2}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(3)2}]>]>} : tensor<6x6xf32>
  return %0 : tensor<6x6xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @sub_axis_size_does_not_divide(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  // expected-error @+1 {{sub-axis next pre-size 6 doesn't divide the size of the full axis 8: "a":(2)3}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(2)3}]>]>} : tensor<6x6xf32>
  return %0 : tensor<6x6xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @sub_axis_next_pre_size_beyond_axis(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{sub-axis next pre-size 16 doesn't divide the size of the full axis 8: "a":(4)4}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(4)4}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=2]>

func.func @sub_axis_size_equals_full_size(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{sub-axis size is equal to the full axis size: "a":(1)8}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a":(1)8}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=4]>

func.func @sub_axes_overlap(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{overlapping sub-axes: "a":(1)4, "a":(2)4}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a":(2)4}, {"b":(2)2}], replicated={"a":(1)4}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=8, "b"=4]>

func.func @sub_axes_overlap_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{overlapping sub-axes: "a":(1)4, "a":(2)4}}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a":(2)4}, {"b":(2)2}], replicated={"a":(1)4}>]>} : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @sharding_bound_manual_computation(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) { // expected-note  {{parent bounding this axis as manual}}
    %0 = stablehlo.add %arg1, %arg1 {sdy.sharding=#sdy.sharding_per_value<[ <@mesh, [{"a"}, {}]>]>} : tensor<8x32xf32> // expected-error {{'stablehlo.add' op result - operates on axis "a" which is already bound by a parent sdy.manual_computation op}}
    sdy.return %0 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @maximal_mesh = <[], device_ids=[0]>

func.func @maximal_sharding_with_dim_shardings(%arg0: tensor<8x8xf32>) -> tuple<tensor<8x8xf32>> {
  // expected-error @+1 {{a maximal sharding can only have a sharding with rank 0 and no replicated or unreduced axes}}
  %0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>>
  return %0 : tuple<tensor<8x8xf32>>
}

// -----

sdy.mesh @maximal_mesh = <[], device_ids=[0]>

func.func @maximal_sharding_no_results_with_dim_shardings(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{a maximal sharding can only have a sharding with rank 0 and no replicated or unreduced axes}}
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [{}]>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// -----

sdy.mesh @maximal_mesh = <[], device_ids=[0]>

func.func @maximal_sharding_with_replicated_axes(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{a maximal sharding can only have a sharding with rank 0 and no replicated or unreduced axes}}
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [], replicated={"a"}>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// -----

sdy.mesh @maximal_mesh = <[], device_ids=[0]>

func.func @maximal_sharding_with_unreduced_axes(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{a maximal sharding can only have a sharding with rank 0 and no replicated or unreduced axes}}
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, [], unreduced={"a"}>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @two_tuple(%arg0: tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8x8xf32>> {
  // expected-error @+1 {{ops can only have a sharding for a tuple of size 1}}
  %0 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8x8xf32>>
  return %0 : tuple<tensor<8x8xf32>, tensor<8x8xf32>>
}
