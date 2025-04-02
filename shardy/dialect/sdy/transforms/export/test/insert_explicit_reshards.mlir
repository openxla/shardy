// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xt = <["x"=2, "t"=4]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>
sdy.mesh @mesh_iota = <["x"=2, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=2, "y"=2], device_ids=[3, 2, 1, 0]>

// CHECK-LABEL: func @funcop_result_sharding_does_not_match
func.func @funcop_result_sharding_does_not_match(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_unsharded_but_different_meshes_between_return_and_func_result
func.func @funcop_result_unsharded_but_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_matches_but_different_meshes_between_return_and_func_result
func.func @funcop_result_sharding_matches_but_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xt, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result
func.func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}, {"t"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xt, [{}, {"t"}]> : tensor<8x16xf32>
  // CHECK: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result_multiple_results
func.func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result_multiple_results(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"t"}, {}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}, {}]>}, tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xt, [{"x"}, {}]> : tensor<8x32xf32>
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<32x16xf32>
  // CHECK: return %[[RESHARD1]], %[[RESHARD2]] : tensor<8x32xf32>, tensor<32x16xf32>
  return %arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>
}

// CHECK-LABEL: func @funcop_result_identical_sharding_but_different_meshes_between_return_and_func_result
func.func @funcop_result_identical_sharding_but_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}, {"t":(2)2}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xt, [{"x"}, {"t":(2)2}]> : tensor<8x16xf32>
  // CHECK: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_funcop_result_empty
func.func @funcop_result_sharding_does_not_match_funcop_result_empty(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_return_operand_empty
func.func @funcop_result_sharding_does_not_match_return_operand_empty(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_match
func.func @funcop_result_sharding_match(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK: return %arg0 : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_match_both_func_result_and_return_operand_empty
func.func @funcop_result_sharding_match_both_func_result_and_return_operand_empty(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<8x16xf32> {
  // CHECK-NOT: sdy.reshard
  // CHECK: return %arg0 : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_multiple_results
func.func @funcop_result_sharding_does_not_match_multiple_results(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<8x32xf32>
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<32x16xf32>
  // CHECK: return %[[RESHARD1]], %[[RESHARD2]] : tensor<8x32xf32>, tensor<32x16xf32>
  return %arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_multiple_results_only_one_does_not_match
func.func @funcop_result_sharding_does_not_match_multiple_results_only_one_does_not_match(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<32x16xf32>
  // CHECK: return %arg0, %[[RESHARD]] : tensor<8x32xf32>, tensor<32x16xf32>
  return %arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_multiple_results_different_meshes
func.func @funcop_result_sharding_does_not_match_multiple_results_different_meshes(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<8x32xf32>
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{}, {"x"}]> : tensor<32x16xf32>
  // CHECK: return %[[RESHARD1]], %[[RESHARD2]] : tensor<8x32xf32>, tensor<32x16xf32>
  return %arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_and_reshard_result_twice
func.func @funcop_result_sharding_does_not_match_and_reshard_result_twice(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_contracting_unsharded
func.func @dot_compatible_contracting_unsharded(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>
  // CHECK: return %[[DOT]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_contracting_dim_sharded
func.func @dot_compatible_contracting_dim_sharded(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
  // CHECK: return %[[ALL_REDUCE]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// This is a reduce-scatter pattern.
// CHECK-LABEL: func @dot_contracting_dim_and_result_dim_sharded_same_axis
func.func @dot_contracting_dim_and_result_dim_sharded_same_axis(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {"y"}]>
  // CHECK: return %[[RESHARD]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// This is a reduce-scatter pattern.
// CHECK-LABEL: func @dot_contracting_dim_and_result_dim_sharded_same_axis_2
func.func @dot_contracting_dim_and_result_dim_sharded_same_axis_2(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x", "y"}, {}]>
  // CHECK: return %[[RESHARD]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_contracting_dim_and_result_dim_sharded_same_axis_incompatible_order
func.func @dot_contracting_dim_and_result_dim_sharded_same_axis_incompatible_order(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"y", "x"}, {}]>
  // CHECK: return %[[RESHARD]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_lhs_contracting_and_rhs_non_contracting_dims
func.func @dot_incompatible_lhs_contracting_and_rhs_non_contracting_dims(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>
  // CHECK: return %[[DOT]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_subaxis_no_overlap
func.func @dot_incompatible_subaxis_no_overlap(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(2)2}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x":(2)2}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x":(1)2}, {"x":(2)2}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}, {"x":(2)2}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_jk
func.func @dot_compatible_jk(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_k
func.func @dot_compatible_k(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_empty
func.func @dot_compatible_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_contracting_dim_empty
func.func @dot_compatible_contracting_dim_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_empty
func.func @dot_incompatible_same_non_contracting_dims_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_i
func.func @dot_incompatible_same_non_contracting_dims_out_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_j
func.func @dot_incompatible_same_non_contracting_dims_out_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_out_empty
func.func @dot_incompatible_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_empty
func.func @dot_incompatible_i_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_i
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_j
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}, {"y"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_mismatch
func.func @dot_incompatible_i_mismatch(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // TODO(b/404475296): The cost of a2a is smaller than all-gather, hence it could reshard the result instead.
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}


// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded_smaller_local_contracting_dim
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded_smaller_local_contracting_dim(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]> : tensor<16x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_rhs_non_contracting_dim_is_sharded
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_rhs_non_contracting_dim_is_sharded(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_j_is_larger
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_j_is_larger(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_i_is_larger
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_i_is_larger(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<16x8xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_small_k
// TODO(enver): A better solution could be to reshard the result.
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_small_k(%arg0: tensor<128x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}]> : tensor<128x4xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<4x256xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x4xf32>, tensor<4x256xf32>) -> tensor<128x256xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<128x256xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x4xf32>, tensor<4x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_large_k
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_large_k(%arg0: tensor<8x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<1024x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x1024xf32>, tensor<1024x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x1024xf32>, tensor<1024x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_sub_axis_overlaps
func.func @dot_incompatible_sub_axis_overlaps(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_all_factors_mismatch
func.func @dot_incompatible_all_factors_mismatch(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[DOT]] out_sharding=<@mesh, [{}, {"y"}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_all_factors_mismatch_small_k
func.func @dot_incompatible_all_factors_mismatch_small_k(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}]> : tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_is_local
func.func @dot_reshard_is_local(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<32x16xf32>
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x16xf32>
  %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_does_not_change_input_sharding(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
func.func @dot_reshard_does_not_change_input_sharding(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_with_sharding_rule
func.func @dot_with_sharding_rule(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32} reduction={k}>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32} reduction={k}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_contracting_dim_sharded_multiple_axes
func.func @dot_genaral_contracting_dim_sharded_multiple_axes(
    %arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {"x", "z"}]>},
    %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"x", "z"}, {}]>})
    -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}]>}) {
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x", "z"} %[[DOTGENERAL]] out_sharding=<@mesh_xyz, [{"y"}, {}, {}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<4x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_multiple_contracting_dims_sharded
func.func @dot_genaral_multiple_contracting_dims_sharded(
    %arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"y"}, {"x", "z"}]>},
    %arg1: tensor<8x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"x", "z"}, {}]>})
    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}]>}) {
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1, 2] x [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}]>]>} : (tensor<4x8x32xf32>, tensor<8x32x16xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y", "x", "z"} %[[DOTGENERAL]] out_sharding=<@mesh_xyz, [{}, {}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<4x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1, 2] x [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}]>]>} : (tensor<4x8x32xf32>, tensor<8x32x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @dot_genaral_incompatable_with_batching_dims
func.func @dot_genaral_incompatable_with_batching_dims(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh, [{}, {"x"}, {"y"}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_incompatable_batching_factor_mismatch_on_all_tensors
func.func @dot_genaral_incompatable_batching_factor_mismatch_on_all_tensors(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"y"}, {}, {}]> : tensor<4x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{"z"}, {}, {}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_multiple_axes
func.func @dot_multiple_axes(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y"}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"z"}, {"x", "y"}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"z"} %[[DOT]] out_sharding=<@mesh_xyz, [{}, {"x", "y"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_multiple_axes_with_overlap
func.func @dot_multiple_axes_with_overlap(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z", "x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {"z"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"z"}, {"x", "y"}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"z"} %[[DOT]] out_sharding=<@mesh_xyz, [{}, {"x", "y"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_multiple_axes_with_overlap_on_suffix
func.func @dot_multiple_axes_with_overlap_on_suffix(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z", "x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {"z":(1)2}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"z":(1)2}, {"x", "y", "z":(2)2}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"z":(1)2} %[[DOT]] out_sharding=<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_is_strict_prefix_of_other
func.func @dot_genaral_one_is_strict_prefix_of_other(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y", "x":(1)2}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}]>}) -> tensor<4x8x16xf32> {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y", "x":(1)2}, {}, {}]> : tensor<4x32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %arg0, %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x":(1)2}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh_xyz, [{}, {}, {}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_prefix_has_larger_count
func.func @dot_genaral_one_prefix_has_larger_count(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y", "x":(1)2}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y", "x"}, {}, {}]>}) ->(tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"y", "x"}, {}, {}]> : tensor<4x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{"z"}, {}, {}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_suffix_has_larger_count_on_another_factor
func.func @dot_genaral_one_suffix_has_larger_count_on_another_factor(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y", "x":(1)2}, {"x":(2)2}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y", "x"}, {}, {}]>}) ->(tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x":(2)2}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y", "x":(1)2}, {}, {}]> : tensor<4x32x16xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x":(1)2}, {"x":(2)2}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{}, {"x":(2)2}, {}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x":(2)2}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_batching_dimension_shardings_have_common_prefix
func.func @dot_genaral_batching_dimension_shardings_have_common_prefix(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "x":(1)2,"t":(1)2}, {"t":(2)2}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "x":(1)2,"t":(2)2}, {}, {"t":(1)2}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y", "x":(1)2}, {"t":(2)2}, {}]> : tensor<64x8x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{"y", "x":(1)2}, {}, {"t":(1)2}]> : tensor<64x32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[RESHARD1]], %[[RESHARD2]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y", "x":(1)2}, {"t":(2)2}, {"t":(1)2}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[DOT]] <@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]> : tensor<64x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<64x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_one_is_strict_prefix_on_subaxes
func.func @dot_one_is_strict_prefix_on_subaxes(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"x":(2)2}, {}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x":(1)2}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x":(2)2} %[[DOT]] out_sharding=<@mesh_xyz, [{"x":(1)2}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_xyz, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_and_trimmable
func.func @dot_genaral_overlaps_and_trimmable(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"x","y","z"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y":(1)2}, {"x"}, {}]> : tensor<64x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(1)2}, {"x"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %1 <@mesh_xyzt, [{}, {"x", "y", "z"}, {}]> : tensor<64x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"x","y","z"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_from_most_major
func.func @dot_genaral_overlaps_from_most_major(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {}]>}) {
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(1)2}, {}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"y"}, {}]> : tensor<64x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<64x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_and_trimmable_on_subaxis
func.func @dot_genaral_overlaps_and_trimmable_on_subaxis(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y":(2)2}, {"y":(1)2}, {}]> : tensor<64x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(2)2}, {"y":(1)2}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"y"}, {}]> : tensor<64x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_and_trimmable_on_subaxis
func.func @dot_genaral_overlaps_and_trimmable_on_subaxis_multiple_axes(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"x","y","z"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y":(2)2}, {"x", "y":(1)2}, {}]> : tensor<64x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(2)2}, {"x", "y":(1)2}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"x", "y", "z"}, {}]> : tensor<64x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"x","y","z"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @cholesky_sharded_input_batch_dim_only
func.func @cholesky_sharded_input_batch_dim_only(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> tensor<8x4x8x8xf32> {
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_output_batch_dim_only
func.func @cholesky_sharded_output_batch_dim_only(%arg0: tensor<8x4x8x8xf32>) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_batch_dim_only_different
func.func @cholesky_sharded_batch_dim_only_different(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}, {}]>}){
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{"y"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_input_cholesky_dim_only
func.func @cholesky_sharded_input_cholesky_dim_only(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) -> tensor<8x4x8x8xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_output_cholesky_dim_only
func.func @cholesky_sharded_output_cholesky_dim_only(%arg0: tensor<8x4x8x8xf32>) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}){
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"x"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"x"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_only_different
func.func @cholesky_sharded_cholesky_dim_only_different(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"y"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"y"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"y"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_only_same
func.func @cholesky_sharded_cholesky_dim_only_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}){
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"x"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  // TODO(enver): Instead reshard to [{"x"}, {}, {}, {}] and perform the operation on this smaller tensor.
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"x"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_input_batch_dim_and_output_cholesky_dim_same
func.func @cholesky_sharded_input_batch_dim_and_output_cholesky_dim_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) {
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"x"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"x"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_output_batch_dim_and_input_cholesky_dim_same
func.func @cholesky_sharded_output_batch_dim_and_input_cholesky_dim_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same
func.func @cholesky_sharded_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {"y"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {"y"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{"x"}, {}, {}, {"y"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {"y"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same_both_cholesky_dimensions
func.func @cholesky_sharded_same_both_cholesky_dimensions(%arg0: tensor<128x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}) -> (tensor<128x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y", "z"}, {}, {}, {}]> : tensor<128x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y", "z"}, {}, {}, {}]>]>} : tensor<128x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]> : tensor<128x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>]>} : (tensor<128x4x8x8xf32>) -> tensor<128x4x8x8xf32>
  return %0 :  tensor<128x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim
func.func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim(%arg0: tensor<16x2x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}) -> (tensor<16x2x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {}, {}, {}]> : tensor<16x2x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {}, {}, {}]>]>} : tensor<16x2x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]> : tensor<16x2x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x2x8x8xf32>
  // TODO(enver): Instead reshard to [{"x", "y", "z":1(2)}, {}, {}, {}].
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>]>} : (tensor<16x2x8x8xf32>) -> tensor<16x2x8x8xf32>
  return %0 :  tensor<16x2x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim_second_batch_dim_larger
func.func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim_second_batch_dim_larger(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]>]>} : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
  return %0 :  tensor<16x8x8x8xf32>
}

// CHECK-LABEL: func @cholesky_batch_and_cholesky_dims_shardings_can_merge
func.func @cholesky_batch_and_cholesky_dims_shardings_can_merge(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2,"y"}, {"z"}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2, "y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]>]>} : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2, "y"}, {"z"}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2, "y"}, {"z"}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
  return %0 :  tensor<16x8x8x8xf32>
}

// CHECK-LABEL: func @cholesky_cholesky_dims_shardings_can_merge
func.func @cholesky_cholesky_dims_shardings_can_merge(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "z"}, {}, {}, {}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "z"}, {}, {}, {}]>]>} : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
  return %0 :  tensor<16x8x8x8xf32>
}


// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_input_sharding_larger
func.func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_input_sharding_larger(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}, {"z"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"y"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_output_sharding_larger
func.func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_output_sharding_larger(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}, {"z"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY:.*]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @reverse_no_permutation_dim_is_sharded_output_sharding_is_larger
func.func @reverse_no_permutation_dim_is_sharded_output_sharding_is_larger(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}, {}]>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]]
  // CHECK-NEXT: return %[[REVERSE]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_no_permutation_dim_is_sharded_input_sharding_is_larger
func.func @reverse_no_permutation_dim_is_sharded_input_sharding_is_larger(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}, {}]>}){
  // CHECK: %[[REVERSE:.*]] = stablehlo.reverse %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REVERSE]] <@mesh, [{"y"}, {}, {}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_only_input_permutation_dim_is_sharded
func.func @reverse_only_input_permutation_dim_is_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {}]>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]]
  // CHECK-NEXT: return %[[REVERSE]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_multiple_input_permutation_dims_are_sharded
func.func @reverse_multiple_input_permutation_dims_are_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"z":(1)2}, {}, {"z":(2)2}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(2)2}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x":(2)2}, {}, {}, {}]>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]]
  // CHECK-NEXT: return %[[REVERSE]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x":(2)2}, {}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_only_output_permutation_dim_is_sharded
func.func @reverse_only_output_permutation_dim_is_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {"z":(1)2}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z"}, {}, {}]>}) {
  // CHECK: %[[REVERSE:.*]] = stablehlo.reverse %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REVERSE]] <@mesh_xyz, [{"x"}, {"z"}, {}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {"z"}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_both_input_and_output_permutation_dims_are_sharded
func.func @reverse_both_input_and_output_permutation_dims_are_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z":(1)2}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]>]>}: tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @bitcast_convert_upcast
func.func @bitcast_convert_upcast(%arg0: tensor<4x2x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<4x2xui64> {
  // CHECK: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{}, {}]> : tensor<4x2xui64>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x2xui64>
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  return %0 :  tensor<4x2xui64>
}

// CHECK-LABEL: func @bitcast_convert_upcast_casting_dim_is_sharded
func.func @bitcast_convert_upcast_casting_dim_is_sharded(%arg0: tensor<4x2x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}) -> tensor<4x2xui64> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x2x2xui32>
  // CHECK-NEXT: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{}, {}]> : tensor<4x2xui64>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x2xui64>
  // TODO(enver): Instead reshard only once.
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
  return %0 :  tensor<4x2xui64>
}

// CHECK-LABEL: func @bitcast_convert_downcast
func.func @bitcast_convert_downcast(%arg0: tensor<4x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x2x2xui32> {
  // CHECK: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{}, {}, {}]> : tensor<4x2x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x2x2xui32>
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>}: (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  return %0 :  tensor<4x2x2xui32>
}

// CHECK-LABEL: func @bitcast_convert_downcast_casting_dim_is_sharded
func.func @bitcast_convert_downcast_casting_dim_is_sharded(%arg0: tensor<4x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x2x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}){
  // CHECK: %[[BITCAST_CONVERT:.*]] = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[BITCAST_CONVERT]] <@mesh, [{"x"}, {}, {"y"}]> : tensor<4x2x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x2x2xui32>
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  return %0 :  tensor<4x2x2xui32>
}

// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%arg0: tensor<2x3x5x1x7xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}, {}]>}) -> tensor<2x5x3x11x7x13xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}, {}, {}]> : tensor<2x3x5x1x7xf32>
  // CHECK-NEXT: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %[[RESHARD]], dims = [0, 2, 1, 3, 4] : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  // CHECK-NEXT: return %[[BROADCAST_IN_DIM]] : tensor<2x5x3x11x7x13xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1, 3, 4] : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  return %0 :  tensor<2x5x3x11x7x13xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_input_output_different
func.func @broadcast_in_dim_input_output_different(%arg0: tensor<2x3x5x1x7xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}, {}]>}) -> (tensor<2x5x3x11x7x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {"x"}, {}, {}]> : tensor<2x3x5x1x7xf32>
  // CHECK-NEXT: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %[[RESHARD]], dims = [0, 2, 1, 3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>]>} : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  // CHECK-NEXT: return %[[BROADCAST_IN_DIM]] : tensor<2x5x3x11x7x13xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1, 3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}, {}, {}, {"y"}]>]>} : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  return %0 :  tensor<2x5x3x11x7x13xf32>
}

// CHECK-LABEL: func @concatenate_single_input
func.func @concatenate_single_input(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) -> (tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>) -> tensor<4x32x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x32x256xf32>
  %0 = stablehlo.concatenate %arg0, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>) -> tensor<4x32x256xf32>
  return %0 : tensor<4x32x256xf32>
}

// CHECK-LABEL: func @concatenate
func.func @concatenate(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}]>}) -> tensor<4x80x256xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %arg0, %[[RESHARD1]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {}, {}]> : tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// TODO(b/396121070): In this case we would probably want to replicate the
// concat dimension.
// CHECK-LABEL: func @concatenate_concat_dim_is_sharded
func.func @concatenate_concat_dim_is_sharded(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<4x48x256xf32>) -> tensor<4x80x256xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @add
func.func @add(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32>) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.add %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_input_sharding_is_larger
func.func @add_input_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32>) {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ADD]] <@mesh, [{"y"}, {}]> : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_output_sharding_is_larger
func.func @add_output_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ADD]] <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_input_and_output_sharded_on_separate_dims
func.func @add_input_and_output_sharded_on_separate_dims(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {"y"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[ADD]] <@mesh, [{}, {"y"}]> : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_inputs_are_sharded_the_same_way_output_is_unsharded
func.func @add_inputs_are_sharded_the_same_way_output_is_unsharded(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x32xf32> {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ADD]] <@mesh, [{}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate
func.func @negate(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_input_sharding_is_larger
func.func @negate_input_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{"y"}, {}]> : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_output_sharding_is_larger
func.func @negate_output_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_input_and_output_sharded_on_separate_dims
func.func @negate_input_and_output_sharded_on_separate_dims(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{}, {"y"}]> : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_input_and_output_sharded_on_separate_dims_output_sharding_is_larger
func.func @negate_input_and_output_sharded_on_separate_dims_output_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<32x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}, {}]> : tensor<32x4x8xf32>
  // CHECK-NEXT: %[[DYNAMIC_SLICE:.*]] = stablehlo.dynamic_slice %[[RESHARD1]], %arg1, %arg2, %arg3, sizes = [32, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DYNAMIC_SLICE]] <@mesh, [{}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"y"}]>]>}: (tensor<32x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @dynamic_slice_input_output_same_sharding
func.func @dynamic_slice_input_output_same_sharding(%arg0: tensor<32x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<32x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}, {}]> : tensor<32x8x8xf32>
  // CHECK-NEXT: %[[DYNAMIC_SLICE:.*]] = stablehlo.dynamic_slice %[[RESHARD]], %arg1, %arg2, %arg3, sizes = [32, 4, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}]>]>} : (tensor<32x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x2xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DYNAMIC_SLICE:.*]] <@mesh, [{}, {"x"}, {"y"}]> : tensor<32x4x2xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<32x4x2xf32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 4, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>}: (tensor<32x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x2xf32>
  return %0 : tensor<32x4x2xf32>
}

// CHECK-LABEL: func @dynamic_update_slice
func.func @dynamic_update_slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<32x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE:.*]] = stablehlo.dynamic_update_slice %arg0, %[[RESHARD1]], %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[DYNAMIC_UPDATE_SLICE]] : tensor<32x4x8xf32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>} : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}

// CHECK-LABEL: func @dynamic_update_slice_different_input_and_output_sharding
func.func @dynamic_update_slice_different_input_and_output_sharding(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<32x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}, {"x"}]> : tensor<32x4x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE:.*]] = stablehlo.dynamic_update_slice %[[RESHARD1]], %[[RESHARD2]], %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  // CHECK-NEXT: return %[[DYNAMIC_UPDATE_SLICE]] : tensor<32x4x8xf32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}

// CHECK-LABEL: func @dynamic_slice_batching_dim_is_sharded_on_input
func.func @dynamic_slice_batching_dim_is_sharded_on_input(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<32x1x2xf32> {
  // CHECK: %[[DYNAMIC_SLICE:.*]] = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2]
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DYNAMIC_SLICE]] <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2] : (tensor<32x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @dynamic_update_slice_batching_dim_is_sharded_on_input
func.func @dynamic_update_slice_batching_dim_is_sharded_on_input(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<32x1x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<32x4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE:.*]] = stablehlo.dynamic_update_slice %arg0, %[[RESHARD1]], %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DYNAMIC_UPDATE_SLICE]] <@mesh, [{}, {}, {}]> : tensor<32x4x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}){
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_only_input_permutation_dim_is_sharded_non_permutation_dim_compatible
func.func @pad_only_input_permutation_dim_is_sharded_non_permutation_dim_compatible(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {}, {"x"}]>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]]
  // CHECK-NEXT: return %[[PAD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"x"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_only_input_permutation_dim_is_sharded
func.func @pad_only_input_permutation_dim_is_sharded(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]]
  // CHECK-NEXT: return %[[PAD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_only_output_permutation_dim_is_sharded
func.func @pad_only_output_permutation_dim_is_sharded(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyz, [{"z"}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_both_input_and_output_permutation_dims_are_sharded
func.func @pad_both_input_and_output_permutation_dims_are_sharded(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z"}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"z"}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<32x1x2xf32> {
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @slice_no_permutation_dim_is_sharded
func.func @slice_no_permutation_dim_is_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_only_input_permutation_dim_is_sharded
func.func @slice_only_input_permutation_dim_is_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD]]
  // CHECK-NEXT: return %[[SLICE]]
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_only_output_permutation_dim_is_sharded
func.func @slice_only_output_permutation_dim_is_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"z"}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_both_input_and_output_permutation_dims_are_sharded
func.func @slice_both_input_and_output_permutation_dims_are_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z"}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"z"}, {"y"}]>]>}: (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @sort
func.func @sort(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: "stablehlo.sort"(%[[RESHARD]], %arg1)
  %0:2 = "stablehlo.sort"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<4x32x8xi32>, tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>)
  return %0#0, %0#1 : tensor<4x32x8xi32>, tensor<4x32x8xf32>
}

// CHECK-LABEL: func @sort_all_other_dims_size_one
func.func @sort_all_other_dims_size_one(%arg0: tensor<1x4x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<1x4x1xi32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<1x4x1xi32>
  // CHECK-NEXT: "stablehlo.sort"(%[[RESHARD]])
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
  return %0 : tensor<1x4x1xi32>
}

// CHECK-LABEL: func @sort_single_input_output
func.func @sort_single_input_output(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x32x8xi32>) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD1]])
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %[[SORT]] <@mesh, [{}, {}, {}]> : tensor<4x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_compatible
func.func @sort_compatible(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_input_and_output_shardings_are_same_on_sorting_dimension
func.func @sort_input_and_output_shardings_are_same_on_sorting_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y", "x"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD1]])
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %[[SORT]] <@mesh, [{"x"}, {"y"}, {}]> : tensor<4x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_input_and_output_shardings_are_different_on_sorting_dimension
func.func @sort_input_and_output_shardings_are_different_on_sorting_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {"z"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD1]])
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %[[SORT]] <@mesh_xyz, [{"y"}, {"z"}, {}]> : tensor<4x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_sorting_dim_shardings_has_common_prefix
func.func @sort_sorting_dim_shardings_has_common_prefix(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","z"}, {"x"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","t"}, {"z"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"z", "y"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD1]])
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %[[SORT]] <@mesh_xyzt, [{"y", "t"}, {"z"}, {}]> : tensor<4x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y","t"}, {"z"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}


// CHECK-LABEL: func @sort_sorting_dim_shardings_has_common_prefix_and_large
func.func @sort_sorting_dim_shardings_has_common_prefix_and_large(%arg0: tensor<64x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","t","z"}, {"x"}, {}]>}) -> (tensor<64x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","t"}, {"z"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"z", "y"}, {"t"}]> : tensor<64x32x8xi32>
  // CHECK-NEXT: %[[SORT:.*]] = "stablehlo.sort"(%[[RESHARD1]])
  // CHECK: %[[RESHARD2:.*]] = sdy.reshard %[[SORT]] <@mesh_xyzt, [{"y", "t"}, {"z"}, {}]> : tensor<64x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y","t"}, {"z"}, {}]>]>} : (tensor<64x32x8xi32>) -> (tensor<64x32x8xi32>)
  return %0 : tensor<64x32x8xi32>
}

// CHECK-LABEL: func @sort_incompatible_on_nonsort_dimensions
func.func @sort_incompatible_on_nonsort_dimensions(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %0 <@mesh, [{}, {"y"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x32x8xi32>
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_compatible_on_nonsort_dimension
func.func @sort_compatible_on_nonsort_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<256x32x64x100xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}, {"y"}, {}]>}) -> (tensor<100x32x256x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"y"}, {"z"}, {}]>}) {
  // CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x"}, {}, {"y"}]>]>} : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[TRANSPOSE]] <@mesh_xyz, [{}, {"y"}, {"z"}, {}]> : tensor<100x32x256x64xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<100x32x256x64xf32>
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"y"}, {"z"}, {}]>]>} : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  return %0 : tensor<100x32x256x64xf32>
}

// CHECK-LABEL: func @triangular_solve
func.func @triangular_solve(%arg0: tensor<8x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: %[[TRIANGULAR_SOLVE:.*]] = "stablehlo.triangular_solve"(%arg0, %[[RESHARD1]])
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[TRIANGULAR_SOLVE]] <@mesh, [{}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x3x5xf32>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  return %0 : tensor<8x3x5xf32>
}

// CHECK-LABEL: func @triangular_solve_replicated_dim_is_sharded
func.func @triangular_solve_replicated_dim_is_sharded(%arg0: tensor<8x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<8x3x3xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: %[[TRIANGULAR_SOLVE:.*]] = "stablehlo.triangular_solve"(%[[RESHARD1]], %[[RESHARD2]])
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[TRIANGULAR_SOLVE]] <@mesh, [{}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x3x5xf32>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  return %0 : tensor<8x3x5xf32>
}

// CHECK-LABEL: func @fft_complex
func.func @fft_complex(%arg0: tensor<8x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<8x32x64xcomplex<f32>> {
  // CHECK-NOT: sdy.reshard
  %0  = stablehlo.fft %arg0, type = FFT, length = [32, 64] : (tensor<8x32x64xcomplex<f32>>) -> tensor<8x32x64xcomplex<f32>>
  return %0 : tensor<8x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_real
func.func @fft_real(%arg0: tensor<8x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<8x32x33xcomplex<f32>> {
  // CHECK-NOT: sdy.reshard
  %0  = stablehlo.fft %arg0, type = RFFT, length = [32, 64] : (tensor<8x32x64xf32>) -> tensor<8x32x33xcomplex<f32>>
  return %0 : tensor<8x32x33xcomplex<f32>>
}

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<48x48x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<48x48x3xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>)
    -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>) {
  // CHECK-NOT: sdy.reshard
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<i32>, %arg6: tensor<f32>, %arg7 : tensor<i32>):
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }) {window_dimensions = array<i64: 3, 1, 3>,
      window_strides = array<i64: 3, 1, 3>,
      padding = dense<[[0, 0], [2, -2], [0, 0]]> : tensor<3x2xi64>}
      : (tensor<48x48x3xf32>, tensor<48x48x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>)
  func.return %0#0, %0#1 : tensor<16x48x1xf32>, tensor<16x48x1xi32>
}

// CHECK-LABEL: @scatter_single_input
func.func @scatter_single_input(%arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<2x3x2xi64>, %arg2: tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>{
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>, tensor<2x3x2x2xf32>) -> tensor<3x4x2xf32>
  return %0 : tensor<3x4x2xf32>
}

// CHECK-LABEL: func @select_and_scatter
func.func @select_and_scatter(%arg0: tensor<10x24x24x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}, %arg1: tensor<10x12x12x64xf32>, %arg2: tensor<f32>)
   -> tensor<10x24x24x64xf32> {
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  // CHECK-NOT: sdy.reshard
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.compare GT, %arg3, %arg4 :(tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  return %1 : tensor<10x24x24x64xf32>
}

// CHECK-LABEL: @gather
func.func @gather(%arg0: tensor<3x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"x"}]>}, %arg1: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
  return %0 : tensor<2x3x2x2xf32>
}

// CHECK-LABEL: func @reshape
func.func @reshape(%arg0: tensor<16x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}, {"x"}]> : tensor<16x2x4xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<16x8xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// TODO(enver): Add a unit test for overflow axes on reshapes.

// CHECK-LABEL: func @convolution
func.func @convolution(%arg0 : tensor<2x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}, %arg1 : tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<2x224x224x192xf32>, tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// CHECK-LABEL: func @custom_call
func.func @custom_call(%arg0: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<128x128xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @reduce_single_result_reduction_dim_not_sharded
func.func @reduce_single_result_reduction_dim_not_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REDUCE]] <@mesh, [{}, {}]> : tensor<2x13xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2x13xf32>
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_reduction_dim_sharded
func.func @reduce_single_result_reduction_dim_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1]
  // CHECK-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_multiple_results
func.func @reduce_multiple_results(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<2x64x13xi32>)
    -> (tensor<64xf32>, tensor<64xi32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<2x64x13xi32>
  // CHECK-NEXT: %[[REDUCE:.*]]:2 = stablehlo.reduce(%arg0 init: %cst), (%[[RESHARD]] init: %c) across dimensions = [0, 2]
  // CHECK-NOT:  sdy.sharding
  // CHECK:      %[[ALL_REDUCE1:.*]] = sdy.all_reduce {"x"} %[[REDUCE]]#0 out_sharding=<@mesh, [{}]> : tensor<64xf32>
  // CHECK-NEXT: %[[ALL_REDUCE2:.*]] = sdy.all_reduce {"x"} %[[REDUCE]]#1 out_sharding=<@mesh, [{}]> : tensor<64xi32>
  // CHECK-NEXT: return %[[ALL_REDUCE1]], %[[ALL_REDUCE2]] : tensor<64xf32>, tensor<64xi32>
  %2:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1) across dimensions = [0, 2] :
    (tensor<2x64x13xf32>, tensor<2x64x13xi32>, tensor<f32>, tensor<i32>) -> (tensor<64xf32>, tensor<64xi32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %3 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %4 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f32>, tensor<i32>
    }
  return %2#0, %2#1 : tensor<64xf32>, tensor<64xi32>
}

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[CLAMP:.*]] = stablehlo.clamp %[[RESHARD1]], %arg1, %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CLAMP]] <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<4x8xf32>
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @clamp_scalar_min_max
func.func @clamp_scalar_min_max(%arg0: tensor<f32>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}, %arg2: tensor<f32>) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y"}, {"z"}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[CLAMP:.*]] = stablehlo.clamp %arg0, %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}]>]>} : (tensor<f32>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %1 : tensor<4x8xf32>
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}]>]>}: (tensor<f32>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @select
func.func @select(%arg0: tensor<4x8xi1>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xi1>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[SELECT:.*]] = stablehlo.select %[[RESHARD1]], %arg1, %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xi1>, tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[SELECT]] <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<4x8xf32>
  %0 = stablehlo.select %arg0, %arg1, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @select_scalar_pred
func.func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg2 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[SELECT:.*]] = stablehlo.select %arg0, %arg1, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<i1>, tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[SELECT]] <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8xf32>
  %0 = stablehlo.select %arg0, %arg1, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<i1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @while
func.func @while(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT stablehlo.while(%iterArg = %[[RESHARD]], %iterArg_2 = %c)
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>, <@mesh, []>]>}
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %iterArg <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
    %3 = stablehlo.negate %iterArg {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}: tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]], %{{.*}} : tensor<210xf32>, tensor<i32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  %1 = stablehlo.negate %0#0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @while_argument_used_outside_block
func.func @while_argument_used_outside_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]> : tensor<210xf32>
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %[[RESHARD]], %iterArg_2 = %c)
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>, <@mesh, []>]>}
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %iterArg <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
    %3 = stablehlo.negate %iterArg {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}: tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]], %{{.*}} : tensor<210xf32>, tensor<i32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %[[WHILE]]#0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.add %arg0, %[[RESHARD]]
  %1 = stablehlo.add %arg0, %0#0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @while_missing_sharding
func.func @while_missing_sharding(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<210xf32>
  // CHECK-NEXT stablehlo.while(%iterArg = %[[RESHARD]], %iterArg_2 = %c)
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32>
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %iterArg <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
    %3 = stablehlo.negate %iterArg {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}: tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]], %{{.*}} : tensor<210xf32>, tensor<i32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  %1 = stablehlo.negate %0#0 : tensor<210xf32>
  return %1: tensor<210xf32>
}

// CHECK-LABEL: func @while_fully_replicated_everywhere
func.func @while_fully_replicated_everywhere(%arg0: tensor<210xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<210xf32>) {
  // CHECK-NOT: sdy.reshard
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %c_2 = stablehlo.constant dense<0.0> : tensor<f32>
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<210xf32>, tensor<i32>
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.dot %iterArg, %iterArg : (tensor<210xf32>, tensor<210xf32>) -> tensor<f32>
    %4 = stablehlo.compare  LT, %3, %c_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.and %2, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %c_0 : tensor<i32>
    %3 = stablehlo.negate %iterArg : tensor<210xf32>
    %4 = stablehlo.add %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<210xf32>
    stablehlo.return %4, %2 : tensor<210xf32>, tensor<i32>
  }
  %1 = stablehlo.negate %0#0 : tensor<210xf32>
  return %1: tensor<210xf32>
}

// CHECK-LABEL: func @case
func.func @case(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}, %arg1: tensor<i32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  %0 = "stablehlo.case"(%arg1) ({
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.abs %[[RESHARD]]
    %2 = stablehlo.abs %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]] : tensor<210xf32>
    stablehlo.return %2 : tensor<210xf32>
  }, {
    %2 = stablehlo.cosine %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %2 <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.abs %[[RESHARD]]
    %3 = stablehlo.abs %2 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
    stablehlo.return %3 : tensor<210xf32>
  }, {
    %2 = stablehlo.abs %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]] : tensor<210xf32>
    stablehlo.return %2 : tensor<210xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<i32>) -> tensor<210xf32>
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @case_argument_used_outside_block
func.func @case_argument_used_outside_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg1: tensor<i32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  %1 = "stablehlo.case"(%arg1) ({
    // CHECK: stablehlo.return %[[RESHARD1]]
    stablehlo.return %arg0 : tensor<210xf32>
  }, {
    // CHECK: %[[RESHARD2:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[RESHARD2]]
    // CHECK-NEXT: stablehlo.return %[[ABS]] : tensor<210xf32>
    %4 = stablehlo.abs %arg0 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    stablehlo.return %4 : tensor<210xf32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<i32>) -> tensor<210xf32>
  // CHECK: stablehlo.add %[[NEGATE]], %arg0
  %2 = stablehlo.add %0, %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  %3 = stablehlo.add %1, %2 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
  return %3 : tensor<210xf32>
}

// CHECK-LABEL: func @named_computation
func.func @named_computation(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: sdy.named_computation<"foo">(%[[RESHARD]])
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @named_computation_empty_block
func.func @named_computation_empty_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: sdy.named_computation<"foo">(%arg0)
  %0 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] (%arg1: tensor<210xf32>) {
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %arg1 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: sdy.manual_computation(%[[RESHARD]])
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @manual_computation_with_manual_axes
func.func @manual_computation_with_manual_axes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x","y"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x","z"}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_xyzt, [{"x","y"}]>] out_shardings=[<@mesh_xyzt, [{"x", "z"}]>] manual_axes={"x"} (%arg1: tensor<52xf32>) {
    // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{"t"}]> : tensor<52xf32>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"t"}]>]>} : tensor<52xf32>
    // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ABS]] <@mesh_xyzt, [{"z"}]> : tensor<52xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD2]] : tensor<52xf32>
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh_xyzt, [{"t"}]>]>} : tensor<52xf32>
    sdy.return %2 : tensor<52xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_xyzt, [{"x","z"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @optimization_barrier
func.func @optimization_barrier(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} %[[RESHARD]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}

// CHECK-LABEL: func @optimization_barrier_different_meshes
func.func @optimization_barrier_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xt, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} %[[RESHARD]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}

// CHECK-LABEL: func @optimization_barrier_meshes_different_device_order
func.func @optimization_barrier_meshes_different_device_order(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} %[[RESHARD]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_iota_sharded
func.func @negate_from_empty_sharding_to_iota_sharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_iota_unsharded
func.func @negate_from_empty_sharding_to_iota_unsharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_non_iota_sharded
func.func @negate_from_empty_sharding_to_non_iota_sharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_non_iota_unsharded
func.func @negate_from_empty_sharding_to_non_iota_unsharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}
