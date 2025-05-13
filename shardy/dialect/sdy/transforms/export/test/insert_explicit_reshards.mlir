// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xt = <["x"=2, "t"=4]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>
sdy.mesh @mesh_xyzp = <["x"=4, "y"=2, "z"=4, "p"=3]>
sdy.mesh @mesh_xpq = <["x"=4, "p"=3, "q"=5]>
sdy.mesh @mesh_xy = <["x"=2, "y"=3]>
sdy.mesh @mesh_iota = <["x"=3, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=3, "y"=2], device_ids=[5, 4, 3, 2, 1, 0]>
sdy.mesh @mesh_maximal = #sdy.mesh<[], device_ids=[0]>
sdy.mesh @mesh_maximal_copy = #sdy.mesh<[], device_ids=[0]>
sdy.mesh @mesh_maximal_another = #sdy.mesh<[], device_ids=[1]>

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

// CHECK-LABEL: func @dot_incompatible_a_times_a
func.func @dot_incompatible_a_times_a(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT return %[[DOT]]
  %0 = stablehlo.dot %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_all_same_shardings
func.func @dot_incompatible_all_same_shardings(%arg0: tensor<4x4096xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x4096xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[DOT]] out_sharding=<@mesh, [{}, {}]> : tensor<4x4096xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {}]>
  // CHECK-NEXT return %[[DOT]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x4096xf32>, tensor<4096x4096xf32>) -> tensor<4x4096xf32>
  return %0 : tensor<4x4096xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_i
func.func @dot_incompatible_same_non_contracting_dims_out_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
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

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded_small_lhs
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded_small_lhs(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded_large_lhs
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded_large_lhs(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
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
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x1024xf32>, tensor<1024x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x1024xf32>, tensor<1024x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_sub_axis_overlaps
func.func @dot_incompatible_sub_axis_overlaps(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x":(1)2}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(2)2}, {"x":(1)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{"x":(2)2}, {"x":(1)2}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
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

// The following test target is from b/410643498.
// CHECK-LABEL: func @dot_genaral_multiple_contracting_dims_conflicts
func.func @dot_genaral_multiple_contracting_dims_conflicts(
  %arg0: tensor<16x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x"}, {"z"}, {"t"}]>},
  %arg1: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"z"}, {"t"}, {"x", "y"}]>})
  ->(tensor<16x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x"}, {"z", "t"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"z"}, {"t"}]> : tensor<16x32x64xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, contracting_dims = [1, 2] x [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"x", "y"}]>]>} : (tensor<16x32x64xf32>, tensor<32x64x128xf32>) -> tensor<16x128xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"z", "t"} %[[DOT]] out_sharding=<@mesh_xyzt, [{}, {"x", "y"}]> : tensor<16x128xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_xyzt, [{"x"}, {"z", "t"}]> : tensor<16x128xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x128xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1, 2] x [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"x"}, {"z", "t"}]>]>} : (tensor<16x32x64xf32>, tensor<32x64x128xf32>) -> tensor<16x128xf32>
  return %0 : tensor<16x128xf32>
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
func.func @dot_genaral_one_suffix_has_larger_count_on_another_factor(%arg0: tensor<4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y", "x":(1)2}, {"x":(2)2}, {}]>}, %arg1: tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y", "x"}, {}, {}]>}) ->(tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x":(2)2}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y", "x":(1)2}, {}, {}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x":(1)2}, {"x":(2)2}, {}]>]>} : (tensor<4x8x8xf32>, tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{}, {"x":(2)2}, {}]> : tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x":(2)2}, {}]>]>} : (tensor<4x8x8xf32>, tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_batching_dimension_shardings_have_common_prefix
func.func @dot_genaral_batching_dimension_shardings_have_common_prefix(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "x":(1)2, "t":(1)2}, {"t":(2)2}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y", "x":(1)2, "t":(2)2}, {}, {"t":(1)2}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y", "x":(1)2, "t":(2)2}, {}, {}]> : tensor<64x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y", "x":(1)2, "t":(2)2}, {}, {"t":(1)2}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]> : tensor<64x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_lhs_sharding_has_subaxes_and_is_strict_prefix_of_rhs_sharding_and_rhs_is_large
func.func @dot_lhs_sharding_has_subaxes_and_is_strict_prefix_of_rhs_sharding_and_rhs_is_large(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {"x"}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[DOT]] out_sharding=<@mesh_xyz, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_xyz, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_lhs_sharding_has_subaxes_and_is_strict_prefix_of_rhs_sharding_and_rhs_is_small
func.func @dot_lhs_sharding_has_subaxes_and_is_strict_prefix_of_rhs_sharding_and_rhs_is_small(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"x":(2)2}]>}, %arg1: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}]> : tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{}, {}]> : tensor<4x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_result_size_is_equal_to_the_sum_of_operand_sizes
func.func @dot_result_size_is_equal_to_the_sum_of_operand_sizes(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}]>}, %arg1: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}]> : tensor<4x2xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{}, {}]> : tensor<2x4xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}]>]>} : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<4x4xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}]>]>} : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
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
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]> : tensor<4x32x8x2xf32>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]], dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]>]>} : tensor<4x32x8x2xf32>
  // CHECK-NEXT: return %[[REVERSE]] : tensor<4x32x8x2xf32>
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
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {}, {}]> : tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_concat_dim_is_sharded
func.func @concatenate_concat_dim_is_sharded(%arg0: tensor<8x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<8x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<8x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}, {}]> : tensor<8x32x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"x", "y"}, {}, {}]> : tensor<8x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %[[RESHARD2]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{"x"}, {"y"}, {}]> : tensor<8x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<8x32x256xf32>, tensor<8x48x256xf32>) -> tensor<8x80x256xf32>
  return %0 : tensor<8x80x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices
func.func @concatenate_operands_are_results_of_slices(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK-NOT: sdy.reshard
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_from_slices_of_the_same_tensor
func.func @concatenate_operands_are_from_slices_of_the_same_tensor(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x96x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg0 [0:4, 0:24, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x24x256xf32>
  // CHECK-NOT: sdy.reshard
  %2 = stablehlo.concatenate %0, %arg0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x40x256xf32>, tensor<4x24x256xf32>) -> tensor<4x96x256xf32>
  return %2 : tensor<4x96x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts
func.func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %0 <@mesh, [{}, {"x"}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %1 <@mesh, [{}, {"x"}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %[[RESHARD2]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x80x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [
{}, {"x"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts_but_conflicting_on_batching_dim
func.func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts_but_conflicting_on_batching_dim(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(2)2}, {}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %0 <@mesh, [{}, {"x"}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %1 <@mesh, [{}, {"x"}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %[[RESHARD2]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x80x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices_conflicting_shardings
func.func @concatenate_operands_are_results_of_slices_conflicting_shardings(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %1 <@mesh, [{}, {"y"}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %0, %[[RESHARD]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x80x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
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

// CHECK-LABEL: func @pad_both_input_and_output_permutation_dims_are_sharded_input_is_larger
// TODO(enver): Consider to prefer t over x along the batch dimension, even if it means two reshards.
func.func @pad_both_input_and_output_permutation_dims_are_sharded_input_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {"t"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y"}, {}, {"x"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyzt, [{}, {"y"}, {"t"}]> : tensor<30x26x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<30x26x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {"t"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_both_input_and_output_permutation_dims_are_sharded_output_is_larger
func.func @pad_both_input_and_output_permutation_dims_are_sharded_output_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x30x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {"t"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"y"}, {"t"}]> : tensor<28x28x16xf32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {"t"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  // CHECK-NEXT: return %[[PAD]] : tensor<30x30x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {"t"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  return %0 : tensor<30x30x16xf32>
}

// CHECK-LABEL: func @pad_same_permutation_dim_is_sharded_on_both_sides_input_is_larger
func.func @pad_same_permutation_dim_is_sharded_on_both_sides_input_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {}, {"y"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<30x26x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<30x26x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_same_permutation_dim_is_sharded_on_both_sides_output_is_larger
func.func @pad_same_permutation_dim_is_sharded_on_both_sides_output_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {}, {"y"}]>}, %arg1: tensor<f32>) -> (tensor<30x30x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<28x28x16xf32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  // CHECK-NEXT: return %[[PAD]] : tensor<30x30x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  return %0 : tensor<30x30x16xf32>
}

// CHECK-LABEL: func @pad_input_and_output_permutation_dims_are_sharded_same_way_input_is_larger
func.func @pad_input_and_output_permutation_dims_are_sharded_same_way_input_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"z"}, {}, {"y"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"z"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<30x26x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<30x26x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_input_and_output_permutation_dims_are_sharded_same_way_output_is_larger
func.func @pad_input_and_output_permutation_dims_are_sharded_same_way_output_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"z"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x30x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<28x28x16xf32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  // CHECK-NEXT: return %[[PAD]] : tensor<30x30x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  return %0 : tensor<30x30x16xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<32x1x2xf32> {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<32x1x2xf32>
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
  // TODO(enver): Consider preferring larger sharding axes along every batch and slice dimension.
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"x"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{}, {"z"}, {"y"}]> : tensor<4x8x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x8x128xf32>
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"z"}, {"y"}]>]>}: (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_both_operand_and_result_have_sharded_permutation_factors_result_has_larger_sharding_on_permutation_factor
func.func @slice_both_operand_and_result_have_sharded_permutation_factors_result_has_larger_sharding_on_permutation_factor(%arg0: tensor<2048x1152x192xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "y"}, {}, {"z"}]>}) -> (tensor<2048x1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "z"}, {"y"}, {}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {}, {"z"}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"x", "z"}, {"y"}, {}]> : tensor<2048x1024x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2048x1024x128xf32>
  %0 = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "z"}, {"y"}, {}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  return %0 : tensor<2048x1024x128xf32>
}

// CHECK-LABEL: func @slice_both_operand_and_result_have_sharded_permutation_factors_operand_has_larger_sharding_on_permutation_factor
func.func @slice_both_operand_and_result_have_sharded_permutation_factors_operand_has_larger_sharding_on_permutation_factor(%arg0: tensor<2048x1152x192xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "z"}, {}, {"y"}]>}) -> (tensor<2048x1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "y"}, {"z"}, {}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "z"}, {}, {"y"}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"x", "y"}, {"z"}, {}]> : tensor<2048x1024x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2048x1024x128xf32>
  %0 = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {"z"}, {}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  return %0 : tensor<2048x1024x128xf32>
}

// CHECK-LABEL: func @slice_input_and_output_permutation_dims_are_sharded_the_same_way
func.func @slice_input_and_output_permutation_dims_are_sharded_the_same_way(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) -> (tensor<16x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.slice %arg0 [8:24, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"y"}]>]>}: (tensor<32x64x128xf32>) -> tensor<16x8x128xf32>
  return %0 : tensor<16x8x128xf32>
}

// CHECK-LABEL: func @slice_input_and_output_permutation_dims_are_sharded_differently_but_nonempty_prefix
func.func @slice_input_and_output_permutation_dims_are_sharded_differently_but_nonempty_prefix(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z", "x"}, {}, {"y"}]>}) -> (tensor<16x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z", "y"}, {}, {}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [8:24, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z", "x"}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<16x8x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"z", "y"}, {}, {}]> : tensor<16x8x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<16x8x128xf32>
  %0 = stablehlo.slice %arg0 [8:24, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z", "y"}, {}, {}]>]>}: (tensor<32x64x128xf32>) -> tensor<16x8x128xf32>
  return %0 : tensor<16x8x128xf32>
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

// CHECK-LABEL: func @fft
func.func @fft(%arg0: tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y", "z"}, {}, {}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y", "z"}, {}, {}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  return %0 : tensor<128x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_inverse
func.func @fft_inverse(%arg0: tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) ->(tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y", "z"}, {}, {}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  IFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y", "z"}, {}, {}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = IFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  return %0 : tensor<128x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_real_truncated_result
func.func @fft_real_truncated_result(%arg0: tensor<128x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x33xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y"}, {}, {}]> : tensor<128x32x64xf32>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  RFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y"}, {}, {}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"p"}]> : tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x33xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = RFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  return %0 : tensor<128x32x33xcomplex<f32>>
}

// CHECK-LABEL: func @fft_inverse_real_expanded_result
func.func @fft_inverse_real_expanded_result(%arg0: tensor<128x32x33xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>}) -> (tensor<128x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y"}, {}, {}]> : tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  IRFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y"}, {}, {}]>]>} : (tensor<128x32x33xcomplex<f32>>) -> tensor<128x32x64xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xf32>
  %0  = stablehlo.fft %arg0, type = IRFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x33xcomplex<f32>>) -> tensor<128x32x64xf32>
  return %0 : tensor<128x32x64xf32>
}

// CHECK-LABEL: func @fft_small_batch_dimension
// TODO(enver): Subaxes of "z" should be distributed to the batching dimension.
func.func @fft_small_batch_dimension(%arg0: tensor<16x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<16x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y"}, {}, {}]> : tensor<16x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y"}, {}, {}]>]>} : (tensor<16x32x64xcomplex<f32>>) -> tensor<16x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<16x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<16x32x64xcomplex<f32>>) -> tensor<16x32x64xcomplex<f32>>
  return %0 : tensor<16x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_single_fft_dimension
func.func @fft_single_fft_dimension(%arg0: tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "z"}, {"y"}, {}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  FFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "z"}, {"y"}, {}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = FFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  return %0 : tensor<128x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_single_fft_dimension_real_truncated_result
func.func @fft_single_fft_dimension_real_truncated_result(%arg0: tensor<128x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x33xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x"}, {"y"}, {}]> : tensor<128x32x64xf32>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  RFFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"p"}]> : tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x33xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = RFFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  return %0 : tensor<128x32x33xcomplex<f32>>
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
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}, {}]> : tensor<16x2x4xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"y", "x"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_x_to_x_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_x_to_x_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_x_to_y_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_x_to_y_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{"y"}]> : tensor<32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_y_to_y_and_y_underfits_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_y_to_y_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_y_to_x_and_y_underfits_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_y_to_x_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_xy_to_xy_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_xy_to_xy_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_xy_to_yx_and_x_fits_exactly_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_xy_to_yx_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y", "x":(1)2}, {"x":(2)2}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_xy_to_x_and_x_fits_exactly_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_xy_to_x_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{"x"}]> : tensor<32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_sharding_is_from_yx_to_yx_and_y_underfits_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_sharding_is_from_yx_to_yx_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"y", "x":(1)2}, {"x":(2)2}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_yx_to_y_and_y_underfits_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_yx_to_y_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded
func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded_and_result_is_sharded
func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded_and_result_is_sharded(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_of_subaxes
func.func @reshape_simple_merge_sharding_is_from_of_subaxes(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<2x8xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_of_subaxes_only_result_is_sharded
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_of_subaxes_only_result_is_sharded(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]> : tensor<2x8xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<2x8xf32>) -> tensor<16xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<16xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<2x8xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_x_from_x_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_split_sharding_is_to_x_from_x_and_x_fits_exactly_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_y_from_y_and_y_underfits_to_first_dim
func.func @reshape_simple_split_sharding_is_to_y_from_y_and_y_underfits_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_xy_from_xy_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_split_sharding_is_to_xy_from_xy_and_x_fits_exactly_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_yx_from_yx_and_y_underfits_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_split_sharding_is_to_yx_from_yx_and_y_underfits_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{"y"}, {"x"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %1 : tensor<4x8xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_only_second_dim_is_sharded
func.func @reshape_simple_split_sharding_is_to_only_second_dim_is_sharded(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %1 : tensor<4x8xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_to_subaxes
func.func @reshape_simple_split_sharding_is_to_to_subaxes(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]>]>} : (tensor<16xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// CHECK-LABEL: func @reshape_strided_view_on_both_operand_and_result
func.func @reshape_strided_view_on_both_operand_and_result(%arg0: tensor<2x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<2x2x4xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<2x2x4xf32>) -> tensor<4x2x2xf32>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh, [{}, {}, {"y"}]> : tensor<4x2x2xf32>
  // CHECK-NEXT: return %2 : tensor<4x2x2xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"y"}]>]>} : (tensor<2x2x4xf32>) -> tensor<4x2x2xf32>
  return %0 : tensor<4x2x2xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_x_merged_dimensions_are_sharded
func.func @reshape_ij_k_to_i_jk_and_x_to_x_merged_dimensions_are_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_x_singleton_dimensions_are_sharded
func.func @reshape_ij_k_to_i_jk_and_x_to_x_singleton_dimensions_are_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_xy
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_yx
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_xy
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_yx
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_x
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_x(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_y
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_y(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_x
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_x(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_y
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_y(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_z_singleton_dims_are_sharded
func.func @reshape_ij_k_to_i_jk_and_x_to_z_singleton_dims_are_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh_xyz, [{"z"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_z_factor_j_is_sharded
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_x_to_z_factor_j_is_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {"y"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh_xyz, [{"z", "y"}, {"x"}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh_xyz, [{"z"}, {"y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %2 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_xy
func.func @reshape_ij_k_to_i_jk_and_x_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_yx
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_x_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %2 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_y_to_xy
func.func @reshape_ij_k_to_i_jk_and_y_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_y_to_yx
func.func @reshape_ij_k_to_i_jk_and_y_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
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

// CHECK-LABEL: func @custom_call_compact_wy_helper
func.func @custom_call_compact_wy_helper(%arg0: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<128x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<128x128xf32>
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @custom_call_inspect_sharding
func.func @custom_call_inspect_sharding(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @InspectSharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @InspectSharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @custom_call_x64_combine
func.func @custom_call_x64_combine(%arg0: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) -> (tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {"y"}]> : tensor<8x2xui32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64Combine(%arg0, %[[RESHARD]]) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  // CHECK-NEXT: return %[[CUSTOM_CALL]] : tensor<8x2xui64>
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}

// CHECK-LABEL: func @custom_call_x64_split_high
func.func @custom_call_x64_split_high(%arg0: tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xui32>
  %0 = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>}  : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_x64_split_low
func.func @custom_call_x64_split_low(%arg0: tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xui32>
  %0 = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_xla_megascale_provide_metadata
func.func @custom_call_xla_megascale_provide_metadata(%arg0: tensor<8x2xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @xla.megascale.provide_metadata(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xbf16>) -> tensor<8x2xbf16>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xbf16>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xbf16>
  %0 = stablehlo.custom_call @xla.megascale.provide_metadata(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>} : (tensor<8x2xbf16>) -> tensor<8x2xbf16>
  return %0 : tensor<8x2xbf16>
}

// CHECK-LABEL: func @custom_call_move_to_device
func.func @custom_call_move_to_device(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @MoveToDevice(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @MoveToDevice(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_move_to_host
func.func @custom_call_move_to_host(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @MoveToHost(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @MoveToHost(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_layout_constraint
func.func @custom_call_layout_constraint(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @LayoutConstraint(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_eigh
func.func @custom_call_eigh(%arg0: tensor<8x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<8x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, k]) {i=8, j=4, k=4}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {"y"}]> : tensor<8x4x4xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @Eigh(%[[RESHARD1]]) {backend_config = "1,1,100,1e-6", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{}, {"y"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[CUSTOM_CALL]]#0, %[[RESHARD2]] : tensor<8x4x4xf32>, tensor<8x4xf32>
  %0:2 = stablehlo.custom_call @Eigh(%arg0) {backend_config = "1,1,100,1e-6", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>)
  return %0#0, %0#1 : tensor<8x4x4xf32>, tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_qr
// TODO(enver): Actually the factors that need replication can be moved to batching dim.
func.func @custom_call_qr(%arg0: tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}) -> (tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}, tensor<8x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"p"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @Qr(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {}, {}]>, <@mesh_xpq, [{"x"}, {}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh_xpq, [{"x"}, {"q"}, {"p"}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh_xpq, [{"x"}, {"p"}]> : tensor<8x3xf32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<8x5x3xf32>, tensor<8x3xf32>
  %0:2 = stablehlo.custom_call @Qr(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>, <@mesh_xpq, [{"x"}, {"p"}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_qr_decomposition_block
func.func @custom_call_qr_decomposition_block(%arg0: tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}) -> (tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}, tensor<8x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"p"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @QrDecompositionBlock(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {}, {}]>, <@mesh_xpq, [{"x"}, {}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh_xpq, [{"x"}, {"q"}, {"p"}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh_xpq, [{"x"}, {"p"}]> : tensor<8x3xf32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<8x5x3xf32>, tensor<8x3xf32>
  %0:2 = stablehlo.custom_call @QrDecompositionBlock(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>, <@mesh_xpq, [{"x"}, {"p"}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_householder_product
func.func @custom_call_householder_product(%arg0: tensor<8x12x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>}, %arg1: tensor<8x5xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}]>}) -> (tensor<8x12x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, j, k]) {i=8, j=12, k=16, l=5} need_replication={j, k, l}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]> : tensor<8x12x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xpq, [{"x"}, {}]> : tensor<8x5xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%[[RESHARD1]], %[[RESHARD2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {}, {}]>]>} : (tensor<8x12x16xf32>, tensor<8x5xf32>) -> tensor<8x12x16xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]> : tensor<8x12x16xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x12x16xf32>
  %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>]>} : (tensor<8x12x16xf32>, tensor<8x5xf32>) -> tensor<8x12x16xf32>
  return %0 : tensor<8x12x16xf32>
}

// CHECK-LABEL: func @custom_call_erf
func.func @custom_call_erf(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @mhlo.erf(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @mhlo.erf (%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_topk_of_1d
func.func @custom_call_topk_of_1d(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, tensor<16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i], [i]) {i=16}>
  // CHECK: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%arg0)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{"x"}]>]>} : (tensor<16xf32>) -> (tensor<16xf32>, tensor<16xi32>)
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"y"}]> : tensor<16xf32>
  // CHECK-NEXT: return %[[RESHARD]], %[[CUSTOM_CALL]]#1 : tensor<16xf32>, tensor<16xi32>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 1 : i64,
        largest = true},
    mhlo.version = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>, <@mesh, [{"x"}]>]>}
    : (tensor<16xf32>) -> (tensor<16xf32>, tensor<16xi32>)
  return %0#0, %0#1 : tensor<16xf32>, tensor<16xi32>
}

// CHECK-LABEL: func @custom_call_topk_of_2d
func.func @custom_call_topk_of_2d(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<16x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<16x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} blocked_propagation={j}>
  // CHECK: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%arg0)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x8xf32>) -> (tensor<16x1xf32>, tensor<16x1xi32>)
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x"}, {}]> : tensor<16x1xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"y"}, {}]> : tensor<16x1xi32>
  // CHECK-NEXT: return %[[RESHARD1]], %[[RESHARD2]] : tensor<16x1xf32>, tensor<16x1xi32>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 1 : i64,
        largest = true},
    mhlo.version = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{"y"}, {}]>]>}
    : (tensor<16x8xf32>) -> (tensor<16x1xf32>, tensor<16x1xi32>)
  return %0#0, %0#1 : tensor<16x1xf32>, tensor<16x1xi32>
}

// CHECK-LABEL: func @custom_call_top2_of_2d
func.func @custom_call_top2_of_2d(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<16x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} blocked_propagation={j}>
  // CHECK: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%arg0)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %0#1 <@mesh, [{"y"}, {"x":(1)2}]> : tensor<16x2xi32>
  // CHECK-NEXT: return %[[CUSTOM_CALL]]#0, %[[RESHARD]] : tensor<16x2xf32>, tensor<16x2xi32>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 2 : i64,
        largest = true},
    mhlo.version = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>}
    : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xi32>
}

// CHECK-LABEL: func @custom_call_approx_topk
func.func @custom_call_approx_topk(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(2)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {"x":(2)2}]>
  // CHECK-NEXT: %[[APPROX_TOPK:.*]]:2 = stablehlo.custom_call @ApproxTopK(%[[RESHARD1]], %arg1, %arg2, %arg3)
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[APPROX_TOPK]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[APPROX_TOPK]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_approx_topk_majority_does_not_fit_all_factors
func.func @custom_call_approx_topk_majority_does_not_fit_all_factors(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(1)2}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[APPROX_TOPK:.*]]:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg1, %arg2, %arg3)
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[APPROX_TOPK]]#0 <@mesh, [{}, {"x":(1)2}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[APPROX_TOPK]]#1 <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x":(1)2}]>, <@mesh, [{}, {"y"}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_partial_reduce
func.func @custom_call_partial_reduce(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(2)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {"x":(2)2}]>
  // CHECK-NEXT: %[[PARTIAL_REDUCE:.*]]:2 = stablehlo.custom_call @PartialReduce(%[[RESHARD1]], %arg1, %arg2, %arg3)
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[PARTIAL_REDUCE]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[PARTIAL_REDUCE]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_partial_reduce_string_backend_config
func.func @custom_call_partial_reduce_string_backend_config(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, j], [i, j]) {i=16, j=4} blocked_propagation={j}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @PartialReduce(%arg0, %[[RESHARD1]], %arg2, %arg3)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    backend_config = "{\22log2_reduction\22: 5, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 2, \22recall_target\22: 0.950000}",
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @unregisterd_custom_call_with_existing_rule
func.func @unregisterd_custom_call_with_existing_rule(%arg0: tensor<4x2xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x4xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}){
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"x":(1)2}, {"y"}]> : tensor<2x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2x4xf32>
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @unregisterd_custom_call_without_existing_rule
func.func @unregisterd_custom_call_without_existing_rule(%arg0: tensor<4x2xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x4xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}){
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
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
func.func @manual_computation_with_manual_axes(%arg0: tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x","y"}]>}) -> (tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x","z"}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_xyzt, [{"x","y"}]>] out_shardings=[<@mesh_xyzt, [{"x", "z"}]>] manual_axes={"x"} (%arg1: tensor<52xf32>) {
    // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{"t"}]> : tensor<52xf32>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"t"}]>]>} : tensor<52xf32>
    // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ABS]] <@mesh_xyzt, [{"z"}]> : tensor<52xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD2]] : tensor<52xf32>
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh_xyzt, [{"t"}]>]>} : tensor<52xf32>
    sdy.return %2 : tensor<52xf32>
  } : (tensor<208xf32>) -> (tensor<208xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_xyzt, [{"x","z"}]>]>} : tensor<208xf32>
  return %1 : tensor<208xf32>
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

// CHECK-LABEL: func @negate_from_iota_unsharded_to_non_iota_unsharded
func.func @negate_from_iota_unsharded_to_non_iota_unsharded(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_non_iota_sharded_to_empty_sharding
func.func @negate_from_non_iota_sharded_to_empty_sharding(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) -> (tensor<210xf32>) {
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.negate %arg0 : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_axes_compatible_different_device_orders
func.func @negate_axes_compatible_different_device_orders(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_axes_incompatible_different_device_orders
func.func @negate_axes_incompatible_different_device_orders(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"y"}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"y"}]>]>} : tensor<210xf32>
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_axes_incompatible_different_device_orders_output_sharding_is_larger
func.func @negate_axes_incompatible_different_device_orders_output_sharding_is_larger(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD]]
  // CHECK-NEXT: return %[[NEGATE]]
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_same_axes_different_meshes
func.func @negate_same_axes_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_different_axes_different_meshes
func.func @negate_different_axes_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"y"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @dot_same_axes_different_meshes
func.func @dot_same_axes_different_meshes(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_same_axes_different_device_orders_lhs_and_result_majority
func.func @dot_same_axes_different_device_orders_lhs_and_result_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh_iota, [{}, {"y"}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]]
  // CHECK-NEXT: return %[[DOT]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_same_axes_different_device_orders_lhs_and_rhs_majority
func.func @dot_same_axes_different_device_orders_lhs_and_rhs_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {"y"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh_non_iota, [{"x"}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_different_axes_different_device_orders_lhs_and_result_majority
func.func @dot_different_axes_different_device_orders_lhs_and_result_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_iota, [{}, {"y"}]> : tensor<24x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh_iota, [{"y"}, {"x"}]> : tensor<6x12xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<6x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"y"}, {"x"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_different_axes_different_device_orders_lhs_and_rhs_majority
func.func @dot_different_axes_different_device_orders_lhs_and_rhs_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"y"}, {"x"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh_non_iota, [{"y"}, {"x"}]> : tensor<6x12xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<6x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"y"}, {"x"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @negate_identical_maximal_meshes
func.func @negate_identical_maximal_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_maximal, []>}) -> tensor<210xf32> {
  // CHECK-NOT: sdy.reshard
  // TODO(enver): Reshard to output mesh.
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_maximal_copy, []>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_different_maximal_meshes
func.func @negate_different_maximal_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_maximal, []>}) -> tensor<210xf32> {
  // CHECK-NOT: sdy.reshard
  // TODO(enver): Reshard to output mesh.
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_maximal_another, []>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @rng_bit_generator
func.func @rng_bit_generator(%arg0: tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> tensor<2xui64> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<2xui64>
  // CHECK-NEXT: %output_state, %output = stablehlo.rng_bit_generator %[[RESHARD1]], algorithm =  DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>, <@mesh, [{"y"}, {"x":(2)2}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %output_state <@mesh, [{"x":(1)2}]> : tensor<2xui64>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD2]]
  %0, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>, <@mesh, [{"y"}, {"x":(2)2}]>]>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x1000xui32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}]>]>} : tensor<2xui64>
  return %1 : tensor<2xui64>
}
