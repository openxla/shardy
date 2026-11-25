// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true mark-partial-result-with-unreduced-axes=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xt = <["x"=2, "t"=4]>
sdy.mesh @mesh_xyz = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_iota = <["x"=4, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=4, "y"=2], device_ids=[7, 6, 5, 4, 3, 2, 1, 0]>

// CHECK-LABEL: func @funcop_result_sharding_does_not_match
func.func @funcop_result_sharding_does_not_match(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  // CHECK: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_unsharded_but_different_meshes_between_return_and_func_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
func.func @funcop_result_unsharded_but_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_xt, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_matches_but_different_meshes_between_return_and_func_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @funcop_result_sharding_matches_but_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh_xt, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[RESHARD2]] <@mesh_xt, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}, {"t"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh_xt, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[RESHARD2]] <@mesh_xt, [{}, {"t"}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result_multiple_results
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"t"}, {}]>})
func.func @funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result_multiple_results(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"t"}, {}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}, {}]>}, tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD1_1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD1_2:.*]] = sdy.reshard %[[RESHARD1_1]] <@mesh_xt, [{}, {}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD1_3:.*]] = sdy.reshard %[[RESHARD1_2]] <@mesh_xt, [{"x"}, {}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD2_1:.*]] = sdy.reshard %[[ARG1]] <@mesh_xt, [{}, {}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[RESHARD2_2:.*]] = sdy.reshard %[[RESHARD2_1]] <@mesh, [{}, {}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[RESHARD2_3:.*]] = sdy.reshard %[[RESHARD2_2]] <@mesh, [{}, {"x"}]> : tensor<32x16xf32>
  // CHECK-NEXT: return %[[RESHARD1_3]], %[[RESHARD2_3]] : tensor<8x32xf32>, tensor<32x16xf32>
  return %arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>
}

// CHECK-LABEL: func @funcop_result_identical_sharding_but_different_meshes_between_return_and_func_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>})
func.func @funcop_result_identical_sharding_but_different_meshes_between_return_and_func_result(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}, {"t":(2)2}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh_xt, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[RESHARD2]] <@mesh_xt, [{"x"}, {"t":(2)2}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x16xf32>
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
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
func.func @funcop_result_sharding_does_not_match_multiple_results_different_meshes(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"x"}, {}]> : tensor<8x32xf32>
  // CHECK-NEXT: %[[RESHARD2_1:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[RESHARD2_2:.*]] = sdy.reshard %[[RESHARD2_1]] <@mesh_xyz, [{}, {}]> : tensor<32x16xf32>
  // CHECK-NEXT: %[[RESHARD2_3:.*]] = sdy.reshard %[[RESHARD2_2]] <@mesh_xyz, [{}, {"x"}]> : tensor<32x16xf32>
  // CHECK-NEXT: return %[[RESHARD1]], %[[RESHARD2_3]] : tensor<8x32xf32>, tensor<32x16xf32>
  return %arg0, %arg1 : tensor<8x32xf32>, tensor<32x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match_and_reshard_result_twice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
func.func @funcop_result_sharding_does_not_match_and_reshard_result_twice(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[ARG0]], %[[ARG1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"y"}>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_cross_mesh_unreduced
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @funcop_result_cross_mesh_unreduced(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}, {}], unreduced={"t"}>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh_xt, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"t"} %[[R1]] out_sharding=<@mesh_xt, [{}, {}], unreduced={"t"}> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[R2U]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_cross_mesh_unreduced_to_unreduced
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>})
func.func @funcop_result_cross_mesh_unreduced_to_unreduced(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}, {}], unreduced={"t"}>}) {
  // CHECK-NEXT: %[[AR:.*]] = sdy.all_reduce {"x"} %[[ARG0]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[AR]] <@mesh_xt, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"t"} %[[R]] out_sharding=<@mesh_xt, [{}, {}], unreduced={"t"}> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[R2U]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_same_axes_diff_ids_to_unreduced
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>})
func.func @funcop_result_same_axes_diff_ids_to_unreduced(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[MOVED:.*]] = sdy.reshard %[[ARG0]] <@mesh_non_iota, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[S2U:.*]] = sdy.sharded_to_unreduced [{"x"}, {}] %[[MOVED]] out_sharding=<@mesh_non_iota, [{}, {}], unreduced={"x"}> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[S2U]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}
