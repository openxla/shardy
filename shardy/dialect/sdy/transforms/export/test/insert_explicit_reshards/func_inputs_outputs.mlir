// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xt = <["x"=2, "t"=4]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>

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
