// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=4]>

// CHECK-LABEL: func @dot_compatible_ik
func.func @dot_compatible_ik(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @funcop_result_sharding_does_not_match
func.func @funcop_result_sharding_does_not_match(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
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
// CHECK-NEXT: %[[DOT:.*]]  = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @funcop_result_sharding_does_not_match_and_reshard_result_twice(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_subaxis_no_overlap
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x":(2)2}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"x":(2)2}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_subaxis_no_overlap(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(2)2}]>}) {
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
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<8x16xf32> {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_i
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_j
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_out_empty
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_empty
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_i_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_i
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_j
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"y"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_mismatch
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_i_mismatch(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_rhs_non_contracting_dim_is_sharded
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_rhs_non_contracting_dim_is_sharded(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_j_is_larger
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_j_is_larger(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_i_is_larger
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<16x8xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<16x8xf32>
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_i_is_larger(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_small_k
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}]> : tensor<128x4xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<4x256xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x4xf32>, tensor<4x256xf32>) -> tensor<128x256xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<128x256xf32>
// TODO(enver): A better solution could be to reshard the result.
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_small_k(%arg0: tensor<128x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x4xf32>, tensor<4x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped_large_k
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x1024xf32>, tensor<1024x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_i_j_swapped_large_k(%arg0: tensor<8x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<1024x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x1024xf32>, tensor<1024x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_sub_axis_overlaps
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x":(1)2}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(2)2}, {"x":(1)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_sub_axis_overlaps(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_all_factors_mismatch
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_all_factors_mismatch(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_all_factors_mismatch_small_k
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}]> : tensor<8x4xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<4x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_all_factors_mismatch_small_k(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_is_local
func.func @dot_reshard_is_local(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<32x16xf32>
// CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
// CHECK-NEXT: return %[[NEGATE]] : tensor<8x16xf32>
  %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_does_not_change_input_sharding(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
func.func @dot_reshard_does_not_change_input_sharding(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_with_sharding_rule
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_with_sharding_rule(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_incompatable_with_batching_dims
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh, [{}, {"x"}, {"y"}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_incompatable_with_batching_dims(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_incompatable_batching_factor_mismatch_on_all_tensors
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"y"}, {}, {}]> : tensor<4x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{"z"}, {}, {}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_incompatable_batching_factor_mismatch_on_all_tensors(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_multiple_axes
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"z"}, {"x", "y"}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_multiple_axes(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_multiple_axes_with_overlap
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {"z"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"z"}, {"x", "y"}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_multiple_axes_with_overlap(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z", "x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_multiple_axes_with_overlap_on_suffix
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {"z":(1)2}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"z":(1)2}, {"x", "y", "z":(2)2}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_multiple_axes_with_overlap_on_suffix(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z", "x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_is_strict_prefix_of_other
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y", "x":(1)2}, {}, {}]> : tensor<4x32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %arg0, %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x":(1)2}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh_xyz, [{}, {}, {}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_one_is_strict_prefix_of_other(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x":(1)2}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}]>}) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_prefix_has_larger_count
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"y", "x"}, {}, {}]> : tensor<4x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{"z"}, {}, {}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_one_prefix_has_larger_count(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x":(1)2}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x"}, {}, {}]>}) ->(tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_suffix_has_larger_count_on_another_factor
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y", "x":(1)2}, {}, {}]> : tensor<4x32x16xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x":(1)2}, {"x":(2)2}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{}, {"x":(2)2}, {}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_one_suffix_has_larger_count_on_another_factor(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x":(1)2}, {"x":(2)2}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x"}, {}, {}]>}) ->(tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x":(2)2}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x":(2)2}, {}]>]>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_batching_dimension_shardings_have_common_prefix
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y", "x":(1)2}, {"t":(2)2}, {}]> : tensor<64x8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{"y", "x":(1)2}, {}, {"t":(1)2}]> : tensor<64x32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[RESHARD1]], %[[RESHARD2]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y", "x":(1)2}, {"t":(2)2}, {"t":(1)2}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
// CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[DOT]] <@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]> : tensor<64x8x16xf32>
// CHECK-NEXT: return %[[RESHARD3]] : tensor<64x8x16xf32>
func.func @dot_genaral_batching_dimension_shardings_have_common_prefix(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","x":(1)2,"t":(1)2}, {"t":(2)2}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","x":(1)2,"t":(2)2}, {}, {"t":(1)2}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_one_is_strict_prefix_on_subaxes
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"x":(2)2}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x":(1)2}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh_xyz, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_one_is_strict_prefix_on_subaxes(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_and_trimmable
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y":(1)2}, {"x"}, {}]> : tensor<64x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(1)2}, {"x"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %1 <@mesh_xyzt, [{}, {"x", "y", "z"}, {}]> : tensor<64x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
func.func @dot_genaral_overlaps_and_trimmable(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"x","y","z"}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"x","y","z"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_from_most_major
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(1)2}, {}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"y"}, {}]> : tensor<64x8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<64x8x16xf32>
func.func @dot_genaral_overlaps_from_most_major(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(1)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_and_trimmable_on_subaxis
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y":(2)2}, {"y":(1)2}, {}]> : tensor<64x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(2)2}, {"y":(1)2}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"y"}, {}]> : tensor<64x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
func.func @dot_genaral_overlaps_and_trimmable_on_subaxis(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_overlaps_and_trimmable_on_subaxis
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y":(2)2}, {"x", "y":(1)2}, {}]> : tensor<64x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(2)2}, {"x", "y":(1)2}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"x", "y", "z"}, {}]> : tensor<64x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
func.func @dot_genaral_overlaps_and_trimmable_on_subaxis_multiple_axes(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"x","y","z"}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"x","y","z"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @cholesky
func.func @cholesky(%arg0: tensor<2x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> tensor<2x4x8x8xf32> {
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<2x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {}]> : tensor<2x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<2x4x8x8xf32>) -> tensor<2x4x8x8xf32>
  return %0 :  tensor<2x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_replicated_dim_is_sharded
func.func @cholesky_replicated_dim_is_sharded(%arg0: tensor<2x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {"y"}]>}) -> tensor<2x4x8x8xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.cholesky %arg0, lower = true : (tensor<2x4x8x8xf32>) -> tensor<2x4x8x8xf32>
  return %0 :  tensor<2x4x8x8xf32>
}

// CHECK-LABEL: func @reverse
func.func @reverse(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> tensor<4x32x8x2xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reverse %arg0, dims = [1, 3] : tensor<4x32x8x2xf32>
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
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4x2x2xui32>) -> tensor<4x2xui64>
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
func.func @bitcast_convert_downcast_casting_dim_is_sharded(%arg0: tensor<4x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x2x2xui32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}){
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.bitcast_convert %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} : (tensor<4x2xui64>) -> tensor<4x2x2xui32>
  return %0 :  tensor<4x2x2xui32>
}

// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%arg0: tensor<2x3x5x1x7xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}, {}]>}) -> tensor<2x5x3x11x7x13xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 1, 3, 4] : (tensor<2x3x5x1x7xf32>) -> tensor<2x5x3x11x7x13xf32>
  return %0 :  tensor<2x5x3x11x7x13xf32>
}

// CHECK-LABEL: func @concatenate_single_input
func.func @concatenate_single_input(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) -> tensor<4x32x256xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: stablehlo.concatenate %[[RESHARD]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>) -> tensor<4x32x256xf32>
  %0 = stablehlo.concatenate %arg0, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>) -> tensor<4x32x256xf32>
  return %0 : tensor<4x32x256xf32>
}

// CHECK-LABEL: func @concatenate
func.func @concatenate(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) -> tensor<4x80x256xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %arg0, %[[RESHARD1]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {}, {}]> : tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_replicated_dim_is_sharded
func.func @concatenate_replicated_dim_is_sharded(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<4x48x256xf32>) -> tensor<4x80x256xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @add
func.func @add(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32>) -> tensor<4x32xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.add %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<32x1x2xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2] : (tensor<32x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @dynamic_update_slice
func.func @dynamic_update_slice(%arg0: tensor<32x4x8xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<32x1x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<32x4x8xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"x"}]>}, %arg1: tensor<f32>) -> tensor<30x26x16xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<32x1x2xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @sort
func.func @sort(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>) {
  // CHECK-NOT: sdy.reshard
  %0:2 = "stablehlo.sort"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<4x32x8xi32>, tensor<4x32x8xf32>) -> (tensor<4x32x8xi32>, tensor<4x32x8xf32>)
  return %0#0, %0#1 : tensor<4x32x8xi32>, tensor<4x32x8xf32>
}

// CHECK-LABEL: func @sort_all_other_dims_size_one
func.func @sort_all_other_dims_size_one(%arg0: tensor<1x4x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<1x4x1xi32> {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 1 : i64, is_stable = true} : (tensor<1x4x1xi32>) -> tensor<1x4x1xi32>
  return %0 : tensor<1x4x1xi32>
}

// CHECK-LABEL: func @sort_single_input_output
func.func @sort_single_input_output(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x32x8xi32>) {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @sort_incompatible_on_nonsort_dimensions
func.func @sort_incompatible_on_nonsort_dimensions(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  // CHECK:  %[[SORT:.*]] = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}>
  // CHECK:  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}]>]>}
  // CHECK-NEXT:  %[[RESHARD:.*]] = sdy.reshard %[[SORT]] <@mesh, [{}, {"y"}, {}]> : tensor<4x32x8xi32>
  // CHECK-NEXT:  return %[[RESHARD]] : tensor<4x32x8xi32>
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
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


// CHECK-LABEL: func @sort_input_and_output_shardings_are_same_on_sorting_dimension
func.func @sort_input_and_output_shardings_are_same_on_sorting_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  // TODO(enver): Support cases that factors need replication and sharded in the same way, which still requires resharding since the sorting dimension is fully replicated.
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}


// CHECK-LABEL: func @sort_input_and_output_shardings_are_different_on_sorting_dimension
func.func @sort_input_and_output_shardings_are_different_on_sorting_dimension(%arg0: tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z"}, {}]>}) -> (tensor<4x32x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare GT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}, {}]>]>} : (tensor<4x32x8xi32>) -> (tensor<4x32x8xi32>)
  return %0 : tensor<4x32x8xi32>
}

// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<256x32x64x100xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}, {}]>}) -> tensor<100x32x256x64xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  return %0 : tensor<100x32x256x64xf32>
}

// CHECK-LABEL: func @triangular_solve
func.func @triangular_solve(%arg0: tensor<8x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: %[[TRIANGULAR_SOLVE:.*]] = "stablehlo.triangular_solve"(%arg0, %[[RESHARD1]]) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[TRIANGULAR_SOLVE]] <@mesh, [{}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x3x5xf32>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  return %0 : tensor<8x3x5xf32>
}

// CHECK-LABEL: func @triangular_solve_replicated_dim_is_sharded
func.func @triangular_solve_replicated_dim_is_sharded(%arg0: tensor<8x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK-NOT: sdy.reshard
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
func.func @reshape(%arg0: tensor<16x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

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

// CHECK-LABEL: func @reduce_single_result
func.func @reduce_single_result(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<2x13xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_multiple_results
func.func @reduce_multiple_results(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<2x64x13xi32>)
    -> (tensor<64xf32>, tensor<64xi32>) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
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
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @clamp_scalar_min_max
func.func @clamp_scalar_min_max(%arg0: tensor<f32>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<f32>) -> tensor<4x8xf32> {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
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
