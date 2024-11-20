// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=4]>

// CHECK-LABEL: func @dot_compatible_ik
func.func @dot_compatible_ik(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
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
// CHECK-NEXT: %[[DOT:.*]]  = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @funcop_result_sharding_does_not_match_and_reshard_result_twice(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_subaxis_no_overlap
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x":(2)2}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)2}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"x":(2)2}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_subaxis_no_overlap(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(2)2}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x":(2)2}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_jk
func.func @dot_compatible_jk(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_k
func.func @dot_compatible_k(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_empty
func.func @dot_compatible_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_compatible_contracting_dim_empty
func.func @dot_compatible_contracting_dim_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_empty
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<8x16xf32> {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_i
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_non_contracting_dims_out_j
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_same_non_contracting_dims_out_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_out_empty
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_out_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_empty
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_i_empty(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_i
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_i(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_same_factor_for_contracting_dim_and_output_j
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {"y"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_same_factor_for_contracting_dim_and_output_j(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_i_mismatch
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_i_mismatch(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_lhs_non_contracting_dim_is_sharded(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_rhs_non_contracting_dim_is_sharded
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_incompatible_in_out_mismatch_same_axis_on_different_factors_rhs_non_contracting_dim_is_sharded(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_in_out_mismatch_i_j_swapped
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
// TODO(enver): A better solution could be to reshard the operands, if k-factor is small.
func.func @dot_incompatible_in_out_mismatch_i_j_swapped(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_sub_axis_overlaps
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
func.func @dot_incompatible_sub_axis_overlaps(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_incompatible_all_factors_mismatch
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]> : tensor<32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"y"}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD3]] : tensor<8x16xf32>
func.func @dot_incompatible_all_factors_mismatch(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_is_local
func.func @dot_reshard_is_local(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : tensor<32x16xf32>
// CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
// CHECK-NEXT: return %[[NEGATE]] : tensor<8x16xf32>
  %1 = stablehlo.dot %arg0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_reshard_does_not_change_input_sharding(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k], [k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
func.func @dot_reshard_does_not_change_input_sharding(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_without_sharding_rule
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT: return %[[RESHARD]] : tensor<8x16xf32>
func.func @dot_without_sharding_rule(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_incompatable_with_batching_dims
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh, [{}, {"x"}, {"y"}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_incompatable_with_batching_dims(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_incompatable_batching_factor_mismatch_on_all_tensors
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"z"}, {}, {}]> : tensor<4x8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"z"}, {}, {}]> : tensor<4x32x16xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %[[RESHARD2]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: return %[[DOTGENERAL]] : tensor<4x8x16xf32>
func.func @dot_genaral_incompatable_batching_factor_mismatch_on_all_tensors(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
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
// CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {}]> : tensor<8x32xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
// TODO(enver): A better solution would be to reshard LHS to [{},{"z":(1)2}] instead of fully replicate.
func.func @dot_multiple_axes_with_overlap_on_suffix(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z", "x":(2)2}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x", "y", "z":(2)2}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_is_strict_prefix_of_other
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y", "x":(1)2}, {}, {}]> : tensor<4x32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %arg0, %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x":(1)2}, {}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh_xyz, [{}, {}, {}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_one_is_strict_prefix_of_other(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x":(1)2}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}]>}) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_prefix_has_larger_count
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"y", "x"}, {}, {}]> : tensor<4x8x32xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x"}, {}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{"z"}, {}, {}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_one_prefix_has_larger_count(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x":(1)2}, {}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x"}, {}, {}]>}) ->(tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_one_suffix_has_larger_count_on_another_factor
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y", "x":(1)2}, {}, {}]> : tensor<4x32x16xf32>
// CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %arg0, %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y", "x":(1)2}, {"x":(2)2}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyz, [{}, {"x":(2)2}, {}]> : tensor<4x8x16xf32>
// CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8x16xf32>
func.func @dot_genaral_one_suffix_has_larger_count_on_another_factor(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x":(1)2}, {"x":(2)2}, {}]>}, %arg1: tensor<4x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y","x"}, {}, {}]>}) ->(tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x":(2)2}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"x":(2)2}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=4, j=8, k=16, l=32}>} : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @dot_genaral_batching_dimension_shardings_have_common_prefix
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y", "x":(1)2}, {"t":(2)2}, {}]> : tensor<64x8x32xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{"y", "x":(1)2}, {}, {"t":(1)2}]> : tensor<64x32x16xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[RESHARD1]], %[[RESHARD2]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y", "x":(1)2}, {"t":(2)2}, {"t":(1)2}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=64, j=8, k=16, l=32}>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
// CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[DOT]] <@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]> : tensor<64x8x16xf32>
// CHECK-NEXT: return %[[RESHARD3]] : tensor<64x8x16xf32>
func.func @dot_genaral_batching_dimension_shardings_have_common_prefix(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","x":(1)2,"t":(1)2}, {"t":(2)2}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y","x":(1)2,"t":(2)2}, {}, {"t":(1)2}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"t":(2)2}, {"t":(1)2}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, l], [i, l, k])->([i, j, k]) {i=64, j=8, k=16, l=32}>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
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
