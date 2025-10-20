// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>

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
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<8x16xf32>
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
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x", "y", "z"} %[[DOTGENERAL]] out_sharding=<@mesh_xyz, [{}, {}]> : tensor<4x16xf32>
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

// CHECK-LABEL: func @dot_genaral_overlaps_and_trimmable_on_subaxis_multiple_axes
func.func @dot_genaral_overlaps_and_trimmable_on_subaxis_multiple_axes(%arg0: tensor<64x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}, %arg1: tensor<64x32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y":(2)2}, {}, {}]>}) ->(tensor<64x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"x","y","z"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{"y":(2)2}, {"x", "y":(1)2}, {}]> : tensor<64x8x32xf32>
  // CHECK-NEXT: %[[DOTGENERAL:.*]] = stablehlo.dot_general %[[RESHARD1]], %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y":(2)2}, {"x", "y":(1)2}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOTGENERAL]] <@mesh_xyzt, [{}, {"x", "y", "z"}, {}]> : tensor<64x8x16xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<64x8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"x","y","z"}, {}]>]>} : (tensor<64x8x32xf32>, tensor<64x32x16xf32>) -> tensor<64x8x16xf32>
  return %0 : tensor<64x8x16xf32>
}

// CHECK-LABEL: func @dot_only_contracting_dims_sharded_and_has_same_shardings
func.func @dot_only_contracting_dims_sharded_and_has_same_shardings(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {}]>
  // CHECK: return %[[ALL_REDUCE]]
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_on_square_matrices_lhs_2nd_dim_rhs_2nd_dim_sharded_the_same_way
func.func @dot_on_square_matrices_lhs_2nd_dim_rhs_2nd_dim_sharded_the_same_way(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dot_on_rectangular_inputs_square_output_small_contracting_dim_lhs_2nd_dim_rhs_2nd_dim_sharded_the_same_way
func.func @dot_on_rectangular_inputs_square_output_small_contracting_dim_lhs_2nd_dim_rhs_2nd_dim_sharded_the_same_way(
    %arg0: tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
    %arg1: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x2xf32>, tensor<2x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dot_on_rectangular_inputs_square_output_large_contracting_dim_lhs_2nd_dim_rhs_2nd_dim_sharded_the_same_way
func.func @dot_on_rectangular_inputs_square_output_large_contracting_dim_lhs_2nd_dim_rhs_2nd_dim_sharded_the_same_way(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
    %arg1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>})
    -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dot_on_rectangular_inputs_square_output_small_contracting_dim_lhs_2nd_dim_output_2nd_dim_sharded_the_same_way
func.func @dot_on_rectangular_inputs_square_output_small_contracting_dim_lhs_2nd_dim_output_2nd_dim_sharded_the_same_way(
    %arg0: tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
    %arg1: tensor<2x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: return %[[DOT]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x2xf32>, tensor<2x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dot_on_rectangular_inputs_square_output_large_contracting_dim_lhs_2nd_dim_output_2nd_dim_sharded_the_same_way
func.func @dot_on_rectangular_inputs_square_output_large_contracting_dim_lhs_2nd_dim_output_2nd_dim_sharded_the_same_way(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
    %arg1: tensor<16x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
