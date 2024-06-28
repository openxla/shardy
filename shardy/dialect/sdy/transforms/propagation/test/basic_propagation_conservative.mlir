// RUN: sdy_opt %s -sdy-basic-propagate="conservative-propagation=true" 2>&1 | FileCheck %s

sdy.mesh @mesh_a_4_b_2 = <"a"=4, "b"=2>
sdy.mesh @mesh_a_2_b_8 = <"a"=2, "b"=8>
sdy.mesh @mesh_a_16_b_2 = <"a"=16, "b"=2>

// CHECK-LABEL: func @reshape_split_dim
func.func @reshape_split_dim(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a"}]>}) -> tensor<2x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  // CHECK-NOT: sdy.sharding
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @multi_axis_major_dim_uses_all_axes
func.func @multi_axis_major_dim_uses_all_axes(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4_b_2, [{"a", "b"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4_b_2, [{"a", "b", ?}, {?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<32xf32>) -> tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @multi_axis_major_dim_not_fully_sharded
func.func @multi_axis_major_dim_not_fully_sharded(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_8, [{"a", "b"}]>}) -> tensor<4x4xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_8, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @multi_axes_split_fully_sharded
func.func @multi_axes_split_fully_sharded(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a", "b"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.reshape %arg0 : (tensor<32xf32>) -> tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_a_16_b_2, [{"a"}, {}, {"b"}]>}) -> tensor<32x1x2xf32> {
  // CHECK-NEXT: stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_16_b_2, [{"a", ?}, {?}, {?}]>]>}
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}
