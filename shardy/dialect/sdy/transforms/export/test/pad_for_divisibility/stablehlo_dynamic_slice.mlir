// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func private @pad_dynamic_slice_full_sharded_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}, {}]>}, %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>, %[[IDX2:.*]]: tensor<i32>)
// CHECK-SAME: -> (tensor<1x8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}, {}]>})
func.func private @pad_dynamic_slice_full_sharded_dim(
  %arg0: tensor<2x7x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}, {}]>},
  %idx0: tensor<i32>, %idx1: tensor<i32>, %idx2: tensor<i32>)
  -> (tensor<1x7x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}, {}]>}) {
  // CHECK-NEXT: %[[DS:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[IDX0]], %[[IDX1]], %[[IDX2]], sizes = [1, 8, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}, {}]>]>} : (tensor<2x8x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x8x4xf32>
  // CHECK-NEXT: return %[[DS]] : tensor<1x8x4xf32>
  %0 = stablehlo.dynamic_slice %arg0, %idx0, %idx1, %idx2, sizes = [1, 7, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}, {}]>]>} : (tensor<2x7x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x7x4xf32>
  return %0 : tensor<1x7x4xf32>
}

// CHECK-LABEL: func private @pad_dynamic_slice_clamps_start_index
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}, %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME: -> tensor<2x4xf32>
func.func private @pad_dynamic_slice_clamps_start_index(
  %arg0: tensor<7x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>},
  %idx0: tensor<i32>, %idx1: tensor<i32>)
  -> tensor<2x4xf32> {
  // CHECK-NEXT: %[[MAX:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-NEXT: %[[CLAMPED:.*]] = stablehlo.minimum %[[IDX0]], %[[MAX]] : tensor<i32>
  // CHECK-NEXT: %[[DS:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[CLAMPED]], %[[IDX1]], sizes = [2, 4] : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<2x4xf32>
  // CHECK-NEXT: return %[[DS]] : tensor<2x4xf32>
  %0 = stablehlo.dynamic_slice %arg0, %idx0, %idx1, sizes = [2, 4] : (tensor<7x4xf32>, tensor<i32>, tensor<i32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
