// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func private @pad_dynamic_update_slice_clamps_dynamic_start_index
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}, %[[UPD:.*]]: tensor<1x4xf32>, %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME: -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
func.func private @pad_dynamic_update_slice_clamps_dynamic_start_index(
  %arg0: tensor<7x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>},
  %upd: tensor<1x4xf32>, %idx0: tensor<i32>, %idx1: tensor<i32>)
  -> (tensor<7x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[MAX:.*]] = stablehlo.constant dense<6> : tensor<i32>
  // CHECK-NEXT: %[[CLAMPED:.*]] = stablehlo.minimum %[[IDX0]], %[[MAX]] : tensor<i32>
  // CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[UPD]], %[[CLAMPED]], %[[IDX1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x4xf32>
  // CHECK-NEXT: return %[[DUS]] : tensor<8x4xf32>
  %0 = stablehlo.dynamic_update_slice %arg0, %upd, %idx0, %idx1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x4xf32>, tensor<1x4xf32>, tensor<i32>, tensor<i32>) -> tensor<7x4xf32>
  return %0 : tensor<7x4xf32>
}

// CHECK-LABEL: func private @pad_dynamic_update_slice_clamps_constant_start_index
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}, %[[UPD:.*]]: tensor<2x4xf32>)
// CHECK-SAME: -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
func.func private @pad_dynamic_update_slice_clamps_constant_start_index(
  %arg0: tensor<7x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>},
  %upd: tensor<2x4xf32>)
  -> (tensor<7x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[C6:.*]] = sdy.constant dense<6> : tensor<i32>
  // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[C5:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[UPD]], %[[C5]], %[[C0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x4xf32>, tensor<2x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x4xf32>
  // CHECK-NEXT: return %[[DUS]] : tensor<8x4xf32>
  %c6 = sdy.constant dense<6> : tensor<i32>
  %c0 = sdy.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %upd, %c6, %c0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x4xf32>, tensor<2x4xf32>, tensor<i32>, tensor<i32>) -> tensor<7x4xf32>
  return %0 : tensor<7x4xf32>
}
