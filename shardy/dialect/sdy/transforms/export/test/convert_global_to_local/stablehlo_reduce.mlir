// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2 = <["x"=2]>
// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @stablehlo_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32x8xi32>)
// CHECK-SAME: -> tensor<32xi32> {
func.func @stablehlo_reduce(%arg0: tensor<32x8xi32>)
    -> (tensor<32xi32>) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.reduce(%[[ARG0]] init: %[[CST]]) across dimensions = [1] : (tensor<32x8xi32>, tensor<i32>) -> tensor<32xi32>
  // CHECK-NEXT:  reducer(%[[ARG1:.*]]: tensor<i32>, %[[ARG2:.*]]: tensor<i32>) {
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %[[ARG1]], %[[ARG2]] : tensor<i32>
  // CHECK-NEXT:    %[[ADD_2:.*]] = stablehlo.add %[[ADD]], %[[ARG2]] : tensor<i32>
  // CHECK-NEXT:    stablehlo.return %[[ADD_2]] : tensor<i32>
  // CHECK-NEXT:  }
  %cst = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce"(%arg0, %cst) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      %2 = stablehlo.add %1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) {
    dimensions = array<i64: 1>
  }: (tensor<32x8xi32>, tensor<i32>) -> tensor<32xi32>
  // CHECK-NEXT: return %[[RES]] : tensor<32xi32>
  return %0 : tensor<32xi32>
}

// CHECK-LABEL: func.func @stablehlo_reduce_unreduced_axes_fallback_all_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>})
func.func @stablehlo_reduce_unreduced_axes_fallback_all_reduce(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>}) {
  // CHECK-NEXT: %[[INIT:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[LOCAL_RED:.*]] = stablehlo.reduce(%[[ARG0]] init: %[[INIT]]) applies stablehlo.multiply across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}]>]>} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK:      %[[ALL_RED:.*]] = "stablehlo.all_reduce"(%[[LOCAL_RED]])
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
  // CHECK-SAME:   use_global_device_ids
  // CHECK:      return %[[ALL_RED]] : tensor<4xf32>
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}]>]>}
      : (tensor<8x16xf32>, tensor<f32>) -> tensor<8xf32>
    reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
      %mul = stablehlo.multiply %lhs, %rhs : tensor<f32>
      stablehlo.return %mul : tensor<f32>
    }
  return %0 : tensor<8xf32>
}
