// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @batch_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16x16x8xf32>
// CHECK-SAME: -> (tensor<16x8x8x8xf32>
func.func @batch_sharded(
    %arg0: tensor<32x16x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>})
    -> (tensor<32x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // CHECK-NEXT: %[[RES:.*]] = "stablehlo.reduce_window"(%[[ARG0]], %[[CST]]) <{
  // CHECK-SAME:   padding = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
  // CHECK-SAME:   window_dimensions = array<i64: 1, 3, 3, 1>,
  // CHECK-SAME:   window_strides = array<i64: 1, 2, 2, 1>}> ({
  // CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[MAX:.*]] = stablehlo.maximum %[[ARG1]], %[[ARG2]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[MAX]] : tensor<f32>
  // CHECK-NEXT: })
  // CHECK-SAME: (tensor<16x16x16x8xf32>, tensor<f32>) -> tensor<16x8x8x8xf32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>
  }> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>}
  : (tensor<32x16x16x8xf32>, tensor<f32>) -> tensor<32x8x8x8xf32>

  // CHECK-NEXT: return %[[RES]] : tensor<16x8x8x8xf32>
  return %0 : tensor<32x8x8x8xf32>
}
