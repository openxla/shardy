// RUN: sdy_opt %s -allow-unregistered-dialect -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @select_and_scatter_batch_sharded
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<16x16x16x8xf32>
// CHECK-SAME: %[[SOURCE:.*]]: tensor<16x8x8x8xf32>
// CHECK-SAME: -> (tensor<16x16x16x8xf32>
func.func @select_and_scatter_batch_sharded(
    %arg0: tensor<32x16x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>},
    %arg1: tensor<32x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>})
    -> (tensor<32x16x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) {

  // CHECK-NEXT: %[[INIT:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>

  // CHECK-NEXT: %[[RES:.*]] = "stablehlo.select_and_scatter"(%[[OPERAND]], %[[SOURCE]], %[[INIT]]) <{
  // CHECK-SAME:   padding = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
  // CHECK-SAME:   window_dimensions = array<i64: 1, 3, 3, 1>,
  // CHECK-SAME:   window_strides = array<i64: 1, 2, 2, 1>}> ({
  // CHECK-NEXT: ^bb0(%[[SEL_ARG1:.*]]: tensor<f32>, %[[SEL_ARG2:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[CMP:.*]] = stablehlo.compare GE, %[[SEL_ARG1]], %[[SEL_ARG2]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return %[[CMP]] : tensor<i1>
  // CHECK-NEXT: }, {
  // CHECK-NEXT: ^bb0(%[[SCAT_ARG1:.*]]: tensor<f32>, %[[SCAT_ARG2:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[SCAT_ARG1]], %[[SCAT_ARG2]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: })
  // CHECK-SAME: : (tensor<16x16x16x8xf32>, tensor<16x8x8x8xf32>, tensor<f32>) -> tensor<16x16x16x8xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %init) <{
    padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>
  }> ({
  ^bb0(%sel_arg1: tensor<f32>, %sel_arg2: tensor<f32>):
    %cmp = stablehlo.compare GE, %sel_arg1, %sel_arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }, {
  ^bb0(%scat_arg1: tensor<f32>, %scat_arg2: tensor<f32>):
    %add = stablehlo.add %scat_arg1, %scat_arg2 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>}
  : (tensor<32x16x16x8xf32>, tensor<32x8x8x8xf32>, tensor<f32>) -> tensor<32x16x16x8xf32>

  // CHECK-NEXT: return %[[RES]] : tensor<16x16x16x8xf32>
  return %0 : tensor<32x16x16x8xf32>
}
