// RUN: sdy_opt %s -sdy-convert-global-to-local -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2 = <["x"=2]>

// CHECK-LABEL: func @not_sharded
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<16x4xf32>, %[[ARG1:.*]]: tensor<16x4xf32>) -> tensor<16x8xf32>
func.func @not_sharded(
  %arg0: tensor<16x4xf32>,
  %arg1: tensor<16x4xf32>)
  -> tensor<16x8xf32> {
  // CHECK: %[[RES:.*]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], dim = 1 : (tensor<16x4xf32>, tensor<16x4xf32>) -> tensor<16x8xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<16x4xf32>, tensor<16x4xf32>) -> tensor<16x8xf32>
  // CHECK: return %[[RES]] : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @sharded_non_concat_dim
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
// CHECK-SAME:     %[[ARG1:.*]]: tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
// CHECK-SAME:     %[[ARG2:.*]]: tensor<8x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
// CHECK-SAME:    -> (tensor<8x7xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
func.func @sharded_non_concat_dim(
  %arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg2: tensor<16x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<16x7xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // CHECK: %[[RES:.*]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], %[[ARG2]], dim = 1
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  // CHECK-SAME: : (tensor<8x4xf32>, tensor<8x2xf32>, tensor<8x1xf32>) -> tensor<8x7xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 1
  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<16x4xf32>, tensor<16x2xf32>, tensor<16x1xf32>) -> tensor<16x7xf32>
  // CHECK: return %[[RES]] : tensor<8x7xf32>
  return %0 : tensor<16x7xf32>
}

// -----

sdy.mesh @mesh_2 = <["x"=2]>

func.func @sharded_concat_dim(
  %arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // expected-error @+2 {{dimension 0 is sharded but the concatenation is on this dimension.}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.concatenate' that was explicitly marked illegal}}
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0
  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
